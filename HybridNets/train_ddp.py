import argparse
import atexit
import datetime
import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val_ddp import val
from backbone import HybridNetsBackbone
from hybridnets.loss import FocalLoss
from utils.utils import get_last_weights, init_weights, boolean_string, save_checkpoint, Params
from utils.metrics_logging import append_jsonl
from hybridnets.dataset import BddDataset
from hybridnets.loss import FocalLossSeg, TverskyLoss
from hybridnets.autoanchor import run_anchor
from utils.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_checkpoint(path, rank):
    load_kwargs = {'map_location': f'cuda:{rank}'}
    try:
        return torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)


def average_tensor(tensor, world_size):
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def all_ranks_have_valid_loss(loss, device):
    valid = torch.isfinite(loss) & (loss.detach() != 0)
    valid_tensor = torch.tensor(1 if valid.item() else 0, device=device, dtype=torch.int32)
    dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
    return bool(valid_tensor.item())


@torch.no_grad()
def segmentation_metric_sums(segmentation, seg_annot, seg_mode):
    device = segmentation.device
    stats = torch.zeros(17, dtype=torch.float64, device=device)
    logits = segmentation.detach().float()
    target = seg_annot.detach().long().to(device)

    if seg_mode == MULTICLASS_MODE:
        probabilities = logits.softmax(dim=1)
        log_loss_sum = F.cross_entropy(logits, target, reduction='sum')
        true_class_probability = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        topk_count = min(2, probabilities.size(1))
        topk = probabilities.topk(topk_count, dim=1)
        confidence = topk.values[:, 0]
        margin = topk.values[:, 0] - topk.values[:, 1] if topk_count > 1 else topk.values[:, 0]
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        correct_pixels = topk.indices[:, 0] == target
    else:
        probabilities = torch.sigmoid(logits)
        target_float = target.float()
        log_loss_sum = F.binary_cross_entropy_with_logits(logits, target_float, reduction='sum')
        prediction = probabilities >= 0.5
        confidence = torch.where(prediction, probabilities, 1.0 - probabilities)
        margin = torch.abs(probabilities - 0.5) * 2.0
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
            + (1.0 - probabilities) * (1.0 - probabilities).clamp_min(1e-12).log()
        )
        true_class_probability = torch.where(target_float >= 0.5, probabilities, 1.0 - probabilities)
        correct_pixels = prediction.long() == target

    confidence_flat = confidence.detach().double().reshape(-1)
    margin_flat = margin.detach().double().reshape(-1)
    entropy_flat = entropy.detach().double().reshape(-1)
    correct_flat = correct_pixels.detach().reshape(-1)

    stats[0] = confidence_flat.sum()
    stats[1] = (confidence_flat ** 2).sum()
    stats[2] = entropy_flat.sum()
    stats[3] = margin_flat.sum()
    stats[4] = confidence_flat.numel()
    stats[5] = (confidence_flat < 0.5).sum()
    stats[6] = (confidence_flat < 0.7).sum()
    stats[7] = (confidence_flat < 0.9).sum()
    stats[8] = correct_flat.sum()
    if correct_flat.any():
        stats[9] = confidence_flat[correct_flat].sum()
        stats[10] = correct_flat.sum()
    incorrect_flat = ~correct_flat
    if incorrect_flat.any():
        stats[11] = confidence_flat[incorrect_flat].sum()
        stats[12] = incorrect_flat.sum()
    stats[13] = (entropy_flat ** 2).sum()
    stats[14] = log_loss_sum.detach().double()
    stats[15] = true_class_probability.numel()
    stats[16] = true_class_probability.detach().double().sum()
    return stats


def summarize_segmentation_metric_sums(stats):
    pixel_count = max(float(stats[4].item()), 1.0)
    correct_pixel_count = max(float(stats[10].item()), 1.0)
    incorrect_pixel_count = max(float(stats[12].item()), 1.0)
    log_loss_count = max(float(stats[15].item()), 1.0)

    confidence_mean = stats[0].item() / pixel_count
    confidence_variance = max(stats[1].item() / pixel_count - confidence_mean ** 2, 0.0)
    entropy_mean = stats[2].item() / pixel_count
    entropy_variance = max(stats[13].item() / pixel_count - entropy_mean ** 2, 0.0)
    return {
        'num_pixels': int(stats[4].item()),
        'pixel_accuracy': stats[8].item() / pixel_count,
        'segmentation_log_loss': stats[14].item() / log_loss_count,
        'mean_true_class_probability': stats[16].item() / log_loss_count,
        'confidence': {
            'mean': confidence_mean,
            'std': confidence_variance ** 0.5,
            'mean_on_correct_pixels': stats[9].item() / correct_pixel_count,
            'mean_on_incorrect_pixels': stats[11].item() / incorrect_pixel_count,
            'low_confidence_fraction_lt_0_5': stats[5].item() / pixel_count,
            'low_confidence_fraction_lt_0_7': stats[6].item() / pixel_count,
            'low_confidence_fraction_lt_0_9': stats[7].item() / pixel_count,
        },
        'uncertainty': {
            'mean_entropy': entropy_mean,
            'entropy_std': entropy_variance ** 0.5,
            'mean_class_margin': stats[3].item() / pixel_count,
        },
    }


def get_model_state(checkpoint):
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state, DDP):
        return state.module.model.state_dict()
    if isinstance(state, ModelWithLoss):
        return state.model.state_dict()
    if isinstance(state, nn.Module):
        return state.state_dict()
    return state


def bytes_to_gib(value):
    if value is None:
        return 'unknown'
    return f'{value / 1024 ** 3:.2f} GiB'


def read_cgroup_memory_value(paths):
    for path in paths:
        try:
            with open(path) as f:
                value = f.read().strip()
        except OSError:
            continue
        if value == 'max':
            return None
        try:
            return int(value)
        except ValueError:
            continue
    return None


def get_system_memory_info():
    page_size = os.sysconf('SC_PAGE_SIZE')
    total_pages = os.sysconf('SC_PHYS_PAGES')
    available_pages = os.sysconf('SC_AVPHYS_PAGES')
    return {
        'host_total': total_pages * page_size,
        'host_available': available_pages * page_size,
        'container_limit': read_cgroup_memory_value([
            '/sys/fs/cgroup/memory.max',
            '/sys/fs/cgroup/memory/memory.limit_in_bytes',
        ]),
        'container_used': read_cgroup_memory_value([
            '/sys/fs/cgroup/memory.current',
            '/sys/fs/cgroup/memory/memory.usage_in_bytes',
        ]),
    }


def print_resource_summary(visible_gpu_count):
    memory = get_system_memory_info()
    print('System RAM:')
    print(f"  Host total:       {bytes_to_gib(memory['host_total'])}")
    print(f"  Host available:   {bytes_to_gib(memory['host_available'])}")
    print(f"  Container limit:  {bytes_to_gib(memory['container_limit'])}")
    print(f"  Container used:   {bytes_to_gib(memory['container_used'])}")

    print('GPU VRAM:')
    if visible_gpu_count == 0:
        print('  No CUDA GPUs visible.')
        return
    for i in range(visible_gpu_count):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory
        try:
            with torch.cuda.device(i):
                free, runtime_total = torch.cuda.mem_get_info()
            used = runtime_total - free
            print(f'  GPU {i}: {props.name}')
            print(f'    Total: {bytes_to_gib(total)}')
            print(f'    Free:  {bytes_to_gib(free)}')
            print(f'    Used:  {bytes_to_gib(used)}')
        except RuntimeError as e:
            print(f'  GPU {i}: {props.name}')
            print(f'    Total: {bytes_to_gib(total)}')
            print(f'    Runtime VRAM check failed: {e}')
            
            

def print_gpu_memory_summary(label='', rank=None, all_gpus=False, reset_peak=False):
    if not torch.cuda.is_available():
        print(f'GPU memory summary{f" [{label}]" if label else ""}: CUDA is not available.')
        return

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        print(f'GPU memory summary{f" [{label}]" if label else ""}: no CUDA GPUs visible.')
        return

    if rank is not None and not all_gpus:
        device_ids = [rank]
    else:
        device_ids = list(range(visible_gpu_count))

    title = f'GPU memory summary [{label}]' if label else 'GPU memory summary'
    print(title)
    for device_id in device_ids:
        if device_id < 0 or device_id >= visible_gpu_count:
            print(f'  GPU {device_id}: not visible')
            continue

        try:
            torch.cuda.synchronize(device_id)
        except RuntimeError:
            pass

        props = torch.cuda.get_device_properties(device_id)
        try:
            with torch.cuda.device(device_id):
                free, total = torch.cuda.mem_get_info()
        except RuntimeError:
            free = None
            total = props.total_memory

        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        max_allocated = torch.cuda.max_memory_allocated(device_id)
        max_reserved = torch.cuda.max_memory_reserved(device_id)
        used = None if free is None else total - free

        print(f'  GPU {device_id}: {props.name}')
        print(f'    Total VRAM:      {bytes_to_gib(total)}')
        print(f'    Free VRAM:       {bytes_to_gib(free)}')
        print(f'    Runtime used:    {bytes_to_gib(used)}')
        print(f'    Torch allocated: {bytes_to_gib(allocated)}')
        print(f'    Torch reserved:  {bytes_to_gib(reserved)}')
        print(f'    Peak allocated:  {bytes_to_gib(max_allocated)}')
        print(f'    Peak reserved:   {bytes_to_gib(max_reserved)}')

        if reset_peak:
            torch.cuda.reset_peak_memory_stats(device_id)


def should_log_gpu_memory(opt, step, iteration):
    if not opt.gpu_memory_debug:
        return False
    if iteration == 0:
        return True
    return opt.gpu_memory_debug_interval > 0 and step % opt.gpu_memory_debug_interval == 0


def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('--log_path', type=str, default='checkpoints/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='checkpoints/')
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='Path for line-delimited JSON metrics. Defaults to <log_path>/<project>/tensorboard/metrics.jsonl')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='Whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    parser.add_argument('--cal_map', type=boolean_string, default=True,
                        help='Calculate mAP in validation')
    parser.add_argument('-v', '--verbose', type=boolean_string, default=True,
                        help='Whether to print results per class when valing')
    parser.add_argument('--plots', type=boolean_string, default=True,
                        help='Whether to plot confusion matrix when valing')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                        help='Confidence threshold for detection validation')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='NMS IoU threshold for detection validation')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use. Defaults to all visible CUDA GPUs.')
    parser.add_argument('--mosaic', type=boolean_string, default=False,
                        help='Use mosaic augmentation, '
                             'recommended when training object detection only.')
    parser.add_argument('--amp', type=boolean_string, default=False,
                        help='Automatic Mixed Precision training')
    parser.add_argument('--gpu_memory_debug', type=boolean_string, default=False,
                        help='Print GPU VRAM usage around major training memory changes.')
    parser.add_argument('--gpu_memory_debug_interval', type=int, default=0,
                        help='Print per-step GPU VRAM details every N steps. 0 prints only the first step of each epoch.')

    args = parser.parse_args()
    return args


def get_segmentation_mode(params):
    return MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE


class ModelWithLoss(nn.Module):
    def __init__(self, model, seg_mode, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.seg_criterion1 = TverskyLoss(mode=seg_mode, alpha=0.7, beta=0.3, gamma=4.0 / 3, from_logits=True)
        self.seg_criterion2 = FocalLossSeg(mode=seg_mode, alpha=0.25)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, seg_annot, obj_list=None, skip_detection_loss=False, skip_seg_loss=False):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        if skip_detection_loss:
            zero = segmentation.new_zeros(1)
            cls_loss = zero
            reg_loss = zero
        elif self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)

        if skip_seg_loss:
            seg_loss = segmentation.new_zeros(1)
        else:
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)
            seg_loss = tversky_loss + 1 * focal_loss

        return cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation


def train(rank, opt):
    print("Training process started for rank:", rank)
    torch.cuda.set_device(rank)
    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} start', rank=rank, reset_peak=True)

    project_path = os.path.join(SCRIPT_DIR, 'projects', f'{opt.project}.yml')
    if not os.path.exists(project_path):
        project_path = os.path.join(SCRIPT_DIR, 'projects', f'{opt.project}.yaml')
    params = Params(project_path)
    if not opt.mosaic:
        params.dataset['mosaic'] = 0.0
        params.dataset['mixup'] = 0.0
        print(f'[Info] rank {rank} disabled mosaic and mixup from CLI --mosaic False')
    if opt.freeze_det and opt.freeze_seg:
        raise ValueError('Cannot freeze both detection and segmentation heads: no active task loss would remain.')

    torch.cuda.manual_seed(69)
    torch.manual_seed(69)

    seg_mode = get_segmentation_mode(params)
    print(f'[Info] rank {rank} using segmentation mode: {seg_mode}')

    train_dataloader, val_dataloader = prepare(rank, params, opt, seg_mode)

    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} before model create', rank=rank, reset_peak=True)
    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone, seg_mode=seg_mode)
    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} after model create', rank=rank)

    # load last weights
    ckpt = {}
    # last_step = None
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        # try:
        #     last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        # except:
        #     last_step = 0

        try:
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} before checkpoint load', rank=rank, reset_peak=True)
            ckpt = load_checkpoint(weights_path, rank)
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} after checkpoint load', rank=rank)
            model.load_state_dict(get_model_state(ckpt), strict=False)
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} after checkpoint state_dict load', rank=rank)
        except (RuntimeError, TypeError) as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing non-encoder weights...')
        init_weights(model.bifpn)
        init_weights(model.bifpndecoder)
        init_weights(model.segmentation_head)
        init_weights(model.regressor)
        init_weights(model.classifier)

    print('[Info] Successfully!!!')

    if opt.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        print('[Info] freezed backbone')

    if opt.freeze_det:
        model.regressor.requires_grad_(False)
        model.classifier.requires_grad_(False)
        model.anchors.requires_grad_(False)
        print('[Info] freezed detection head')

    if opt.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        print('[Info] freezed segmentation head')

    writer = None
    if rank == 0:
        writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # wrap the model with loss function, to reduce the memory usage on gpu0 and speedup
    setup(rank, opt.num_gpus)
    model = ModelWithLoss(model, seg_mode=seg_mode, debug=opt.debug)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} before model.to(cuda)', rank=rank, reset_peak=True)
    model = model.to(rank, memory_format=torch.channels_last)
    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} after model.to(cuda)', rank=rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    if opt.gpu_memory_debug:
        print_gpu_memory_summary(f'rank {rank} after DDP wrap', rank=rank)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError('No trainable parameters remain after applying freeze options.')

    if opt.optim == 'adamw':
        optimizer = ZeroRedundancyOptimizer(
            trainable_params,
            optimizer_class=torch.optim.AdamW,
            lr=opt.lr
        )
    else:
        optimizer = ZeroRedundancyOptimizer(
            trainable_params,
            optimizer_class=torch.optim.SGD,
            lr=opt.lr,
            momentum=0.9,
            nesterov=True
        )
    scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt.get('step', 0) if isinstance(ckpt, dict) else 0
    best_fitness = ckpt.get('best_fitness', 0) if isinstance(ckpt, dict) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(train_dataloader)
    try:
        for epoch in range(opt.num_epochs):
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} before epoch {epoch}', rank=rank, reset_peak=True)
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            epoch_cls_loss = []
            epoch_reg_loss = []
            epoch_seg_loss = []
            epoch_segmentation_metric_sums = torch.zeros(17, dtype=torch.float64, device=rank)
            train_dataloader.sampler.set_epoch(epoch)
            progress_bar = tqdm(train_dataloader, ascii=True, disable=rank != 0)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    # print("WTF")
                    log_gpu_memory = should_log_gpu_memory(opt, step, iter)
                    if log_gpu_memory:
                        print_gpu_memory_summary(
                            f'rank {rank} step {step} before batch to cuda',
                            rank=rank,
                            reset_peak=True
                        )
                    imgs = data['img'].to(rank, non_blocking=params.pin_memory,
                                          memory_format=torch.channels_last)
                    annot = data['annot'].to(rank, non_blocking=params.pin_memory)
                    seg_annot = data['segmentation'].to(rank, non_blocking=params.pin_memory)
                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} after batch to cuda', rank=rank)

                    optimizer.zero_grad(set_to_none=True)
                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} before forward', rank=rank)
                    with torch.cuda.amp.autocast(enabled=opt.amp):
                        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(
                            imgs,
                            annot,
                            seg_annot,
                            obj_list=params.obj_list,
                            skip_detection_loss=opt.freeze_det,
                            skip_seg_loss=opt.freeze_seg,
                        )
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()
                        seg_loss = seg_loss.mean()
                        loss = cls_loss + reg_loss + seg_loss
                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} after forward', rank=rank)
                    if not all_ranks_have_valid_loss(loss, rank):
                        continue

                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} before backward', rank=rank)
                    scaler.scale(loss).backward()
                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} after backward', rank=rank)
                    scaler.step(optimizer)
                    scaler.update()
                    if log_gpu_memory:
                        print_gpu_memory_summary(f'rank {rank} step {step} after optimizer step', rank=rank)

                    epoch_loss.append(loss.detach().item())
                    epoch_cls_loss.append(cls_loss.detach().item())
                    epoch_reg_loss.append(reg_loss.detach().item())
                    epoch_seg_loss.append(seg_loss.detach().item())
                    epoch_segmentation_metric_sums += segmentation_metric_sums(segmentation, seg_annot, seg_mode)

                    avg_loss = average_tensor(loss, opt.num_gpus)
                    avg_cls_loss = average_tensor(cls_loss, opt.num_gpus)
                    avg_reg_loss = average_tensor(reg_loss, opt.num_gpus)
                    avg_seg_loss = average_tensor(seg_loss, opt.num_gpus)

                    if rank == 0:
                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, avg_cls_loss.item(),
                                avg_reg_loss.item(), avg_seg_loss.item(), avg_loss.item()))
                        writer.add_scalars('Loss', {'train': avg_loss.item()}, step)
                        writer.add_scalars('Regression_loss', {'train': avg_reg_loss.item()}, step)
                        writer.add_scalars('Classfication_loss', {'train': avg_cls_loss.item()}, step)
                        writer.add_scalars('Segmentation_loss', {'train': avg_seg_loss.item()}, step)

                        # log learning_rate
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0 and rank == 0:
                        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    raise

            train_loss_sums = torch.tensor(
                [
                    float(np.sum(epoch_loss)),
                    float(np.sum(epoch_cls_loss)),
                    float(np.sum(epoch_reg_loss)),
                    float(np.sum(epoch_seg_loss)),
                    float(len(epoch_loss)),
                ],
                dtype=torch.float64,
                device=rank,
            )
            dist.all_reduce(train_loss_sums, op=dist.ReduceOp.SUM)
            train_count = max(float(train_loss_sums[4].item()), 1.0)
            train_loss = train_loss_sums[0].item() / train_count
            train_cls_loss = train_loss_sums[1].item() / train_count
            train_reg_loss = train_loss_sums[2].item() / train_count
            train_seg_loss = train_loss_sums[3].item() / train_count
            train_optimizer_steps = int(train_count / max(opt.num_gpus, 1))
            dist.all_reduce(epoch_segmentation_metric_sums, op=dist.ReduceOp.SUM)
            train_segmentation_metrics = summarize_segmentation_metric_sums(epoch_segmentation_metric_sums)
            scheduler.step(train_loss)

            if rank == 0:
                train_metrics = {
                    'phase': 'train',
                    'project': opt.project,
                    'epoch': epoch,
                    'step': step,
                    'loss': train_loss,
                    'classification_loss': train_cls_loss,
                    'regression_loss': train_reg_loss,
                    'segmentation_loss': train_seg_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'num_batches': train_optimizer_steps,
                    'num_rank_batches': int(train_count),
                    'batch_size_per_rank': opt.batch_size,
                    'global_batch_size': opt.batch_size * opt.num_gpus,
                    'num_gpus': opt.num_gpus,
                    'num_workers_per_rank': opt.num_workers,
                    'freeze_backbone': opt.freeze_backbone,
                    'freeze_det': opt.freeze_det,
                    'freeze_seg': opt.freeze_seg,
                    'mosaic': params.dataset['mosaic'],
                    'mixup': params.dataset['mixup'],
                    'amp': opt.amp,
                    'conf_thres': opt.conf_thres,
                    'iou_thres': opt.iou_thres,
                    'segmentation_mode': seg_mode,
                    'segmentation_classes': ['background', *params.seg_list] if seg_mode != BINARY_MODE else params.seg_list,
                    'detection_classes': params.obj_list,
                }
                train_metrics.update(train_segmentation_metrics)
                append_jsonl(opt.metrics_path, train_metrics)

            if epoch % opt.val_interval == 0:
                if opt.gpu_memory_debug:
                    print_gpu_memory_summary(f'rank {rank} before validation epoch {epoch}', rank=rank, reset_peak=True)
                best_fitness, best_loss, best_epoch = val(model, rank, optimizer, val_dataloader, params, opt, writer, epoch,
                                                          step, best_fitness, best_loss, best_epoch, seg_mode)
                if opt.gpu_memory_debug:
                    print_gpu_memory_summary(f'rank {rank} after validation epoch {epoch}', rank=rank)
    except KeyboardInterrupt:
        if rank == 0:
            save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
    finally:
        if writer is not None:
            writer.close()
        cleanup_dist()


def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '23456')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    atexit.register(cleanup_dist)

def prepare(rank, params, opt, seg_mode):
    
    print(f"inputsize: {params.model['image_size']}")
    print(f"mean: {params.mean}, std: {params.std}")
    
    print("Making Train Dataset")
    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug,
        lazy_load_labels=True
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=opt.num_gpus, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=params.pin_memory, num_workers=opt.num_workers,
                                    drop_last=True, shuffle=False, sampler=train_sampler, collate_fn=BddDataset.collate_fn)
    
    print("Making Val Dataset")
    val_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=opt.debug,
        lazy_load_labels=True
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=opt.num_gpus, rank=rank, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, pin_memory=params.pin_memory, num_workers=opt.num_workers,
                                    drop_last=False, shuffle=False, sampler=val_sampler, collate_fn=BddDataset.collate_fn)
    
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    print("Starting training...")
    opt = get_args()
    print("Arguments parsed.")
    print(opt)
    visible_gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {visible_gpu_count}")
    print_resource_summary(visible_gpu_count)
    print_gpu_memory_summary('startup all visible GPUs', all_gpus=True)
    if opt.num_gpus is None:
        opt.num_gpus = visible_gpu_count
    if opt.num_gpus < 1:
        raise SystemExit('train_ddp.py requires at least one CUDA GPU. Use train.py for CPU/single-process training.')
    if visible_gpu_count < opt.num_gpus:
        raise SystemExit(
            f'Requested --num_gpus {opt.num_gpus}, but only {visible_gpu_count} CUDA GPU(s) are visible.'
        )

    print(f"Using {opt.num_gpus} GPU(s): {list(range(opt.num_gpus))}")
    opt.saved_path = opt.saved_path + f'/{opt.project}/'
    print(f"Model checkpoints will be saved to: {opt.saved_path}")
    opt.log_path = opt.log_path + f'/{opt.project}/tensorboard/'
    print(f"Tensorboard logs will be saved to: {opt.log_path}")
    os.makedirs(opt.log_path, exist_ok=True)
    if opt.metrics_path is None:
        opt.metrics_path = os.path.join(opt.log_path, 'metrics.jsonl')
    print(f"JSON metrics will be saved to: {opt.metrics_path}")
    print(f"Ensuring checkpoint directory exists: {opt.saved_path}")
    os.makedirs(opt.saved_path, exist_ok=True)
    print("Setup complete. Spawning processes for training...")
    print(1)
    mp.spawn(
        train,
        args=(opt,),
        nprocs=opt.num_gpus
    )
