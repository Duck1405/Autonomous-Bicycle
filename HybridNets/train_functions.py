import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import atexit
import datetime
import os
import traceback

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from utils import smp_metrics
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



from utils.utils import get_last_weights, init_weights, boolean_string, \
    save_checkpoint, DataLoaderX, Params

def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('--name', type=str, default=None,
                        help='Optional run/job name to print and store with checkpoints')
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
    parser.add_argument('--save_interval', type=int, default=999999999999, help='Number of steps between saving')
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

def should_log_gpu_memory(opt, step, iteration):
    if not opt.gpu_memory_debug:
        return False
    if iteration == 0:
        return True
    return opt.gpu_memory_debug_interval > 0 and step % opt.gpu_memory_debug_interval == 0


def get_model_state(checkpoint, ModelWithLoss):
    state = checkpoint.get('model', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state, DDP):
        return state.module.model.state_dict()
    if isinstance(state, ModelWithLoss):
        return state.model.state_dict()
    if isinstance(state, nn.Module):
        return state.state_dict()
    return state

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


@torch.no_grad()
def segmentation_confusion_metric_sums(segmentation, seg_annot, seg_mode, num_classes):
    device = segmentation.device
    logits = segmentation.detach().float()
    target = seg_annot.detach().long().to(device)

    if seg_mode == MULTICLASS_MODE:
        prediction = logits.softmax(dim=1).argmax(dim=1)
    else:
        prediction = torch.sigmoid(logits)

    tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(
        prediction,
        target,
        mode=seg_mode,
        threshold=0.5 if seg_mode != MULTICLASS_MODE else None,
        num_classes=num_classes if seg_mode == MULTICLASS_MODE else None,
    )
    return torch.stack(
        [
            tp_seg.sum(0),
            fp_seg.sum(0),
            fn_seg.sum(0),
            tn_seg.sum(0),
        ],
        dim=1,
    ).to(device=device, dtype=torch.float64)


def summarize_segmentation_confusion_sums(stats, class_names):
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        true_positive = stats[i, 0].item()
        false_positive = stats[i, 1].item()
        false_negative = stats[i, 2].item()
        true_negative = stats[i, 3].item()
        predicted_pixels = true_positive + false_positive
        target_pixels = true_positive + false_negative

        precision = true_positive / max(predicted_pixels, 1.0)
        recall = true_positive / max(target_pixels, 1.0)
        f1_score = (2.0 * precision * recall) / max(precision + recall, 1e-12)
        iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
        true_negative_rate = true_negative / max(true_negative + false_positive, 1.0)

        per_class_metrics[name] = {
            'iou': iou,
            'balanced_accuracy': 0.5 * (recall + true_negative_rate),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'dice': f1_score,
            'target_pixels': int(target_pixels),
            'predicted_pixels': int(predicted_pixels),
            'true_positive_pixels': int(true_positive),
            'false_positive_pixels': int(false_positive),
            'false_negative_pixels': int(false_negative),
            'true_negative_pixels': int(true_negative),
        }
    return {'per_class': per_class_metrics}


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


def bytes_to_gib_value(value):
    if value is None:
        return None
    return value / 1024 ** 3


def collect_system_utilization_metrics(process=None):
    if process is None:
        process = psutil.Process(os.getpid())
    system_cpu_percent = psutil.cpu_percent(interval=None)
    process_cpu_percent = process.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    process_memory = process.memory_info()
    memory = get_system_memory_info()
    container_limit = memory['container_limit']
    container_used = memory['container_used']
    container_ram_percent = None
    if container_limit:
        container_ram_percent = (container_used or 0) / container_limit * 100.0

    return {
        'system_cpu_percent': system_cpu_percent,
        'process_cpu_percent': process_cpu_percent,
        'cpu_logical_count': psutil.cpu_count(logical=True),
        'cpu_physical_count': psutil.cpu_count(logical=False),
        'system_ram_percent': ram.percent,
        'system_ram_used_gib': bytes_to_gib_value(ram.used),
        'system_ram_available_gib': bytes_to_gib_value(ram.available),
        'system_ram_total_gib': bytes_to_gib_value(ram.total),
        'process_ram_rss_gib': bytes_to_gib_value(process_memory.rss),
        'container_ram_percent': container_ram_percent,
        'container_ram_used_gib': bytes_to_gib_value(container_used),
        'container_ram_limit_gib': bytes_to_gib_value(container_limit),
    }


class SystemUtilizationSampler:
    def __init__(self, total_steps, max_samples=100):
        self.max_samples = max(1, int(max_samples))
        self.total_steps = max(1, int(total_steps))
        self.sample_interval = max(1, (self.total_steps + self.max_samples - 1) // self.max_samples)
        self.samples = []
        self.process = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)

    def maybe_sample(self, iteration):
        if len(self.samples) >= self.max_samples:
            return
        if iteration % self.sample_interval == 0:
            self.samples.append(collect_system_utilization_metrics(self.process))

    def summarize(self):
        if not self.samples:
            self.samples.append(collect_system_utilization_metrics(self.process))

        summary = {
            'system_utilization_sample_count': len(self.samples),
            'system_utilization_sample_interval_batches': self.sample_interval,
        }
        metric_keys = [
            'system_cpu_percent',
            'process_cpu_percent',
            'system_ram_percent',
            'system_ram_used_gib',
            'system_ram_available_gib',
            'process_ram_rss_gib',
            'container_ram_percent',
            'container_ram_used_gib',
        ]
        for key in metric_keys:
            values = [sample[key] for sample in self.samples if sample.get(key) is not None]
            if not values:
                summary[f'{key}_mean'] = None
                summary[f'{key}_max'] = None
                continue
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_max'] = float(np.max(values))

        last_sample = self.samples[-1]
        summary.update(last_sample)
        summary.update({
            'cpu_logical_count': last_sample['cpu_logical_count'],
            'cpu_physical_count': last_sample['cpu_physical_count'],
            'system_ram_total_gib': last_sample['system_ram_total_gib'],
            'container_ram_limit_gib': last_sample['container_ram_limit_gib'],
        })
        return summary


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
