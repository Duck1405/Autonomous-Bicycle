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
from utils.metrics_logging import append_jsonl, standardize_metric_record
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

from hybridnets.model import ModelWithLoss

from train_functions import (get_args, should_log_gpu_memory, 
                             get_model_state, cleanup_dist, 
                             load_checkpoint, average_tensor,
                             all_ranks_have_valid_loss, 
                             segmentation_metric_sums,
                             segmentation_confusion_metric_sums,
                             summarize_segmentation_metric_sums,
                             summarize_segmentation_confusion_sums, bytes_to_gib,
                             read_cgroup_memory_value,
                             get_system_memory_info,
                             SystemUtilizationSampler,
                             print_resource_summary, 
                             print_gpu_memory_summary)




SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE
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
   
        try:
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} before checkpoint load', rank=rank, reset_peak=True)
            ckpt = load_checkpoint(weights_path, rank)
            if opt.gpu_memory_debug:
                print_gpu_memory_summary(f'rank {rank} after checkpoint load', rank=rank)
            model.load_state_dict(get_model_state(ckpt, ModelWithLoss), strict=False)
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
    model = ModelWithLoss(model, debug=opt.debug)
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
            segmentation_class_names = ['background', *params.seg_list] if seg_mode != BINARY_MODE else params.seg_list
            num_segmentation_classes = len(segmentation_class_names)
            epoch_segmentation_metric_sums = torch.zeros(17, dtype=torch.float64, device=rank)
            epoch_segmentation_confusion_sums = torch.zeros(
                num_segmentation_classes,
                4,
                dtype=torch.float64,
                device=rank,
            )
            train_dataloader.sampler.set_epoch(epoch)
            system_utilization_sampler = SystemUtilizationSampler(num_iter_per_epoch, max_samples=100) if rank == 0 else None
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
                    epoch_segmentation_confusion_sums += segmentation_confusion_metric_sums(
                        segmentation,
                        seg_annot,
                        seg_mode,
                        num_segmentation_classes,
                    )

                    avg_loss = average_tensor(loss, opt.num_gpus)
                    avg_cls_loss = average_tensor(cls_loss, opt.num_gpus)
                    avg_reg_loss = average_tensor(reg_loss, opt.num_gpus)
                    avg_seg_loss = average_tensor(seg_loss, opt.num_gpus)

                    if rank == 0:
                        progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                                step, epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, avg_cls_loss.item(),
                                avg_reg_loss.item(), avg_seg_loss.item(), avg_loss.item()))
                        writer.add_scalars('Loss', {'train': avg_loss.item()}, step)
                        writer.add_scalars('Regression_loss', {'train': avg_reg_loss.item()}, step)
                        writer.add_scalars('Classfication_loss', {'train': avg_cls_loss.item()}, step)
                        writer.add_scalars('Segmentation_loss', {'train': avg_seg_loss.item()}, step)

                        # log learning_rate
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalars('Learning_rate', {'train': current_lr}, step)
                        system_utilization_sampler.maybe_sample(iter)
                    step += 1

                    if step % opt.save_interval == 0 and step > 0 and rank == 0:
                        checkpoint_name = f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth'
                        save_checkpoint(model, opt.saved_path, checkpoint_name)
                        print(f'checkpoint saved: {checkpoint_name} (run name: {opt.name or "unnamed"})')

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
            dist.all_reduce(epoch_segmentation_confusion_sums, op=dist.ReduceOp.SUM)
            train_segmentation_metrics = summarize_segmentation_metric_sums(epoch_segmentation_metric_sums)
            train_segmentation_metrics.update(
                summarize_segmentation_confusion_sums(
                    epoch_segmentation_confusion_sums,
                    segmentation_class_names,
                )
            )
            scheduler.step(train_loss)

            if rank == 0:
                system_utilization_metrics = system_utilization_sampler.summarize()
                train_metrics = {
                    'phase': 'train',
                    # 'run_name': opt.name,
                    # 'project': opt.project,
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
                    # 'freeze_backbone': opt.freeze_backbone,
                    # 'freeze_det': opt.freeze_det,
                    # 'freeze_seg': opt.freeze_seg,
                    'mosaic': params.dataset['mosaic'],
                    'mixup': params.dataset['mixup'],
                    'amp': opt.amp,
                    'conf_thres': opt.conf_thres,
                    'iou_thres': opt.iou_thres,
                    'segmentation_mode': seg_mode,
                    'segmentation_classes': segmentation_class_names,
                    # 'detection_classes': params.obj_list,
                }
                train_metrics.update(system_utilization_metrics)
                train_metrics.update(train_segmentation_metrics)
                append_jsonl(opt.metrics_path, standardize_metric_record(train_metrics))

            if epoch % opt.val_interval == 0:
                if opt.gpu_memory_debug:
                    print_gpu_memory_summary(f'rank {rank} before validation epoch {epoch}', rank=rank, reset_peak=True)
                best_fitness, best_loss, best_epoch = val(model, rank, optimizer, val_dataloader, params, opt, writer, epoch,
                                                          step, best_fitness, best_loss, best_epoch, seg_mode)
                if opt.gpu_memory_debug:
                    print_gpu_memory_summary(f'rank {rank} after validation epoch {epoch}', rank=rank)

            if rank == 0:
                epoch_ckpt = {
                    'run_name': opt.name,
                    'epoch': epoch,
                    'step': step,
                    'best_fitness': best_fitness,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'model': model.module.model.state_dict(),
                }
                save_checkpoint(
                    epoch_ckpt,
                    opt.saved_path,
                    f'hybridnets-d{opt.compound_coef}_epoch_{epoch + 1}_{step}.pth'
                )
                print(
                    f'checkpoint saved for epoch {epoch + 1}: '
                    f'hybridnets-d{opt.compound_coef}_epoch_{epoch + 1}_{step}.pth '
                    f'(run name: {opt.name or "unnamed"})'
                )
    except KeyboardInterrupt:
        if rank == 0:
            checkpoint_name = f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth'
            save_checkpoint(model, opt.saved_path, checkpoint_name)
            print(f'checkpoint saved after interrupt: {checkpoint_name} (run name: {opt.name or "unnamed"})')
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
    print(f"Run/job name: {opt.name or 'unnamed'}")
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
