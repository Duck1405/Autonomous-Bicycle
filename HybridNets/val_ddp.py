import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from utils import smp_metrics
from utils.constants import BINARY_MODE, MULTICLASS_MODE
from utils.metrics_logging import append_jsonl, standardize_metric_record
from utils.utils import (
    BBoxTransform,
    ClipBoxes,
    ConfusionMatrix,
    ap_per_class,
    fitness,
    postprocess,
    process_batch,
    save_checkpoint,
    scale_coords,
)


@torch.no_grad()
def val(model, rank, optimizer, val_generator, params, opt, writer, epoch, step, best_fitness, best_loss, best_epoch,
        seg_mode):
    model.eval()
    device = torch.device(f'cuda:{rank}')

    loss_classification_ls = []
    loss_regression_ls = []
    loss_segmentation_ls = []
    stats = []

    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    ncs = 1 if seg_mode == BINARY_MODE else len(params.seg_list) + 1

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    header = ('%15s' + '%11s' * 14) % (
        'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95',
        'mIoU', 'mAcc', 'fIoU', 'sIoU', 'rIoU', 'rAcc', 'lIoU', 'lAcc'
    )
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(ncs)]
    f1_ls = [[] for _ in range(ncs)]
    confidence_stats = torch.zeros(14, dtype=torch.float64, device=device)
    segmentation_log_loss_stats = torch.zeros(3, dtype=torch.float64, device=device)
    segmentation_confusion_stats = torch.zeros(ncs, 4, dtype=torch.float64, device=device)
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    progress_bar = tqdm(val_generator, ascii=True, disable=rank != 0)
    for _, data in enumerate(progress_bar):
        imgs = data['img'].to(rank, non_blocking=params.pin_memory, memory_format=torch.channels_last)
        annot = data['annot'].to(rank, non_blocking=params.pin_memory)
        seg_annot = data['segmentation'].to(rank, non_blocking=params.pin_memory)
        shapes = data['shapes']

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
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

        if opt.cal_map:
            out = postprocess(
                imgs.detach(),
                torch.stack([anchors[0]] * imgs.shape[0], 0).detach(),
                regression.detach(),
                classification.detach(),
                regressBoxes,
                clipBoxes,
                opt.conf_thres,
                opt.iou_thres,
            )

            for i in range(annot.size(0)):
                seen += 1
                labels = annot[i]
                labels = labels[labels[:, 4] != -1]

                output = out[i]
                nl = len(labels)
                pred = np.column_stack([output['rois'], output['scores']])
                pred = np.column_stack([pred, output['class_ids']])
                pred = torch.from_numpy(pred).to(device)

                target_class = labels[:, 4].tolist() if nl else []
                if len(pred) == 0:
                    if nl:
                        stats.append((
                            torch.zeros(0, num_thresholds, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            target_class,
                        ))
                    continue

                if nl:
                    input_shape = imgs.shape[2:]
                    pred[:, :4] = scale_coords(input_shape, pred[:, :4], shapes[i][0], shapes[i][1])
                    labels = scale_coords(input_shape, labels, shapes[i][0], shapes[i][1])
                    correct = process_batch(pred, labels, iou_thresholds)
                    if opt.plots:
                        confusion_matrix.process_batch(pred, labels)
                else:
                    correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool, device=device)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))
        else:
            seen += annot.size(0)

        seg_target = seg_annot.long().to(device)
        segmentation_logits = segmentation.float()
        if seg_mode == MULTICLASS_MODE:
            probabilities = segmentation_logits.log_softmax(dim=1).exp()
            log_loss_sum = F.cross_entropy(segmentation_logits, seg_target, reduction='sum')
            true_class_probability = probabilities.gather(1, seg_target.unsqueeze(1)).squeeze(1)
            topk_count = min(2, probabilities.size(1))
            topk = probabilities.topk(topk_count, dim=1)
            confidence = topk.values[:, 0]
            margin = topk.values[:, 0] - topk.values[:, 1] if topk_count > 1 else topk.values[:, 0]
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            segmentation_for_stats = topk.indices[:, 0]
            correct_pixels = segmentation_for_stats == seg_target
        else:
            probabilities = torch.sigmoid(segmentation_logits)
            target_float = seg_target.float()
            log_loss_sum = F.binary_cross_entropy_with_logits(segmentation_logits, target_float, reduction='sum')
            prediction = probabilities >= 0.5
            confidence = torch.where(prediction, probabilities, 1.0 - probabilities)
            margin = torch.abs(probabilities - 0.5) * 2.0
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
                + (1.0 - probabilities) * (1.0 - probabilities).clamp_min(1e-12).log()
            )
            true_class_probability = torch.where(target_float >= 0.5, probabilities, 1.0 - probabilities)
            correct_pixels = prediction.long() == seg_target
            segmentation_for_stats = probabilities

        segmentation_log_loss_stats[0] += log_loss_sum.detach().double()
        segmentation_log_loss_stats[1] += true_class_probability.numel()
        segmentation_log_loss_stats[2] += true_class_probability.detach().double().sum()

        confidence_flat = confidence.detach().double().reshape(-1)
        margin_flat = margin.detach().double().reshape(-1)
        entropy_flat = entropy.detach().double().reshape(-1)
        correct_flat = correct_pixels.detach().reshape(-1)
        confidence_stats[0] += confidence_flat.sum()
        confidence_stats[1] += (confidence_flat ** 2).sum()
        confidence_stats[2] += entropy_flat.sum()
        confidence_stats[3] += margin_flat.sum()
        confidence_stats[4] += confidence_flat.numel()
        confidence_stats[5] += (confidence_flat < 0.5).sum()
        confidence_stats[6] += (confidence_flat < 0.7).sum()
        confidence_stats[7] += (confidence_flat < 0.9).sum()
        confidence_stats[8] += correct_flat.sum()
        if correct_flat.any():
            confidence_stats[9] += confidence_flat[correct_flat].sum()
            confidence_stats[10] += correct_flat.sum()
        incorrect_flat = ~correct_flat
        if incorrect_flat.any():
            confidence_stats[11] += confidence_flat[incorrect_flat].sum()
            confidence_stats[12] += incorrect_flat.sum()
        confidence_stats[13] += (entropy_flat ** 2).sum()

        tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(
            segmentation_for_stats,
            seg_target,
            mode=seg_mode,
            threshold=0.5 if seg_mode != MULTICLASS_MODE else None,
            num_classes=ncs if seg_mode == MULTICLASS_MODE else None,
        )
        segmentation_confusion_stats += torch.stack(
            [
                tp_seg.sum(0),
                fp_seg.sum(0),
                fn_seg.sum(0),
                tn_seg.sum(0),
            ],
            dim=1,
        ).to(device=device, dtype=torch.float64)
        iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        f1 = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        for i in range(ncs):
            iou_ls[i].append(iou.T[i].detach().cpu().numpy())
            f1_ls[i].append(f1.T[i].detach().cpu().numpy())

    loss_sums = torch.tensor(
        [
            float(np.sum(loss_classification_ls)),
            float(np.sum(loss_regression_ls)),
            float(np.sum(loss_segmentation_ls)),
            float(len(loss_segmentation_ls)),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
    loss_count = max(float(loss_sums[3].item()), 1.0)
    cls_loss = loss_sums[0].item() / loss_count
    reg_loss = loss_sums[1].item() / loss_count
    seg_loss = loss_sums[2].item() / loss_count
    loss = cls_loss + reg_loss + seg_loss

    seen_tensor = torch.tensor(seen, dtype=torch.long, device=device)
    dist.all_reduce(seen_tensor, op=dist.ReduceOp.SUM)
    seen = int(seen_tensor.item())
    dist.all_reduce(confidence_stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(segmentation_log_loss_stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(segmentation_confusion_stats, op=dist.ReduceOp.SUM)

    ddp_stats = [None for _ in range(opt.num_gpus)] if rank == 0 else None
    ddp_iou = [None for _ in range(opt.num_gpus)] if rank == 0 else None
    ddp_f1 = [None for _ in range(opt.num_gpus)] if rank == 0 else None
    dist.gather_object(stats, ddp_stats, dst=0)
    dist.gather_object(iou_ls, ddp_iou, dst=0)
    dist.gather_object(f1_ls, ddp_f1, dst=0)

    if rank == 0:
        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, opt.num_epochs, cls_loss, reg_loss, seg_loss, loss
            )
        )
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        writer.add_scalars('Segmentation_loss', {'val': seg_loss}, step)

        gathered_iou = [[] for _ in range(ncs)]
        gathered_f1 = [[] for _ in range(ncs)]
        for rank_iou in ddp_iou:
            for i in range(ncs):
                gathered_iou[i].extend(rank_iou[i])
        for rank_f1 in ddp_f1:
            for i in range(ncs):
                gathered_f1[i].extend(rank_f1[i])

        for i in range(ncs):
            gathered_iou[i] = np.concatenate(gathered_iou[i]) if gathered_iou[i] else np.array([0.0])
            gathered_f1[i] = np.concatenate(gathered_f1[i]) if gathered_f1[i] else np.array([0.0])

        iou_score = np.mean(gathered_iou)
        f1_score = np.mean(gathered_f1)
        iou_first_decoder = np.mean((gathered_iou[0] + gathered_iou[1]) / 2.0) if ncs > 1 else np.mean(gathered_iou[0])
        iou_second_decoder = np.mean((gathered_iou[0] + gathered_iou[2]) / 2.0) if ncs > 2 else np.mean(gathered_iou[0])

        for i in range(ncs):
            gathered_iou[i] = np.mean(gathered_iou[i])
            gathered_f1[i] = np.mean(gathered_f1[i])

        class_names = ['background', *params.seg_list] if seg_mode != BINARY_MODE else params.seg_list
        per_class_metrics = {}
        for i in range(ncs):
            name = class_names[i] if i < len(class_names) else f'class_{i}'
            true_positive = segmentation_confusion_stats[i, 0].item()
            false_positive = segmentation_confusion_stats[i, 1].item()
            false_negative = segmentation_confusion_stats[i, 2].item()
            true_negative = segmentation_confusion_stats[i, 3].item()
            predicted_pixels = true_positive + false_positive
            target_pixels = true_positive + false_negative
            f1_score = (2.0 * true_positive) / max(2.0 * true_positive + false_positive + false_negative, 1.0)
            per_class_metrics[name] = {
                'iou': float(gathered_iou[i]),
                'balanced_accuracy': float(gathered_f1[i]),
                'precision': true_positive / max(predicted_pixels, 1.0),
                'recall': true_positive / max(target_pixels, 1.0),
                'f1_score': f1_score,
                'dice': f1_score,
                'target_pixels': int(target_pixels),
                'predicted_pixels': int(predicted_pixels),
                'true_positive_pixels': int(true_positive),
                'false_positive_pixels': int(false_positive),
                'false_negative_pixels': int(false_negative),
                'true_negative_pixels': int(true_negative),
            }

        pixel_count = max(float(confidence_stats[4].item()), 1.0)
        log_loss_count = max(float(segmentation_log_loss_stats[1].item()), 1.0)
        confidence_mean = confidence_stats[0].item() / pixel_count
        confidence_variance = max(confidence_stats[1].item() / pixel_count - confidence_mean ** 2, 0.0)
        entropy_mean = confidence_stats[2].item() / pixel_count
        entropy_variance = max(confidence_stats[13].item() / pixel_count - entropy_mean ** 2, 0.0)
        correct_pixel_count = max(float(confidence_stats[10].item()), 1.0)
        incorrect_pixel_count = max(float(confidence_stats[12].item()), 1.0)
        validation_metrics = {
            'phase': 'val',
            'run_name': opt.name,
            'project': opt.project,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'classification_loss': cls_loss,
            'regression_loss': reg_loss,
            'segmentation_loss': seg_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_batches': len(val_generator),
            'num_rank_batches': len(val_generator),
            'num_images': seen,
            'num_pixels': int(confidence_stats[4].item()),
            'batch_size_per_rank': opt.batch_size,
            'global_batch_size': opt.batch_size * opt.num_gpus,
            'num_gpus': opt.num_gpus,
            'num_workers_per_rank': opt.num_workers,
            'freeze_backbone': opt.freeze_backbone,
            'freeze_det': opt.freeze_det,
            'freeze_seg': opt.freeze_seg,
            'mosaic': params.dataset.get('mosaic'),
            'mixup': params.dataset.get('mixup'),
            'amp': opt.amp,
            'pixel_accuracy': confidence_stats[8].item() / pixel_count,
            'segmentation_log_loss': segmentation_log_loss_stats[0].item() / log_loss_count,
            'mean_true_class_probability': segmentation_log_loss_stats[2].item() / log_loss_count,
            'mean_iou': float(iou_score),
            'mean_balanced_accuracy': float(f1_score),
            'foreground_background_iou': float(iou_first_decoder),
            'lane_background_iou': float(iou_second_decoder),
            'confidence': {
                'mean': confidence_mean,
                'std': confidence_variance ** 0.5,
                'mean_on_correct_pixels': confidence_stats[9].item() / correct_pixel_count,
                'mean_on_incorrect_pixels': confidence_stats[11].item() / incorrect_pixel_count,
                'low_confidence_fraction_lt_0_5': confidence_stats[5].item() / pixel_count,
                'low_confidence_fraction_lt_0_7': confidence_stats[6].item() / pixel_count,
                'low_confidence_fraction_lt_0_9': confidence_stats[7].item() / pixel_count,
            },
            'uncertainty': {
                'mean_entropy': entropy_mean,
                'entropy_std': entropy_variance ** 0.5,
                'mean_class_margin': confidence_stats[3].item() / pixel_count,
            },
            'per_class': per_class_metrics,
            'cal_map_enabled': opt.cal_map,
            'conf_thres': opt.conf_thres,
            'iou_thres': opt.iou_thres,
            'segmentation_mode': seg_mode,
            'segmentation_classes': class_names,
            'detection_classes': params.obj_list,
        }

        nt = torch.zeros(1)
        ap_class = []
        if opt.cal_map:
            flat_stats = [x for rank_stats in ddp_stats for x in rank_stats]
            stats = [np.concatenate(x, 0) for x in zip(*flat_stats)] if flat_stats else []

            save_dir = 'plots'
            os.makedirs(save_dir, exist_ok=True)
            if len(stats) and stats[0].any():
                p, r, f1, ap, ap_class = ap_per_class(*stats, plot=opt.plots, save_dir=save_dir, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=1)

            if opt.plots:
                confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
                confusion_matrix.tp_fp()

        print(header)
        row_format = '%15s' + '%11i' * 2 + '%11.3g' * 12
        print(row_format % (
            'all',
            seen,
            nt.sum(),
            mp,
            mr,
            map50,
            map,
            iou_score,
            f1_score,
            iou_first_decoder,
            iou_second_decoder,
            gathered_iou[1] if ncs > 1 else 0.0,
            gathered_f1[1] if ncs > 1 else 0.0,
            gathered_iou[2] if ncs > 2 else 0.0,
            gathered_f1[2] if ncs > 2 else 0.0,
        ))

        if opt.cal_map and (opt.verbose or nc < 50) and nc > 1 and len(ap_class):
            class_format = '%15s' + '%11i' * 2 + '%11.3g' * 4
            for i, c in enumerate(ap_class):
                print(class_format % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        if opt.cal_map:
            results = (mp, mr, map50, map, iou_score, f1_score, loss)
            detection_per_class = {}
            for i, c in enumerate(ap_class):
                detection_per_class[names[c]] = {
                    'precision': float(p[i]),
                    'recall': float(r[i]),
                    'ap50': float(ap50[i]),
                    'ap50_95': float(ap[i]),
                    'targets': int(nt[c]) if c < len(nt) else 0,
                }
            validation_metrics['detection'] = {
                'precision': float(mp),
                'recall': float(mr),
                'map50': float(map50),
                'map50_95': float(map),
                'num_targets': int(nt.sum()),
                'per_class': detection_per_class,
            }
            fi = fitness(np.array(results).reshape(1, -1))
            validation_metrics['fitness'] = float(fi[0])
            best_fitness_value = float(best_fitness[0] if hasattr(best_fitness, '__len__') else best_fitness)
            if float(fi[0]) > best_fitness_value:
                best_fitness = fi
                best_epoch = epoch
                ckpt = {
                    'run_name': opt.name,
                    'epoch': epoch,
                    'step': step,
                    'best_fitness': best_fitness,
                    'model': model.module.model.state_dict(),
                }
                checkpoint_name = f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth'
                print(f'Saving checkpoint with best fitness {fi[0]}: {checkpoint_name} (run name: {opt.name or "unnamed"})')
                save_checkpoint(ckpt, opt.saved_path, checkpoint_name)
        elif loss + opt.es_min_delta < best_loss:
            best_loss = loss
            best_epoch = epoch
            checkpoint_name = f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth'
            save_checkpoint(model, opt.saved_path, checkpoint_name)
            print(f'Saving checkpoint with best loss {best_loss}: {checkpoint_name} (run name: {opt.name or "unnamed"})')

        validation_metrics['best_loss'] = float(best_loss)
        validation_metrics['best_epoch'] = int(best_epoch)
        validation_metrics['best_fitness'] = float(best_fitness[0] if hasattr(best_fitness, '__len__') else best_fitness)
        append_jsonl(opt.metrics_path, standardize_metric_record(validation_metrics))

    best_loss_epoch = torch.tensor([float(best_loss), float(best_epoch)], dtype=torch.float64, device=device)
    dist.broadcast(best_loss_epoch, src=0)
    best_loss = best_loss_epoch[0].item()
    best_epoch = int(best_loss_epoch[1].item())

    if epoch - best_epoch > opt.es_patience > 0:
        if rank == 0:
            print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
            writer.close()
        raise SystemExit(0)

    model.train()
    return best_fitness, best_loss, best_epoch
