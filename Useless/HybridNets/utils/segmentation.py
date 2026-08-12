import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from utils.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE


def segmentation_logits_to_probabilities(segmentation_logits, seg_mode):
    """Convert segmentation logits to per-pixel probabilities."""
    if seg_mode == MULTICLASS_MODE:
        return segmentation_logits.softmax(dim=1)

    if seg_mode == BINARY_MODE:
        foreground = torch.sigmoid(segmentation_logits)
        return torch.cat([1.0 - foreground, foreground], dim=1)

    if seg_mode == MULTILABEL_MODE:
        return torch.sigmoid(segmentation_logits)

    raise ValueError(f'Unsupported segmentation mode: {seg_mode}')


def segmentation_probabilities_to_predictions(probabilities, seg_mode, threshold=0.5):
    """Return the selected segmentation class/labels and confidence per pixel."""
    if seg_mode in {BINARY_MODE, MULTICLASS_MODE}:
        confidence, prediction = probabilities.max(dim=1)
        return prediction, confidence

    if seg_mode == MULTILABEL_MODE:
        prediction = probabilities >= threshold
        confidence = torch.where(prediction, probabilities, 1.0 - probabilities)
        return prediction, confidence

    raise ValueError(f'Unsupported segmentation mode: {seg_mode}')


def resize_segmentation_probabilities(probabilities, size):
    return F.interpolate(probabilities, size=size, mode='bilinear', align_corners=False)


def _mask_surface(mask):
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    surface = mask ^ eroded
    return surface if surface.any() else mask


def _hausdorff_distance(pred_mask, target_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    height, width = pred_mask.shape
    empty_penalty = float(np.hypot(height, width))

    pred_has_pixels = pred_mask.any()
    target_has_pixels = target_mask.any()
    if not pred_has_pixels and not target_has_pixels:
        return 0.0
    if not pred_has_pixels or not target_has_pixels:
        return empty_penalty

    pred_surface = _mask_surface(pred_mask)
    target_surface = _mask_surface(target_mask)
    distance_to_target = ndimage.distance_transform_edt(~target_surface)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface)
    pred_to_target = distance_to_target[pred_surface].max()
    target_to_pred = distance_to_pred[target_surface].max()
    return float(max(pred_to_target, target_to_pred))


@torch.no_grad()
def segmentation_error_metric_sums(segmentation, seg_annot, seg_mode, num_classes):
    device = segmentation.device
    logits = segmentation.detach().float()
    target = seg_annot.detach().long().to(device)

    if seg_mode == MULTICLASS_MODE:
        probabilities = logits.softmax(dim=1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        prediction = probabilities.argmax(dim=1)
    elif seg_mode == BINARY_MODE:
        foreground_probability = torch.sigmoid(logits)
        if num_classes == 1:
            probabilities = foreground_probability
            target_one_hot = target.float().unsqueeze(1)
            prediction = foreground_probability >= 0.5
        else:
            probabilities = torch.cat([1.0 - foreground_probability, foreground_probability], dim=1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
            prediction = probabilities.argmax(dim=1)
    elif seg_mode == MULTILABEL_MODE:
        probabilities = torch.sigmoid(logits)
        target_one_hot = target.float()
        prediction = probabilities >= 0.5
    else:
        raise ValueError(f'Unsupported segmentation mode: {seg_mode}')

    class_mae_sums = (probabilities - target_one_hot).abs().sum(dim=(0, 2, 3)).double()
    pixels_per_class = torch.full(
        (num_classes,),
        fill_value=probabilities.shape[0] * probabilities.shape[2] * probabilities.shape[3],
        dtype=torch.float64,
        device=device,
    )
    hausdorff_sums = torch.zeros(num_classes, dtype=torch.float64, device=device)
    hausdorff_counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    if seg_mode == MULTILABEL_MODE:
        prediction_cpu = prediction.detach().cpu().numpy().astype(bool)
        target_cpu = target_one_hot.detach().cpu().numpy().astype(bool)
        for sample_idx in range(prediction_cpu.shape[0]):
            for class_idx in range(num_classes):
                distance = _hausdorff_distance(
                    prediction_cpu[sample_idx, class_idx],
                    target_cpu[sample_idx, class_idx],
                )
                hausdorff_sums[class_idx] += distance
                hausdorff_counts[class_idx] += 1.0
    else:
        prediction_cpu = prediction.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        for sample_idx in range(prediction_cpu.shape[0]):
            for class_idx in range(num_classes):
                distance = _hausdorff_distance(
                    prediction_cpu[sample_idx] == class_idx,
                    target_cpu[sample_idx] == class_idx,
                )
                hausdorff_sums[class_idx] += distance
                hausdorff_counts[class_idx] += 1.0

    return torch.cat([class_mae_sums, pixels_per_class, hausdorff_sums, hausdorff_counts])


def summarize_segmentation_error_metric_sums(stats, class_names):
    num_classes = len(class_names)
    class_mae_sums = stats[:num_classes]
    class_pixel_counts = stats[num_classes:num_classes * 2]
    class_hausdorff_sums = stats[num_classes * 2:num_classes * 3]
    class_hausdorff_counts = stats[num_classes * 3:num_classes * 4]

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            'mean_absolute_error': class_mae_sums[i].item() / max(class_pixel_counts[i].item(), 1.0),
            'hausdorff_distance': class_hausdorff_sums[i].item() / max(class_hausdorff_counts[i].item(), 1.0),
        }

    return {
        'mean_absolute_error': class_mae_sums.sum().item() / max(class_pixel_counts.sum().item(), 1.0),
        'mean_hausdorff_distance': class_hausdorff_sums.sum().item() / max(class_hausdorff_counts.sum().item(), 1.0),
        'per_class': per_class_metrics,
    }
