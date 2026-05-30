import torch
import torch.nn.functional as F

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
