import json
import os
from datetime import datetime, timezone

import numpy as np
import torch


STANDARD_METRIC_KEYS = (
    'phase',
    'epoch',
    'step',
    'loss',
    'classification_loss',
    'regression_loss',
    'segmentation_loss',
    'learning_rate',
    'num_batches',
    'num_rank_batches',
    'num_images',
    'batch_size_per_rank',
    'global_batch_size',
    'num_gpus',
    'num_workers_per_rank',
    'system_utilization_sample_count',
    'system_utilization_sample_interval_batches',
    'system_cpu_percent',
    'system_cpu_percent_mean',
    'system_cpu_percent_max',
    'process_cpu_percent',
    'process_cpu_percent_mean',
    'process_cpu_percent_max',
    'cpu_logical_count',
    'cpu_physical_count',
    'system_ram_percent',
    'system_ram_percent_mean',
    'system_ram_percent_max',
    'system_ram_used_gib',
    'system_ram_used_gib_mean',
    'system_ram_used_gib_max',
    'system_ram_available_gib',
    'system_ram_available_gib_mean',
    'system_ram_available_gib_max',
    'system_ram_total_gib',
    'process_ram_rss_gib',
    'process_ram_rss_gib_mean',
    'process_ram_rss_gib_max',
    'container_ram_percent',
    'container_ram_percent_mean',
    'container_ram_percent_max',
    'container_ram_used_gib',
    'container_ram_used_gib_mean',
    'container_ram_used_gib_max',
    'container_ram_limit_gib',
    'mosaic',
    'mixup',
    'amp',
    'cal_map_enabled',
    'conf_thres',
    'iou_thres',
    'segmentation_mode',
    'segmentation_classes',
    # 'detection_classes',
    'pixel_accuracy',
    'segmentation_log_loss',
    'mean_true_class_probability',
    'mean_iou',
    'mean_balanced_accuracy',
    'foreground_background_iou',
    'lane_background_iou',
    'confidence',
    'uncertainty',
    'per_class',
    # 'deis tection',
    # 'fitness',
    'best_loss',
    'best_epoch',
   # 'best_fitness',
)


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def standardize_metric_record(record):
    """Return a stable JSONL schema for train and validation metric rows."""
    standardized = {key: None for key in STANDARD_METRIC_KEYS}
    standardized.update(record)
    return standardized


def append_jsonl(path, record):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        'logged_at_utc': datetime.now(timezone.utc).isoformat(),
        **to_jsonable(record),
    }
    with open(path, 'a') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')
