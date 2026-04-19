from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

try:
    from torchvision.models import ResNet50_Weights, ResNet101_Weights
except ImportError:
    ResNet50_Weights = None
    ResNet101_Weights = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_BDD100K_CATEGORIES = [
    "area/alternative",
    "area/drivable",
    "area/unknown",
    "bike",
    "bus",
    "car",
    "lane/crosswalk",
    "lane/double other",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if hasattr(Image, "Resampling"):
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
    NEAREST_RESAMPLE = Image.Resampling.NEAREST
else:
    BILINEAR_RESAMPLE = Image.BILINEAR
    NEAREST_RESAMPLE = Image.NEAREST


def str2bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def default_data_root() -> Path:
    sm_training = os.environ.get("SM_CHANNEL_TRAINING")
    if sm_training:
        return Path(sm_training)
    return Path(__file__).resolve().parent


def default_image_root() -> Path:
    sm_images = os.environ.get("SM_CHANNEL_IMAGES")
    if sm_images:
        return Path(sm_images)
    return default_data_root() / "100k_images"


def default_json_root() -> Path:
    sm_annotations = os.environ.get("SM_CHANNEL_ANNOTATIONS")
    if sm_annotations:
        return Path(sm_annotations)
    return default_data_root() / "100k"


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024**2)


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024**3)


def get_process_ram_usage_mb() -> float | None:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return bytes_to_mib(process.memory_info().rss)


def get_system_ram_usage() -> dict[str, float] | None:
    if psutil is None:
        return None
    memory = psutil.virtual_memory()
    return {
        "used_gib": bytes_to_gib(memory.used),
        "available_gib": bytes_to_gib(memory.available),
        "total_gib": bytes_to_gib(memory.total),
        "percent": float(memory.percent),
    }


def get_cpu_usage_percent() -> float | None:
    if psutil is None:
        return None
    return float(psutil.cpu_percent(interval=None))


def get_torch_gpu_memory(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "device_index": float(device_index),
        "allocated_mib": bytes_to_mib(torch.cuda.memory_allocated(device_index)),
        "reserved_mib": bytes_to_mib(torch.cuda.memory_reserved(device_index)),
        "max_allocated_mib": bytes_to_mib(torch.cuda.max_memory_allocated(device_index)),
        "free_mib": bytes_to_mib(free_bytes),
        "total_mib": bytes_to_mib(total_bytes),
    }


def query_nvidia_smi() -> list[dict[str, int | str | None]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return []

    stats: list[dict[str, int | str | None]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=5)]
        if len(parts) != 6:
            continue
        index, name, utilization, memory_used, memory_total, temperature = parts

        def parse_optional_int(value: str) -> int | None:
            try:
                return int(value)
            except ValueError:
                return None

        stats.append(
            {
                "index": parse_optional_int(index),
                "name": name,
                "utilization": parse_optional_int(utilization),
                "memory_used": parse_optional_int(memory_used),
                "memory_total": parse_optional_int(memory_total),
                "temperature": parse_optional_int(temperature),
            }
        )
    return stats


def build_resource_lines(device: torch.device) -> list[str]:
    lines: list[str] = []
    process_ram_mb = get_process_ram_usage_mb()
    system_ram = get_system_ram_usage()
    cpu_usage = get_cpu_usage_percent()

    ram_parts: list[str] = []
    if process_ram_mb is not None:
        ram_parts.append(f"process_rss={process_ram_mb:.1f} MB")
    if system_ram is not None:
        ram_parts.append(
            "system_ram="
            f"{system_ram['used_gib']:.2f}/{system_ram['total_gib']:.2f} GiB "
            f"({system_ram['percent']:.1f}% used, {system_ram['available_gib']:.2f} GiB avail)"
        )
    if cpu_usage is not None:
        ram_parts.append(f"cpu={cpu_usage:.1f}%")
    if ram_parts:
        lines.append("Host resources: " + " | ".join(ram_parts))
    else:
        lines.append("Host resources: psutil not installed; RAM/CPU telemetry unavailable.")

    torch_gpu = get_torch_gpu_memory(device)
    if torch_gpu is not None:
        lines.append(
            "Torch CUDA allocator: "
            f"device=cuda:{int(torch_gpu['device_index'])} "
            f"allocated={torch_gpu['allocated_mib']:.1f} MiB "
            f"reserved={torch_gpu['reserved_mib']:.1f} MiB "
            f"peak_allocated={torch_gpu['max_allocated_mib']:.1f} MiB "
            f"free={torch_gpu['free_mib']:.1f} MiB "
            f"total={torch_gpu['total_mib']:.1f} MiB"
        )

    nvidia_stats = query_nvidia_smi()
    if nvidia_stats:
        for gpu in nvidia_stats:
            utilization = "n/a" if gpu["utilization"] is None else f"{gpu['utilization']}%"
            temperature = "n/a" if gpu["temperature"] is None else f"{gpu['temperature']}C"
            memory_used = "n/a" if gpu["memory_used"] is None else f"{gpu['memory_used']} MiB"
            memory_total = "n/a" if gpu["memory_total"] is None else f"{gpu['memory_total']} MiB"
            lines.append(
                f"GPU {gpu['index']}: {gpu['name']} | util={utilization} | "
                f"mem={memory_used}/{memory_total} | temp={temperature}"
            )
    elif device.type == "cuda":
        lines.append("GPU telemetry: nvidia-smi unavailable; only torch allocator stats are available.")

    return lines


def progress_write_resource_snapshot(title: str, device: torch.device, show_progress: bool) -> None:
    progress_write(title, show_progress=show_progress)
    for line in build_resource_lines(device):
        progress_write(f"  {line}", show_progress=show_progress)


class BDD100KSegmentationDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        json_root: str | Path,
        split: str,
        class_to_idx: dict[str, int],
        image_size: tuple[int, int],
        normalize: bool = True,
        hflip_prob: float = 0.0,
        max_samples: int | None = None,
    ) -> None:
        self.image_dir = Path(image_root) / split
        self.json_dir = Path(json_root) / split
        self.split = split
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.normalize = normalize
        self.hflip_prob = hflip_prob
        self.samples = self._collect_samples(max_samples)

        if not self.samples:
            raise ValueError(
                f"No matching image/json pairs found for split='{split}' in "
                f"{self.image_dir} and {self.json_dir}."
            )

    def _collect_samples(self, max_samples: int | None) -> list[tuple[Path, Path]]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image split directory not found: {self.image_dir}")
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON split directory not found: {self.json_dir}")

        image_paths = {
            image_path.stem: image_path
            for image_path in self.image_dir.glob("*.jpg")
            if image_path.name != ".DS_Store"
        }
        json_paths = {
            json_path.stem: json_path
            for json_path in self.json_dir.glob("*.json")
            if json_path.name != ".DS_Store"
        }

        sample_ids = sorted(set(image_paths) & set(json_paths))
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]

        return [(image_paths[sample_id], json_paths[sample_id]) for sample_id in sample_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, json_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self._build_mask(json_path, image.shape[:2])

        if self.hflip_prob > 0.0 and np.random.rand() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = np.array(
            Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]),
                resample=BILINEAR_RESAMPLE,
            )
        )
        mask = np.array(
            Image.fromarray(mask).resize(
                (self.image_size[1], self.image_size[0]),
                resample=NEAREST_RESAMPLE,
            )
        )

        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor

    def _build_mask(self, json_path: Path, image_hw: tuple[int, int]) -> np.ndarray:
        height, width = image_hw
        mask_image = Image.new("L", (width, height), color=0)
        drawer = ImageDraw.Draw(mask_image)

        with json_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)

        frames = annotation.get("frames", [])
        if not frames:
            return np.array(mask_image, dtype=np.uint8)

        objects = frames[0].get("objects", [])
        polygon_objects = [obj for obj in objects if "poly2d" in obj]
        box_objects = [obj for obj in objects if "box2d" in obj]

        for obj in polygon_objects:
            self._draw_polygon(drawer, obj)
        for obj in box_objects:
            self._draw_box(drawer, obj, width=width, height=height)

        return np.array(mask_image, dtype=np.uint8)

    def _draw_polygon(self, drawer: ImageDraw.ImageDraw, obj: dict) -> None:
        category = obj.get("category")
        if category not in self.class_to_idx:
            return

        polygons = obj.get("poly2d", [])
        if not polygons:
            return

        if polygons and polygons[0] and isinstance(polygons[0][0], (int, float)):
            polygons = [polygons]

        fill_value = int(self.class_to_idx[category])
        for polygon in polygons:
            points = []
            for point in polygon:
                if len(point) < 2:
                    continue
                x_coord = int(round(point[0]))
                y_coord = int(round(point[1]))
                points.append((x_coord, y_coord))

            if len(points) >= 3:
                drawer.polygon(points, fill=fill_value)

    def _draw_box(self, drawer: ImageDraw.ImageDraw, obj: dict, width: int, height: int) -> None:
        category = obj.get("category")
        if category not in self.class_to_idx:
            return

        box = obj.get("box2d")
        if not box:
            return

        x1 = int(np.clip(round(box.get("x1", 0)), 0, width - 1))
        y1 = int(np.clip(round(box.get("y1", 0)), 0, height - 1))
        x2 = int(np.clip(round(box.get("x2", 0)), 0, width - 1))
        y2 = int(np.clip(round(box.get("y2", 0)), 0, height - 1))

        if x2 <= x1 or y2 <= y1:
            return

        drawer.rectangle((x1, y1, x2, y2), fill=int(self.class_to_idx[category]))


@dataclass
class TrainConfig:
    num_classes: int
    backbone: str = "resnet101"
    output_stride: int = 16
    crop_size: int = 513
    epochs: int = 5
    batch_size: int = 16
    max_iters: int = 30000
    base_lr: float = 0.007
    lr_power: float = 0.9
    momentum: float = 0.9
    weight_decay: float = 1e-4
    bn_decay: float = 0.9997
    ignore_index: int = 255
    device: str = "cuda"
    pretrained_backbone: bool = True


def get_device(preferred_device: str = "cuda") -> torch.device:
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def validate_train_batch_size(
    *,
    batch_size: int,
    test_batch_size: int,
    device: torch.device,
    visible_gpus: int,
    train_enabled: bool,
) -> None:
    if not train_enabled or device.type != "cuda" or visible_gpus <= 1:
        return

    if batch_size < visible_gpus:
        raise ValueError(
            f"Global train batch-size must be at least the number of visible GPUs. "
            f"Received batch-size={batch_size} across {visible_gpus} GPUs."
        )
    if batch_size % visible_gpus != 0:
        raise ValueError(
            f"Global train batch-size must be divisible by the number of visible GPUs for DDP. "
            f"Received batch-size={batch_size} across {visible_gpus} GPUs."
        )
    if test_batch_size < visible_gpus:
        raise ValueError(
            f"Global validation batch-size must be at least the number of visible GPUs. "
            f"Received test-batch-size={test_batch_size} across {visible_gpus} GPUs."
        )
    if test_batch_size % visible_gpus != 0:
        raise ValueError(
            f"Global validation batch-size must be divisible by the number of visible GPUs for DDP. "
            f"Received test-batch-size={test_batch_size} across {visible_gpus} GPUs."
        )


def poly_learning_rate(base_lr: float, current_iter: int, max_iters: int, power: float = 0.9) -> float:
    if current_iter > max_iters:
        return 0.0
    return base_lr * ((1.0 - (current_iter / max_iters)) ** power)


def build_model(config: TrainConfig) -> nn.Module:
    model_builders = {
        "resnet50": deeplabv3_resnet50,
        "resnet101": deeplabv3_resnet101,
    }
    backbone_weights = {
        "resnet50": ResNet50_Weights.DEFAULT if ResNet50_Weights is not None else None,
        "resnet101": ResNet101_Weights.DEFAULT if ResNet101_Weights is not None else None,
    }
    model_builder = model_builders[config.backbone]
    build_kwargs = {
        "num_classes": config.num_classes,
        "aux_loss": False,
    }
    builder_signature = inspect.signature(model_builder)
    if "weights" in builder_signature.parameters:
        build_kwargs["weights"] = None
        build_kwargs["weights_backbone"] = (
            backbone_weights[config.backbone] if config.pretrained_backbone else None
        )
    else:
        build_kwargs["pretrained"] = False
        build_kwargs["pretrained_backbone"] = config.pretrained_backbone
    return model_builder(**build_kwargs)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_criterion(config: TrainConfig) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config.ignore_index)


def extract_segmentation_logits(model_output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["out"]
    return model_output


def build_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    preds = predictions.detach().view(-1)
    truth = targets.detach().view(-1)
    valid_mask = truth != ignore_index
    preds = preds[valid_mask].to(torch.int64)
    truth = truth[valid_mask].to(torch.int64)

    if truth.numel() == 0:
        return torch.zeros((num_classes, num_classes), device=predictions.device, dtype=torch.int64)

    encoded = truth * num_classes + preds
    bins = torch.bincount(encoded, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes).to(torch.int64)


def metrics_from_confusion_matrix(
    confusion: np.ndarray,
    idx_to_class: dict[int, str] | None = None,
) -> dict[str, float | int | list | dict]:
    total = confusion.sum()
    correct = np.trace(confusion)
    accuracy = float(correct / total) if total > 0 else 0.0

    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    union = support + predicted - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.zeros_like(true_positive),
        where=union > 0,
    )

    valid_classes = support > 0
    active_classes = (support > 0) | (predicted > 0)
    macro_precision = float(precision[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_recall = float(recall[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_f1 = float(f1[valid_classes].mean()) if np.any(valid_classes) else 0.0
    mean_iou = float(iou[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_precision_active = float(precision[active_classes].mean()) if np.any(active_classes) else 0.0
    macro_recall_active = float(recall[active_classes].mean()) if np.any(active_classes) else 0.0
    macro_f1_active = float(f1[active_classes].mean()) if np.any(active_classes) else 0.0
    mean_iou_active = float(iou[active_classes].mean()) if np.any(active_classes) else 0.0

    weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    weighted_precision = float((precision * weights).sum())
    weighted_recall = float((recall * weights).sum())
    weighted_f1 = float((f1 * weights).sum())
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    class_order = [
        idx_to_class.get(class_index, f"class_{class_index}")
        if idx_to_class is not None
        else f"class_{class_index}"
        for class_index in range(confusion.shape[0])
    ]
    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_order):
        per_class[class_name] = {
            "class_index": class_index,
            "support_pixels": int(support[class_index]),
            "predicted_pixels": int(predicted[class_index]),
            "true_positive_pixels": int(true_positive[class_index]),
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
            "dice": float(f1[class_index]),
            "iou": float(iou[class_index]),
        }

    return {
        "pixel_accuracy": accuracy,
        "accuracy": accuracy,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "dice_micro": micro_f1,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "dice_macro": macro_f1,
        "precision_macro_active": macro_precision_active,
        "recall_macro_active": macro_recall_active,
        "f1_macro_active": macro_f1_active,
        "dice_macro_active": macro_f1_active,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "dice_weighted": weighted_f1,
        "mean_iou": mean_iou,
        "mean_iou_active": mean_iou_active,
        "total_labeled_pixels": int(total),
        "confusion_matrix": confusion.tolist(),
        "class_order": class_order,
        "per_class": per_class,
    }


def progress_write(message: str, show_progress: bool) -> None:
    if not is_main_process():
        return
    if show_progress and tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def build_progress_desc(split_name: str, epoch_index: int, total_epochs: int) -> str:
    return f"{split_name.capitalize()} {epoch_index}/{total_epochs}"


def format_metric_line(split_name: str, metrics: dict[str, float]) -> str:
    return (
        f"{split_name:<5} "
        f"loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"precision_w={metrics['precision_weighted']:.4f} "
        f"recall_w={metrics['recall_weighted']:.4f} "
        f"f1_w={metrics['f1_weighted']:.4f} "
        f"precision_macro={metrics['precision_macro']:.4f} "
        f"recall_macro={metrics['recall_macro']:.4f} "
        f"f1_macro={metrics['f1_macro']:.4f} "
        f"miou={metrics['mean_iou']:.4f}"
    )


def current_wall_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def sanitize_metric_key(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_metrics_summary_row(
    record: dict,
    idx_to_class: dict[int, str],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "epoch": int(record["epoch"]),
        "lr": float(record["lr"]),
        "epoch_elapsed_seconds": float(record.get("epoch_elapsed_seconds", 0.0)),
    }

    scalar_skip_keys = {"confusion_matrix", "class_order", "per_class"}
    for split_name in ("train", "val"):
        split_metrics = record.get(split_name, {})
        for key, value in split_metrics.items():
            if key in scalar_skip_keys:
                continue
            row[f"{split_name}_{key}"] = value

        per_class_metrics = split_metrics.get("per_class", {})
        for class_index in sorted(idx_to_class):
            class_name = idx_to_class[class_index]
            class_key = sanitize_metric_key(class_name)
            class_metrics = per_class_metrics.get(class_name)
            if class_metrics is None:
                continue
            for metric_name in (
                "support_pixels",
                "predicted_pixels",
                "true_positive_pixels",
                "precision",
                "recall",
                "f1",
                "dice",
                "iou",
            ):
                row[f"{split_name}_{class_key}_{metric_name}"] = class_metrics[metric_name]

    return row


def write_metrics_summary_csv(
    csv_path: str | Path,
    history: list[dict],
    idx_to_class: dict[int, str],
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [build_metrics_summary_row(record, idx_to_class) for record in history]
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(
    csv_path: str | Path,
    confusion_matrix: list[list[int]],
    class_order: list[str],
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + class_order)
        for class_name, row in zip(class_order, confusion_matrix):
            writer.writerow([class_name] + row)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    idx_to_class: dict[int, str],
    ignore_index: int,
    epoch_index: int,
    total_epochs: int,
    split_name: str,
    show_progress: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    epoch_phase_start = time.perf_counter()

    total_loss = torch.zeros(1, device=device, dtype=torch.float64)
    total_samples_seen = torch.zeros(1, device=device, dtype=torch.float64)
    total_batches = 0
    confusion = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)
    progress_bar = None
    iterable: Iterable[tuple[torch.Tensor, torch.Tensor]] = dataloader

    if show_progress and tqdm is not None:
        progress_bar = tqdm(
            dataloader,
            desc=build_progress_desc(split_name, epoch_index, total_epochs),
            total=len(dataloader),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        iterable = progress_bar

    for images, targets in iterable:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with autocast(enabled=amp_enabled and device.type == "cuda"):
                logits = extract_segmentation_logits(model(images))
                loss = criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.detach().to(torch.float64) * batch_size
        total_samples_seen += batch_size
        total_batches += 1
        confusion += build_confusion_matrix(predictions, targets, num_classes, ignore_index)

        if progress_bar is not None:
            running_metrics = metrics_from_confusion_matrix(confusion.detach().cpu().numpy(), idx_to_class=idx_to_class)
            progress_bar.set_postfix(
                loss=f"{(total_loss.item() / max(total_samples_seen.item(), 1.0)):.4f}",
                acc=f"{running_metrics['accuracy']:.4f}",
                f1=f"{running_metrics['f1_weighted']:.4f}",
                miou=f"{running_metrics['mean_iou']:.4f}",
            )

    if is_distributed():
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_seen, op=dist.ReduceOp.SUM)

    metrics = metrics_from_confusion_matrix(confusion.detach().cpu().numpy(), idx_to_class=idx_to_class)
    metrics["loss"] = float(total_loss.item() / max(total_samples_seen.item(), 1.0))
    metrics["elapsed_seconds"] = time.perf_counter() - epoch_phase_start
    metrics["num_batches"] = len(dataloader)
    metrics["num_samples"] = len(dataloader.dataset)
    metrics["num_samples_seen"] = int(round(total_samples_seen.item()))

    if progress_bar is not None:
        progress_bar.close()

    return metrics


def save_model(model: nn.Module, model_dir: str | Path, save_file: str) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / save_file
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), save_path)
    return save_path


def build_epoch_checkpoint_name(save_file: str, epoch_index: int, total_epochs: int) -> str:
    save_path = Path(save_file)
    width = max(3, len(str(total_epochs)))
    epoch_suffix = f"_epoch_{epoch_index:0{width}d}"
    if save_path.suffix:
        return f"{save_path.stem}{epoch_suffix}{save_path.suffix}"
    return f"{save_path.name}{epoch_suffix}"


def append_metrics(
    metrics_path: str | Path,
    record: dict,
    idx_to_class: dict[int, str],
) -> None:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    else:
        history = []

    history.append(record)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    write_metrics_summary_csv(metrics_path.with_suffix(".csv"), history, idx_to_class)

    epoch_reports_dir = metrics_path.parent / "epoch_reports"
    epoch_reports_dir.mkdir(parents=True, exist_ok=True)
    epoch_report_path = epoch_reports_dir / f"epoch_{record['epoch']:03d}_metrics.json"
    with epoch_report_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    for split_name in ("train", "val"):
        split_metrics = record[split_name]
        write_confusion_matrix_csv(
            epoch_reports_dir / f"epoch_{record['epoch']:03d}_{split_name}_confusion_matrix.csv",
            confusion_matrix=split_metrics["confusion_matrix"],
            class_order=split_metrics["class_order"],
        )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: DistributedSampler | None,
    val_sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
    idx_to_class: dict[int, str],
    output_dir: str | Path,
    model_dir: str | Path,
    save_file: str,
    device: str | torch.device = "cpu",
    show_progress: bool = True,
    scaler: GradScaler | None = None,
    amp_enabled: bool = False,
) -> list[dict]:
    device = torch.device(device)
    model.to(device)
    training_start = time.perf_counter()
    global_step = 0
    metrics_history: list[dict] = []
    best_val_f1 = -1.0
    metrics_path = Path(output_dir) / "metrics_history.json"
    epoch_progress = None

    if show_progress and tqdm is not None:
        epoch_progress = tqdm(
            range(config.epochs),
            desc="Epochs",
            total=config.epochs,
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        epoch_iterable: Iterable[int] = epoch_progress
    else:
        epoch_iterable = range(config.epochs)

    progress_write(
        (
            f"Starting training: epochs={config.epochs} "
            f"train_batches_per_epoch={len(train_loader)} "
            f"val_batches_per_epoch={len(val_loader)} "
            f"planned_train_steps={config.epochs * len(train_loader)} "
            f"started_at={current_wall_time()}"
        ),
        show_progress=show_progress,
    )

    for epoch in epoch_iterable:
        epoch_index = epoch + 1
        epoch_start = time.perf_counter()
        progress_write(
            f"Epoch {epoch_index:03d}/{config.epochs:03d} started at {current_wall_time()}",
            show_progress=show_progress,
        )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        current_lr = poly_learning_rate(
            base_lr=config.base_lr,
            current_iter=min(global_step, config.max_iters),
            max_iters=max(config.max_iters, 1),
            power=config.lr_power,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=config.num_classes,
            idx_to_class=idx_to_class,
            ignore_index=config.ignore_index,
            epoch_index=epoch_index,
            total_epochs=config.epochs,
            split_name="train",
            show_progress=show_progress,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        progress_write(
            (
                f"Finished train split for epoch {epoch_index:03d} at {current_wall_time()} "
                f"elapsed={format_elapsed(train_metrics['elapsed_seconds'])}"
            ),
            show_progress=show_progress,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config.num_classes,
            idx_to_class=idx_to_class,
            ignore_index=config.ignore_index,
            epoch_index=epoch_index,
            total_epochs=config.epochs,
            split_name="val",
            show_progress=show_progress,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        progress_write(
            (
                f"Finished val split for epoch {epoch_index:03d} at {current_wall_time()} "
                f"elapsed={format_elapsed(val_metrics['elapsed_seconds'])}"
            ),
            show_progress=show_progress,
        )

        global_step += len(train_loader)
        epoch_elapsed = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch_index,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_elapsed_seconds": epoch_elapsed,
        }
        metrics_history.append(record)
        if is_main_process():
            append_metrics(metrics_path, record, idx_to_class)

        progress_write(
            f"Epoch {epoch_index:03d}/{config.epochs:03d} lr={current_lr:.6f}",
            show_progress=show_progress,
        )
        progress_write(format_metric_line("train", train_metrics), show_progress=show_progress)
        progress_write(format_metric_line("val", val_metrics), show_progress=show_progress)
        progress_write(
            (
                f"Epoch {epoch_index:03d}/{config.epochs:03d} finished at {current_wall_time()} "
                f"elapsed={format_elapsed(epoch_elapsed)}"
            ),
            show_progress=show_progress,
        )

        if is_main_process():
            epoch_path = save_model(
                model,
                model_dir,
                build_epoch_checkpoint_name(save_file, epoch_index, config.epochs),
            )
            progress_write(f"Saved epoch checkpoint to {epoch_path}", show_progress=show_progress)

        if epoch_progress is not None:
            epoch_progress.set_postfix(
                lr=f"{current_lr:.6f}",
                train_loss=f"{train_metrics['loss']:.4f}",
                val_f1=f"{val_metrics['f1_weighted']:.4f}",
                val_miou=f"{val_metrics['mean_iou']:.4f}",
            )

        if is_main_process() and val_metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = val_metrics["f1_weighted"]
            best_path = save_model(model, model_dir, f"best_{save_file}")
            progress_write(f"Saved best checkpoint to {best_path}", show_progress=show_progress)

        if is_distributed():
            dist.barrier()

    final_path: Path | None = None
    if is_main_process():
        final_path = save_model(model, model_dir, save_file)
    if epoch_progress is not None:
        epoch_progress.close()

    total_training_elapsed = time.perf_counter() - training_start
    progress_write(
        f"Training finished at {current_wall_time()} total_elapsed={format_elapsed(total_training_elapsed)}",
        show_progress=show_progress,
    )
    if final_path is not None:
        progress_write(f"Saved final checkpoint to {final_path}", show_progress=show_progress)
    if is_distributed():
        dist.barrier()
    return metrics_history


def discover_categories(json_root: str | Path, split: str = "train") -> list[str]:
    json_dir = Path(json_root) / split
    categories: set[str] = set()

    for json_path in sorted(json_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)

        for frame in annotation.get("frames", []):
            for obj in frame.get("objects", []):
                category = obj.get("category")
                if category:
                    categories.add(category)

    return sorted(categories)


def build_class_to_idx(categories: list[str]) -> dict[str, int]:
    class_to_idx = {"background": 0}
    for index, category in enumerate(categories, start=1):
        class_to_idx[category] = index
    return class_to_idx


def load_or_create_category_map(
    json_root: str | Path,
    category_map_path: str | Path | None,
    split: str = "train",
    auto_discover: bool = True,
    write_map: bool = True,
) -> dict[str, int]:
    categories = discover_categories(json_root, split=split) if auto_discover else DEFAULT_BDD100K_CATEGORIES
    expected_class_to_idx = build_class_to_idx(categories)

    if category_map_path is not None:
        category_map_path = Path(category_map_path)
        if category_map_path.exists():
            with category_map_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            saved_class_to_idx = {str(key): int(value) for key, value in saved.items()}

            if auto_discover:
                expected_categories = set(expected_class_to_idx)
                saved_categories = set(saved_class_to_idx)
                missing_categories = sorted(expected_categories - saved_categories)
                extra_categories = sorted(saved_categories - expected_categories)
                if missing_categories or extra_categories:
                    details: list[str] = []
                    if missing_categories:
                        details.append(f"missing categories: {', '.join(missing_categories)}")
                    if extra_categories:
                        details.append(f"unexpected categories: {', '.join(extra_categories)}")
                    raise ValueError(
                        "The provided category map does not cover all discovered training classes for "
                        f"split='{split}'. {'; '.join(details)}. "
                        "Remove --category-map-path, point it to a new file for an all-class map, or "
                        "use --no-auto-discover-classes true if you intentionally want a fixed class list."
                    )

            return saved_class_to_idx
    else:
        category_map_path = Path(json_root) / "category_map.json"

    if write_map:
        category_map_path.parent.mkdir(parents=True, exist_ok=True)
        with category_map_path.open("w", encoding="utf-8") as handle:
            json.dump(expected_class_to_idx, handle, indent=2, sort_keys=True)

    return expected_class_to_idx


def build_dataloaders(
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    image_size = (args.image_height, args.image_width)
    world_size = get_world_size()
    distributed = is_distributed()
    rank = get_rank()
    train_batch_size = args.batch_size // world_size if distributed else args.batch_size
    test_batch_size = args.test_batch_size // world_size if distributed else args.test_batch_size
    num_workers = max(1, args.num_workers // world_size) if distributed and args.num_workers > 0 else args.num_workers

    train_dataset = BDD100KSegmentationDataset(
        image_root=args.image_root,
        json_root=args.json_root,
        split=args.train_split,
        class_to_idx=class_to_idx,
        image_size=image_size,
        normalize=not args.disable_normalization,
        hflip_prob=args.train_hflip_prob,
        max_samples=args.max_train_samples,
    )
    
    
    
    test_dataset = BDD100KSegmentationDataset(
        image_root=args.image_root,
        json_root=args.json_root,
        split=args.test_split,
        class_to_idx=class_to_idx,
        image_size=image_size,
        normalize=not args.disable_normalization,
        hflip_prob=0.0,
        max_samples=args.max_test_samples,
    )

    # PyTorch 1.7.x can be unstable with pin_memory/persistent workers under mp.spawn + DDP.
    pin_memory = device.type == "cuda" and not distributed
    persistent_workers = num_workers > 0 and not distributed
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=args.drop_last,
        )
        if distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=args.drop_last,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        sampler=test_sampler,
    )
    return train_loader, test_loader, train_sampler, test_sampler


def create_argparser() -> argparse.ArgumentParser:
    image_root = default_image_root()
    json_root = default_json_root()
    data_root = default_data_root()

    parser = argparse.ArgumentParser(
        description="DeepLabv3 training scaffold for BDD100K-style image/json pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(image_root),
        help="Root folder that contains split subfolders like train/, val/, and test/ for images.",
    )
    parser.add_argument(
        "--json-root",
        type=str,
        default=str(json_root),
        help="Root folder that contains split subfolders like train/, val/, and test/ for JSON annotations.",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Dataset split name for training.")
    parser.add_argument("--test-split", type=str, default="val", help="Dataset split name for evaluation/testing.")
    parser.add_argument("--image-height", type=int, default=513, help="Resized training/eval image height.")
    parser.add_argument("--image-width", type=int, default=513, help="Resized training/eval image width.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of full training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Train batch size.")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Test batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--train-hflip-prob", type=float, default=0.5, help="Horizontal flip probability for train data.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples for quick experiments.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap on test samples for quick experiments.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the last incomplete train batch.")
    parser.add_argument("--disable-normalization", action="store_true", help="Skip ImageNet normalization.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Preferred device.")
    parser.add_argument("--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101"], help="Backbone depth.")
    parser.add_argument(
        "--pretrained-backbone",
        type=str2bool,
        default=False,
        help="Load ImageNet-pretrained ResNet backbone weights for the torchvision DeepLabV3 model.",
    )
    parser.add_argument(
        "--output-stride",
        type=int,
        default=16,
        choices=[8, 16],
        help="Deprecated: kept for CLI compatibility, but ignored by the torchvision DeepLabV3 builders.",
    )
    parser.add_argument("--base-lr", type=float, default=0.007, help="Initial learning rate.")
    parser.add_argument("--max-iters", type=int, default=30000, help="Maximum train iterations.")
    parser.add_argument(
        "--bn-decay",
        type=float,
        default=0.9997,
        help="Deprecated: kept for CLI compatibility, but ignored by the torchvision DeepLabV3 builders.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--lr-power", type=float, default=0.9, help="Poly LR exponent.")
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Use automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--sync-bn",
        type=str2bool,
        default=True,
        help="Convert BatchNorm layers to SyncBatchNorm for DDP training.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="Master address for torch.distributed initialization.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29500")),
        help="Master port for torch.distributed initialization.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", str(data_root / "models")),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", str(data_root / "output")),
        help="Directory to save metrics and reports.",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default="final_model.pth",
        help="Filename for the final checkpoint.",
    )
    parser.add_argument(
        "--category-map-path",
        type=str,
        default=None,
        help="Optional JSON file to save/load the discovered category map.",
    )
    parser.add_argument(
        "--no-auto-discover-classes",
        type=str2bool,
        default=False,
        help="Use the built-in BDD100K category list instead of discovering classes from the train split.",
    )
    parser.add_argument(
        "--smoke-test",
        type=str2bool,
        default=False,
        help="Build loaders and run a single forward pass instead of training.",
    )
    parser.add_argument(
        "--train",
        type=str2bool,
        default=True,
        help="Run the training loop after building the dataloaders.",
    )
    parser.add_argument(
        "--progress",
        type=str2bool,
        default=True,
        help="Show tqdm progress bars during training and evaluation.",
    )
    return parser


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    distributed = world_size > 1

    if distributed:
        setup_distributed(rank, world_size, args.master_addr, args.master_port)

    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}" if distributed else "cuda")
            torch.backends.cudnn.benchmark = True
        else:
            device = get_device(args.device)

        if distributed:
            if is_main_process():
                class_to_idx = load_or_create_category_map(
                    json_root=args.json_root,
                    category_map_path=args.category_map_path,
                    split=args.train_split,
                    auto_discover=not args.no_auto_discover_classes,
                    write_map=True,
                )
            dist.barrier()
            if not is_main_process():
                class_to_idx = load_or_create_category_map(
                    json_root=args.json_root,
                    category_map_path=args.category_map_path,
                    split=args.train_split,
                    auto_discover=not args.no_auto_discover_classes,
                    write_map=False,
                )
        else:
            class_to_idx = load_or_create_category_map(
                json_root=args.json_root,
                category_map_path=args.category_map_path,
                split=args.train_split,
                auto_discover=not args.no_auto_discover_classes,
                write_map=True,
            )

        idx_to_class = {index: name for name, index in class_to_idx.items()}

        config = TrainConfig(
            num_classes=len(class_to_idx),
            backbone=args.backbone,
            output_stride=args.output_stride,
            crop_size=args.image_height,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            base_lr=args.base_lr,
            lr_power=args.lr_power,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            bn_decay=args.bn_decay,
            device=args.device,
            pretrained_backbone=args.pretrained_backbone,
        )

        train_loader, test_loader, train_sampler, test_sampler = build_dataloaders(args, class_to_idx, device)
        local_train_batch_size = args.batch_size // world_size if distributed else args.batch_size
        local_test_batch_size = args.test_batch_size // world_size if distributed else args.test_batch_size
        local_num_workers = max(1, args.num_workers // world_size) if distributed and args.num_workers > 0 else args.num_workers

        if is_main_process():
            print(f"Using device: {device}")
            print(f"Image root:   {args.image_root}")
            print(f"JSON root:    {args.json_root}")
            print(f"Train split:  {args.train_split}")
            print(f"Test split:   {args.test_split}")
            print(f"Classes:      {len(class_to_idx)}")
            print(f"Background:   {idx_to_class[0]}")
            print(f"Model dir:    {args.model_dir}")
            print(f"Output dir:   {args.output_dir}")
            print(f"Train samples: {len(train_loader.dataset)}")
            print(f"Test samples:  {len(test_loader.dataset)}")
            print(f"Train batches: {len(train_loader)}")
            print(f"Test batches:  {len(test_loader)}")
            if distributed:
                print(f"DDP world size: {world_size}")
                print(f"Local train batch size per GPU: {local_train_batch_size}")
                print(f"Local val batch size per GPU:   {local_test_batch_size}")
                print(f"Local dataloader workers/GPU:   {local_num_workers}")
                print("DDP dataloader mode: pin_memory=False, persistent_workers=False")

        model = build_model(config)
        if distributed and args.sync_bn and device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if is_main_process():
                print("Enabled SyncBatchNorm for DDP training")

        model = model.to(device)
        if distributed:
            model = DDP(model, device_ids=[rank], output_device=rank)
            if is_main_process():
                print(f"Enabled DistributedDataParallel across {world_size} GPUs")

        if is_main_process():
            print(f"Model classes: {config.num_classes}")

        if args.smoke_test or not args.train:
            train_images, train_masks = next(iter(train_loader))
            test_images, test_masks = next(iter(test_loader))

            train_images = train_images.to(device)
            model.eval()
            with torch.no_grad():
                logits = extract_segmentation_logits(model(train_images))

            if is_main_process():
                print(f"Train batch image shape: {tuple(train_images.shape)}")
                print(f"Train batch mask shape:  {tuple(train_masks.shape)}")
                print(f"Test batch image shape:  {tuple(test_images.shape)}")
                print(f"Test batch mask shape:   {tuple(test_masks.shape)}")
                print(f"Forward output shape:    {tuple(logits.shape)}")
                print(f"Class map saved to:      {args.category_map_path or Path(args.json_root) / 'category_map.json'}")
            return

        criterion = build_criterion(config)
        optimizer = build_optimizer(model, config)
        scaler = GradScaler(enabled=args.amp and device.type == "cuda")
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            train_sampler=train_sampler,
            val_sampler=test_sampler,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            idx_to_class=idx_to_class,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            save_file=args.save_file,
            device=device,
            show_progress=args.progress and is_main_process(),
            scaler=scaler,
            amp_enabled=args.amp,
        )
    finally:
        cleanup_distributed()


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    preferred_device = get_device(args.device)
    visible_gpus = torch.cuda.device_count() if preferred_device.type == "cuda" else 0
    distributed = preferred_device.type == "cuda" and visible_gpus > 1 and args.train and not args.smoke_test
    world_size = visible_gpus if distributed else 1

    validate_train_batch_size(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=preferred_device,
        visible_gpus=visible_gpus,
        train_enabled=args.train and not args.smoke_test,
    )

    if distributed:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, args)


if __name__ == "__main__":
    
    main()
