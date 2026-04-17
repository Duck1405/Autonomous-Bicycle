import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
    return Path("/Users/amannindra/Projects/Auto")


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        bn_momentum: float = 0.0003,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = conv1x1(in_channels, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = conv1x1(planes, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        layers: list[int],
        output_stride: int = 16,
        multi_grid: tuple[int, int, int] = (1, 2, 4),
        bn_momentum: float = 0.0003,
    ) -> None:
        super().__init__()

        if output_stride not in (8, 16):
            raise ValueError("DeepLabv3 ResNet backbone only supports output_stride 8 or 16.")

        self.in_channels = 64
        self.bn_momentum = bn_momentum

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if output_stride == 16:
            layer_strides = [1, 2, 2, 1]
            layer_dilations = [1, 1, 1, 2]
        else:
            layer_strides = [1, 2, 1, 1]
            layer_dilations = [1, 1, 2, 4]

        self.layer1 = self._make_layer(64, layers[0], layer_strides[0], layer_dilations[0])
        self.layer2 = self._make_layer(128, layers[1], layer_strides[1], layer_dilations[1])
        self.layer3 = self._make_layer(256, layers[2], layer_strides[2], layer_dilations[2])
        self.layer4 = self._make_layer(
            512,
            layers[3],
            layer_strides[3],
            layer_dilations[3],
            multi_grid=multi_grid,
        )

        self.out_channels = 2048
        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        dilation: int,
        multi_grid: tuple[int, int, int] | None = None,
    ) -> nn.Sequential:
        if multi_grid is None:
            multi_grid = tuple(1 for _ in range(blocks))
        elif len(multi_grid) != blocks:
            if len(multi_grid) == 3 and blocks > 3:
                repeats = math.ceil(blocks / len(multi_grid))
                multi_grid = (multi_grid * repeats)[:blocks]
            else:
                raise ValueError("multi_grid must match the number of blocks or be a 3-value pattern.")

        out_channels = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=self.bn_momentum),
            )

        layers = [
            Bottleneck(
                in_channels=self.in_channels,
                planes=planes,
                stride=stride,
                dilation=dilation * multi_grid[0],
                downsample=downsample,
                bn_momentum=self.bn_momentum,
            )
        ]
        self.in_channels = out_channels

        for block_index in range(1, blocks):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    planes=planes,
                    stride=1,
                    dilation=dilation * multi_grid[block_index],
                    bn_momentum=self.bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze_batch_norm(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        bn_momentum: float,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)
        pooled = self.proj(pooled)
        return F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: tuple[int, int, int],
        out_channels: int = 256,
        bn_momentum: float = 0.0003,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, bn_momentum))

        modules.append(ASPPPooling(in_channels, out_channels, bn_momentum))

        self.branches = nn.ModuleList(modules)
        merged_channels = out_channels * len(self.branches)
        self.project = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        return self.project(x)


class DeepLabHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        atrous_rates: tuple[int, int, int],
        bn_momentum: float = 0.0003,
    ) -> None:
        super().__init__()
        self.aspp = ASPP(
            in_channels=in_channels,
            atrous_rates=atrous_rates,
            out_channels=256,
            bn_momentum=bn_momentum,
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        return self.classifier(x)


class DeepLabV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet101",
        output_stride: int = 16,
        multi_grid: tuple[int, int, int] = (1, 2, 4),
        bn_decay: float = 0.9997,
    ) -> None:
        super().__init__()

        layers_by_backbone = {
            "resnet50": [3, 4, 6, 3],
            "resnet101": [3, 4, 23, 3],
        }
        if backbone not in layers_by_backbone:
            raise ValueError("backbone must be 'resnet50' or 'resnet101'.")

        bn_momentum = 1.0 - bn_decay
        atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)

        self.backbone = ResNetBackbone(
            layers=layers_by_backbone[backbone],
            output_stride=output_stride,
            multi_grid=multi_grid,
            bn_momentum=bn_momentum,
        )
        self.head = DeepLabHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            atrous_rates=atrous_rates,
            bn_momentum=bn_momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        logits = self.head(features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def freeze_batch_norm(self) -> None:
        self.backbone.freeze_batch_norm()
        for module in self.head.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)


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
                resample=Image.BILINEAR,
            )
        )
        mask = np.array(
            Image.fromarray(mask).resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.NEAREST,
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


def get_device(preferred_device: str = "cuda") -> torch.device:
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def poly_learning_rate(base_lr: float, current_iter: int, max_iters: int, power: float = 0.9) -> float:
    if current_iter > max_iters:
        return 0.0
    return base_lr * ((1.0 - (current_iter / max_iters)) ** power)


def build_model(config: TrainConfig) -> DeepLabV3:
    return DeepLabV3(
        num_classes=config.num_classes,
        backbone=config.backbone,
        output_stride=config.output_stride,
        bn_decay=config.bn_decay,
    )


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_criterion(config: TrainConfig) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=config.ignore_index)


def build_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    preds = predictions.detach().view(-1).cpu()
    truth = targets.detach().view(-1).cpu()
    valid_mask = truth != ignore_index
    preds = preds[valid_mask].to(torch.int64)
    truth = truth[valid_mask].to(torch.int64)

    if truth.numel() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    encoded = truth * num_classes + preds
    bins = torch.bincount(encoded, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes).numpy().astype(np.int64)


def metrics_from_confusion_matrix(confusion: np.ndarray) -> dict[str, float]:
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
    macro_precision = float(precision[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_recall = float(recall[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_f1 = float(f1[valid_classes].mean()) if np.any(valid_classes) else 0.0
    mean_iou = float(iou[valid_classes].mean()) if np.any(valid_classes) else 0.0

    weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)
    weighted_precision = float((precision * weights).sum())
    weighted_recall = float((recall * weights).sum())
    weighted_f1 = float((f1 * weights).sum())

    return {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "mean_iou": mean_iou,
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_batches = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in dataloader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            predictions = torch.argmax(logits, dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        confusion += build_confusion_matrix(predictions, targets, num_classes, ignore_index)

    metrics = metrics_from_confusion_matrix(confusion)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def save_model(model: nn.Module, model_dir: str | Path, save_file: str) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / save_file
    torch.save(model.state_dict(), save_path)
    return save_path


def append_metrics(metrics_path: str | Path, record: dict) -> None:
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


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
    output_dir: str | Path,
    model_dir: str | Path,
    save_file: str,
    device: str | torch.device = "cpu",
) -> list[dict]:
    device = torch.device(device)
    model.to(device)
    global_step = 0
    metrics_history: list[dict] = []
    best_val_f1 = -1.0
    metrics_path = Path(output_dir) / "metrics_history.json"

    for epoch in range(config.epochs):
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
            ignore_index=config.ignore_index,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            optimizer=None,
        )

        global_step += len(train_loader)

        record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics_history.append(record)
        append_metrics(metrics_path, record)

        print(
            f"epoch={epoch + 1:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1_weighted']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_weighted']:.4f} "
            f"val_miou={val_metrics['mean_iou']:.4f}"
        )

        if val_metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = val_metrics["f1_weighted"]
            best_path = save_model(model, model_dir, f"best_{save_file}")
            print(f"Saved best checkpoint to {best_path}")

    final_path = save_model(model, model_dir, save_file)
    print(f"Saved final checkpoint to {final_path}")
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


def load_or_create_category_map(
    json_root: str | Path,
    category_map_path: str | Path | None,
    split: str = "train",
    auto_discover: bool = True,
) -> dict[str, int]:
    if category_map_path is not None:
        category_map_path = Path(category_map_path)
        if category_map_path.exists():
            with category_map_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            return {str(key): int(value) for key, value in saved.items()}
    else:
        category_map_path = Path(json_root) / "category_map.json"

    categories = discover_categories(json_root, split=split) if auto_discover else DEFAULT_BDD100K_CATEGORIES
    class_to_idx = {"background": 0}
    for index, category in enumerate(categories, start=1):
        class_to_idx[category] = index

    category_map_path.parent.mkdir(parents=True, exist_ok=True)
    with category_map_path.open("w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2, sort_keys=True)

    return class_to_idx


def build_dataloaders(
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    image_size = (args.image_height, args.image_width)

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

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=args.drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def create_argparser() -> argparse.ArgumentParser:
    data_root = default_data_root()

    parser = argparse.ArgumentParser(
        description="DeepLabv3 training scaffold for BDD100K-style image/json pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(data_root / "100k_images"),
        help="Root folder that contains split subfolders like train/, val/, and test/ for images.",
    )
    parser.add_argument(
        "--json-root",
        type=str,
        default=str(data_root / "100k_json"),
        help="Root folder that contains split subfolders like train/, val/, and test/ for JSON annotations.",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Dataset split name for training.")
    parser.add_argument("--test-split", type=str, default="test", help="Dataset split name for evaluation/testing.")
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
    parser.add_argument("--output-stride", type=int, default=16, choices=[8, 16], help="DeepLabv3 output stride.")
    parser.add_argument("--base-lr", type=float, default=0.007, help="Initial learning rate.")
    parser.add_argument("--max-iters", type=int, default=30000, help="Maximum train iterations.")
    parser.add_argument("--bn-decay", type=float, default=0.9997, help="BatchNorm decay from the paper.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument("--lr-power", type=float, default=0.9, help="Poly LR exponent.")
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
    return parser


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_to_idx = load_or_create_category_map(
        json_root=args.json_root,
        category_map_path=args.category_map_path,
        split=args.train_split,
        auto_discover=not args.no_auto_discover_classes,
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
    )

    print(f"Using device: {device}")
    print(f"Image root:   {args.image_root}")
    print(f"JSON root:    {args.json_root}")
    print(f"Train split:  {args.train_split}")
    print(f"Test split:   {args.test_split}")
    print(f"Classes:      {len(class_to_idx)}")
    print(f"Background:   {idx_to_class[0]}")
    print(f"Model dir:    {args.model_dir}")
    print(f"Output dir:   {args.output_dir}")


    train_loader, test_loader = build_dataloaders(args, class_to_idx, device)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model = build_model(config).to(device)
    print(f"Model classes: {config.num_classes}")

    if args.smoke_test or not args.train:
        train_images, train_masks = next(iter(train_loader))
        test_images, test_masks = next(iter(test_loader))

        train_images = train_images.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(train_images)

        print(f"Train batch image shape: {tuple(train_images.shape)}")
        print(f"Train batch mask shape:  {tuple(train_masks.shape)}")
        print(f"Test batch image shape:  {tuple(test_images.shape)}")
        print(f"Test batch mask shape:   {tuple(test_masks.shape)}")
        print(f"Forward output shape:    {tuple(logits.shape)}")
        print(f"Class map saved to:      {args.category_map_path or Path(args.json_root) / 'category_map.json'}")

        if args.smoke_test or not args.train:
            return

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        save_file=args.save_file,
        device=device,
    )


if __name__ == "__main__":
    
    main()
