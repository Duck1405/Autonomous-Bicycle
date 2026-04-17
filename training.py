import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader

import argparse
import os
import copy
import json
import gc

import pandas as pd
import numpy as np
from PIL import Image
import cv2


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.ops import StochasticDepth
from torch.cuda.amp import GradScaler, autocast
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

import timm
import psutil
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    log_loss,
)


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


@dataclass
class TrainConfig:
    num_classes: int = 19
    backbone: str = "resnet101"
    output_stride: int = 16
    crop_size: int = 513
    batch_size: int = 16
    max_iters: int = 30000
    base_lr: float = 0.007
    lr_power: float = 0.9
    momentum: float = 0.9
    weight_decay: float = 1e-4
    bn_decay: float = 0.9997
    ignore_index: int = 255
    device: str = "cpu"


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


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def fit(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
    device: str | torch.device = "cpu",
) -> None:
    device = torch.device(device)
    model.to(device)
    global_step = 0

    while global_step < config.max_iters:
        for images, targets in dataloader:
            if global_step >= config.max_iters:
                break

            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            current_lr = poly_learning_rate(
                base_lr=config.base_lr,
                current_iter=global_step,
                max_iters=config.max_iters,
                power=config.lr_power,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            loss = train_step(model, images, targets, optimizer, criterion)

            if global_step % 50 == 0:
                print(f"iter={global_step:05d} lr={current_lr:.6f} loss={loss:.4f}")

            global_step += 1


def build_dataloader():
    raise NotImplementedError(
        "Dataset and DataLoader wiring are intentionally left out. "
        "Plug your dataset in here once your labels and transforms are ready."
    )


if __name__ == "__main__":
    print("Started")
    config = TrainConfig()
    print("config:")
    print(config)
    device = get_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(config)
    
    print(f"device: {device}")
    model.to(device)
    
    parser = argparse.ArgumentParser(
        description="Train model for Animal Classification (Local & SageMaker)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-size", type=int, default=513, help="Input image size (224 or 384)"
    )
    # parser.add_argument(
    #     "--data-dir",
    #     type=str,
    #     default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
    #     help="Directory containing training data (train_labels.csv and train_features/)",
    # )
    parser.add_argument(
        "--data-image-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
        help="Directory containing training data (train_labels.csv and train_features/)",
    )
    parser.add_argument(
        "--data-json-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
        help="Directory containing training data (train_labels.csv and train_features/)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "./models"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
        help="Directory to save metrics and logs",
    )

    parser.add_argument(
        "--save-file",
        type=str,
        default="final_model.pth",
        help="Filename for final saved model",
    )
    args = parser.parse_args()
    # sys_cpus = int(os.environ.get("SM_NUM_CPUS", os.cpu_count()))
    # sys_gpus = int(os.environ.get("SM_NUM_GPUS", 1))
    # Divide CPU workers across GPU processes to avoid oversubscription
    print(f"Image directory: {args.data_image_dir}")
    print(f"Json directory: {args.data_json_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Save file: {args.save_file}")
    
    if os.path.exists(args.data_image_dir):
        print("Image directory exists")
    if os.path.exists(args.data_json_dir):
        print("Json directory exists")
    
    