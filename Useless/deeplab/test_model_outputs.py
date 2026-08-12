from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_INPUT_DIR = Path("test_images_jpg")
DEFAULT_OUTPUT_DIR = Path("test_iamges_jpg") / "test1"
DEFAULT_CHECKPOINT = (
    Path("runs")
    / "deeplabv3_resnet50_drivable_small"
    / "checkpoints"
    / "best_deeplabv3_resnet50_drivable_small.pth"
)
DEFAULT_CATEGORY_MAP = Path("100k") / "category_map.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if hasattr(Image, "Resampling"):
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
    NEAREST_RESAMPLE = Image.Resampling.NEAREST
else:
    BILINEAR_RESAMPLE = Image.BILINEAR
    NEAREST_RESAMPLE = Image.NEAREST


def get_device(preferred_device: str = "cuda") -> torch.device:
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


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
            raise ValueError("output_stride must be 8 or 16.")

        self.in_channels = 64
        self.bn_momentum = bn_momentum

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
                raise ValueError("multi_grid must match the number of blocks or use a 3-value pattern.")

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
        backbone: str = "resnet50",
        output_stride: int = 16,
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


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained DeepLabV3 model on images and save visual outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder of test images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to store masks, overlays, and summaries.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--category-map",
        type=Path,
        default=DEFAULT_CATEGORY_MAP,
        help="JSON class-to-index mapping used during training.",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "resnet101"],
        default="resnet50",
        help="Backbone used by the saved checkpoint.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=320,
        help="Inference resize height before the model forward pass.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="Inference resize width before the model forward pass.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default="cuda",
        help="Preferred inference device.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay strength for predicted regions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    return parser


def load_category_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        class_to_idx = {str(name): int(index) for name, index in json.load(handle).items()}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def list_images(input_dir: Path, limit: int | None) -> list[Path]:
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    load_kwargs: dict[str, object] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True

    checkpoint = torch.load(path, **load_kwargs)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        cleaned[key.replace("module.", "", 1)] = value
    return cleaned


def build_inference_model(
    checkpoint_path: Path,
    backbone: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = DeepLabV3(num_classes=num_classes, backbone=backbone, output_stride=16, bn_decay=0.9997)
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image, image_height: int, image_width: int) -> torch.Tensor:
    resized = image.resize((image_width, image_height), resample=BILINEAR_RESAMPLE)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array).float().unsqueeze(0)


def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    image_height: int,
    image_width: int,
    device: torch.device,
) -> np.ndarray:
    input_tensor = preprocess_image(image, image_height=image_height, image_width=image_width)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask_image = Image.fromarray(prediction, mode="L").resize(image.size, resample=NEAREST_RESAMPLE)
    return np.asarray(mask_image, dtype=np.uint8)


def class_color(class_name: str, class_index: int) -> tuple[int, int, int]:
    named_colors = {
        "background": (0, 0, 0),
        "area/alternative": (255, 191, 0),
        "area/drivable": (0, 200, 0),
    }
    if class_name in named_colors:
        return named_colors[class_name]

    seed = (class_index * 97) % 255
    return ((seed + 80) % 255, (seed + 150) % 255, (seed + 220) % 255)


def colorize_mask(mask: np.ndarray, idx_to_class: dict[int, str]) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, class_name in idx_to_class.items():
        color_mask[mask == class_index] = class_color(class_name, class_index)
    return color_mask


def overlay_mask(image: Image.Image, mask: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    overlay = image_array.copy()
    active = mask > 0
    overlay[active] = ((1.0 - alpha) * image_array[active]) + (alpha * color_mask[active])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def summarize_mask(mask: np.ndarray, idx_to_class: dict[int, str]) -> dict[str, dict[str, float | int]]:
    total_pixels = int(mask.size)
    summary: dict[str, dict[str, float | int]] = {}
    unique_values, counts = np.unique(mask, return_counts=True)
    count_lookup = {int(value): int(count) for value, count in zip(unique_values, counts)}

    for class_index, class_name in idx_to_class.items():
        pixel_count = count_lookup.get(class_index, 0)
        summary[class_name] = {
            "class_index": class_index,
            "pixel_count": pixel_count,
            "pixel_ratio": (pixel_count / total_pixels) if total_pixels > 0 else 0.0,
        }
    return summary


def sanitize_column_name(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def save_results(
    image_path: Path,
    mask: np.ndarray,
    idx_to_class: dict[int, str],
    output_dir: Path,
    alpha: float,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    color_mask = colorize_mask(mask, idx_to_class)
    overlay = overlay_mask(image, mask, color_mask, alpha=alpha)

    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "masks"
    color_masks_dir = output_dir / "masks_color"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    color_masks_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    overlay_path = overlays_dir / f"{stem}_overlay.png"
    mask_path = masks_dir / f"{stem}_mask.png"
    color_mask_path = color_masks_dir / f"{stem}_mask_color.png"

    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    Image.fromarray(color_mask).save(color_mask_path)

    class_summary = summarize_mask(mask, idx_to_class)
    return {
        "image": str(image_path),
        "overlay": str(overlay_path),
        "mask": str(mask_path),
        "color_mask": str(color_mask_path),
        "classes": class_summary,
    }


def write_summary_files(results: list[dict[str, object]], idx_to_class: dict[int, str], output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = ["image", "overlay", "mask", "color_mask"]
    for class_index in sorted(idx_to_class):
        class_name = idx_to_class[class_index]
        column_prefix = sanitize_column_name(class_name)
        fieldnames.append(f"{column_prefix}_pixel_count")
        fieldnames.append(f"{column_prefix}_pixel_ratio")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "image": result["image"],
                "overlay": result["overlay"],
                "mask": result["mask"],
                "color_mask": result["color_mask"],
            }
            classes = result["classes"]
            for class_index in sorted(idx_to_class):
                class_name = idx_to_class[class_index]
                column_prefix = sanitize_column_name(class_name)
                class_values = classes[class_name]
                row[f"{column_prefix}_pixel_count"] = class_values["pixel_count"]
                row[f"{column_prefix}_pixel_ratio"] = f"{class_values['pixel_ratio']:.6f}"
            writer.writerow(row)


def main() -> None:
    args = create_argparser().parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.category_map.exists():
        raise FileNotFoundError(f"Category map not found: {args.category_map}")

    class_to_idx, idx_to_class = load_category_map(args.category_map)
    image_paths = list_images(args.input_dir, args.limit)
    if not image_paths:
        raise ValueError(f"No supported images found in {args.input_dir}")

    device = get_device(args.device)
    model = build_inference_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        num_classes=len(class_to_idx),
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    results: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        mask = predict_mask(
            model=model,
            image=image,
            image_height=args.image_height,
            image_width=args.image_width,
            device=device,
        )
        result = save_results(
            image_path=image_path,
            mask=mask,
            idx_to_class=idx_to_class,
            output_dir=args.output_dir,
            alpha=args.alpha,
        )
        results.append(result)

        drivable_ratio = result["classes"].get("area/drivable", {}).get("pixel_ratio", 0.0)
        print(f"[{index}/{len(image_paths)}] {image_path.name} drivable_ratio={drivable_ratio:.4f}")

    write_summary_files(results, idx_to_class=idx_to_class, output_dir=args.output_dir)
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
