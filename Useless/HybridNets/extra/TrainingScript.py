import argparse
import json
import os
import random
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from backbone import HybridNetsBackbone
from hybridnets.model import ModelWithLoss
from utils.constants import MULTICLASS_MODE
from utils.utils import init_weights


#    # albumentations==1.1.0 \
                # efficientnet_pytorch==0.7.1 \
                # matplotlib \
                # numpy==1.26.4 \
                # opencv_python_headless==4.11.0.86 \
                # prefetch_generator==1.0.1 \
                # pretrainedmodels==0.7.4 \
                # psutil==5.9.0 \
                # PyYAML==6.0.2 \
                # scipy \
                # seaborn==0.11.2 \
                # tensorboardX==2.4.1 \
                # timm==0.5.4 \
                # tqdm==4.61.2 \
                # webcolors==1.11.1


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEGMENTATION_CLASSES = ("background", "road", "lane")
DETECTION_DISABLED_NUM_CLASSES = 1
PAPER_ANCHOR_SCALES = (2**0, 2**0.7, 2**1.32)
PAPER_ANCHOR_RATIOS = ((0.62, 1.58), (1.0, 1.0), (1.58, 0.62))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    road_mask_path: Path
    lane_mask_path: Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def default_num_gpus() -> int:
    raw = first_env("SM_NUM_GPUS")
    if raw is None:
        return 1 if torch.cuda.is_available() else 0
    try:
        return int(raw)
    except ValueError:
        return 1 if torch.cuda.is_available() else 0


def default_live_artifact_dir() -> str:
    checkpoint_env = first_env("SM_CHECKPOINT_DIR", "CHECKPOINT_DIR")
    if checkpoint_env:
        return checkpoint_env
    if os.environ.get("SM_CURRENT_HOST"):
        return "/opt/ml/checkpoints"
    return os.environ.get("SM_MODEL_DIR", "checkpoints/training_script")


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value:.1f} B"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, cls=NumpyEncoder, sort_keys=False)
    temp_path.replace(path)
    return path


def sync_json_targets(paths: list[Path], data) -> None:
    for path in paths:
        save_json(path, data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_tar_member(destination: Path, member_name: str) -> None:
    destination_resolved = destination.resolve()
    member_path = (destination / member_name).resolve()
    if os.path.commonpath((str(destination_resolved), str(member_path))) != str(destination_resolved):
        raise RuntimeError(f"Refusing to extract archive member outside target directory: {member_name}")


def extract_archives(root: Path) -> Path:
    if root.is_file():
        if not any(root.name.endswith(suffix) for suffix in (".tar", ".tar.gz", ".tgz")):
            raise ValueError(f"Bundle path is a file but not a supported tar archive: {root}")
        archives = [root]
        extraction_parent = root.parent
    else:
        archives = sorted(
            [
                path
                for path in root.iterdir()
                if path.is_file() and any(path.name.endswith(suffix) for suffix in (".tar", ".tar.gz", ".tgz"))
            ]
        )
        extraction_parent = root
    if not archives:
        return root

    extracted_root = extraction_parent / "_extracted"
    sentinel = extracted_root / ".complete"
    if sentinel.exists():
        return extracted_root

    extracted_root.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        print(f"Extracting {archive_path} -> {extracted_root}")
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                validate_tar_member(extracted_root, member.name)
            archive.extractall(extracted_root)
    sentinel.touch()
    return extracted_root


def dir_has_files(directory: Path, suffixes: Iterable[str]) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    for child in directory.iterdir():
        if child.is_file() and child.suffix.lower() in suffixes:
            return True
    return False


def is_split_root(candidate: Path, suffixes: Iterable[str], train_split: str, val_split: str) -> bool:
    return dir_has_files(candidate / train_split, suffixes) and dir_has_files(candidate / val_split, suffixes)


def discover_split_root(
    root: Path,
    suffixes: Iterable[str],
    train_split: str,
    val_split: str,
    preferred_children: tuple[str, ...] = (),
) -> Path:
    preferred_candidates: list[Path] = []
    seen_candidates: set[Path] = set()

    for child in preferred_children:
        direct_candidate = root / child
        if direct_candidate not in seen_candidates:
            preferred_candidates.append(direct_candidate)
            seen_candidates.add(direct_candidate)

        recursive_matches = sorted(
            (candidate for candidate in root.rglob(child) if candidate.is_dir()),
            key=lambda path: (len(path.parts), str(path)),
        )
        for candidate in recursive_matches:
            if candidate not in seen_candidates:
                preferred_candidates.append(candidate)
                seen_candidates.add(candidate)

    for candidate in preferred_candidates:
        if is_split_root(candidate, suffixes, train_split, val_split):
            return candidate

    if is_split_root(root, suffixes, train_split, val_split):
        return root

    matches: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_dir() and is_split_root(candidate, suffixes, train_split, val_split):
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"Could not find a directory under {root} containing both {train_split}/ and {val_split}/ "
            f"with files ending in {tuple(suffixes)}"
        )

    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def resolve_images_root(images_root: Path, fallback_root: Path | None, train_split: str, val_split: str) -> Path:
    candidates = [images_root]
    if fallback_root is not None:
        candidates.append(fallback_root)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return discover_split_root(
                candidate,
                IMAGE_EXTENSIONS,
                train_split,
                val_split,
                preferred_children=("imgs", "images", "100k_images", "100k"),
            )
        except Exception as exc:
            last_error = exc
            continue

    if last_error is None:
        raise FileNotFoundError("No valid image root was provided.")
    raise last_error


def locate_required_data(bundle_root: Path, train_split: str, val_split: str) -> tuple[Path, Path]:
    extracted_root = extract_archives(bundle_root)
    road_mask_root = discover_split_root(
        extracted_root,
        (".png",),
        train_split,
        val_split,
        preferred_children=("bdd_seg_gt", "da_seg_annot"),
    )
    lane_mask_root = discover_split_root(
        extracted_root,
        (".png",),
        train_split,
        val_split,
        preferred_children=("bdd_lane_gt", "ll_seg_annot"),
    )
    return road_mask_root, lane_mask_root


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_records(
    images_root: Path,
    road_mask_root: Path,
    lane_mask_root: Path,
    split: str,
) -> list[SampleRecord]:
    images_dir = images_root / split
    road_mask_dir = road_mask_root / split
    lane_mask_dir = lane_mask_root / split

    records: list[SampleRecord] = []
    missing_images = 0
    missing_road_masks = 0
    missing_lane_masks = 0

    lane_candidates = sorted(lane_mask_dir.glob("*.png"))
    for lane_mask_path in lane_candidates:
        stem = lane_mask_path.stem
        image_path = find_image_for_stem(images_dir, stem)
        road_mask_path = road_mask_dir / f"{stem}.png"

        if image_path is None:
            missing_images += 1
            continue
        if not road_mask_path.exists():
            missing_road_masks += 1
            continue
        if not lane_mask_path.exists():
            missing_lane_masks += 1
            continue

        records.append(
            SampleRecord(
                image_path=image_path,
                road_mask_path=road_mask_path,
                lane_mask_path=lane_mask_path,
            )
        )

    print(
        f"{split}: matched={len(records)} missing_images={missing_images} "
        f"missing_road_masks={missing_road_masks} missing_lane_masks={missing_lane_masks}"
    )
    if not records:
        raise RuntimeError(f"No usable samples were found for split '{split}'.")
    return records


def build_train_transforms(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                ],
                p=0.4,
            ),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=12, p=0.3),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=0.15,
            ),
            A.OneOf(
                [
                    A.Affine(
                        scale=(0.97, 1.03),
                        translate_percent={"x": (-0.04, 0.04), "y": (-0.02, 0.02)},
                        rotate=(-3, 3),
                        shear=(-2, 2),
                        fit_output=False,
                        p=1.0,
                    ),
                    A.Perspective(scale=(0.02, 0.04), keep_size=True, fit_output=False, p=1.0),
                ],
                p=0.25,
            ),
            A.Resize(height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=MEAN, std=STD),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=1.0,
            min_visibility=0.1,
        ),
    )


def build_eval_transforms(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=MEAN, std=STD),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=1.0,
            min_visibility=0.0,
        ),
    )


class HybridNetsDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        image_height: int,
        image_width: int,
        train: bool,
        debug: bool = False,
    ) -> None:
        self.records = records[:128] if debug else records
        self.train = train
        self.transform = (
            build_train_transforms(image_height, image_width)
            if train
            else build_eval_transforms(image_height, image_width)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        road_mask = cv2.imread(str(record.road_mask_path), cv2.IMREAD_GRAYSCALE)
        lane_mask = cv2.imread(str(record.lane_mask_path), cv2.IMREAD_GRAYSCALE)
        if road_mask is None:
            raise FileNotFoundError(f"Failed to read road mask: {record.road_mask_path}")
        if lane_mask is None:
            raise FileNotFoundError(f"Failed to read lane mask: {record.lane_mask_path}")

        segmentation = np.zeros(road_mask.shape, dtype=np.uint8)
        segmentation[road_mask > 0] = 1
        segmentation[lane_mask > 0] = 2

        transformed = self.transform(
            image=image,
            mask=segmentation,
            bboxes=[],
            class_labels=[],
        )

        transformed_boxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]
        annotations = np.zeros((len(transformed_boxes), 5), dtype=np.float32)
        if transformed_boxes:
            annotations[:, :4] = np.asarray(transformed_boxes, dtype=np.float32)
            annotations[:, 4] = np.asarray(transformed_labels, dtype=np.float32)

        image_tensor = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float()
        segmentation_tensor = torch.from_numpy(np.asarray(transformed["mask"], dtype=np.int64))
        annotations_tensor = torch.from_numpy(annotations)

        return image_tensor, annotations_tensor, segmentation_tensor, str(record.image_path)

    @staticmethod
    def collate_fn(batch):
        images, annotations, segmentations, paths = zip(*batch)
        max_num_annots = max(annotation.size(0) for annotation in annotations)
        if max_num_annots > 0:
            padded_annots = torch.full((len(annotations), max_num_annots, 5), -1.0, dtype=torch.float32)
            for index, annotation in enumerate(annotations):
                if annotation.size(0) > 0:
                    padded_annots[index, : annotation.size(0)] = annotation
        else:
            padded_annots = torch.full((len(annotations), 1, 5), -1.0, dtype=torch.float32)

        return {
            "img": torch.stack(images, dim=0),
            "annot": padded_annots,
            "segmentation": torch.stack(segmentations, dim=0).long(),
            "paths": list(paths),
        }


def update_confusion_matrix(confusion: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> None:
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    mask = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[mask].to(torch.int64) + predictions[mask].to(torch.int64)
    confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_segmentation_metrics(confusion: torch.Tensor) -> tuple[float, list[float], float, list[float]]:
    confusion = confusion.float()
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    support = confusion.sum(dim=1)

    iou_denominator = true_positive + false_positive + false_negative
    class_iou = torch.where(iou_denominator > 0, true_positive / iou_denominator, torch.zeros_like(true_positive))
    class_acc = torch.where(support > 0, true_positive / support, torch.zeros_like(true_positive))

    miou = class_iou.mean().item()
    macc = class_acc.mean().item()
    return miou, class_iou.tolist(), macc, class_acc.tolist()


def enrich_epoch_metrics(metrics: dict) -> dict:
    metrics = dict(metrics)
    class_support = metrics.get("class_support", [])
    class_iou = metrics.get("class_iou", [])
    class_acc = metrics.get("class_acc", [])
    metrics["class_iou_by_name"] = {
        name: float(value) for name, value in zip(SEGMENTATION_CLASSES, class_iou)
    }
    metrics["class_acc_by_name"] = {
        name: float(value) for name, value in zip(SEGMENTATION_CLASSES, class_acc)
    }
    metrics["class_support_by_name"] = {
        name: int(value) for name, value in zip(SEGMENTATION_CLASSES, class_support)
    }
    return metrics


def unwrap_model(model: nn.Module) -> ModelWithLoss:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    checkpoint_dir: Path,
    epoch: int,
    step: int,
    best_miou: float,
    name: str,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "best_miou": best_miou,
        "model": unwrap_model(model).model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = checkpoint_dir / name
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_weights(backbone: HybridNetsBackbone, weights_path: Path) -> dict:
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    load_result = backbone.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Missing keys while loading weights: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unexpected keys while loading weights: {load_result.unexpected_keys}")
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_model_weights(model: nn.Module, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).model.state_dict(), target_path)
    return target_path


def query_nvidia_smi() -> tuple[list[dict], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except FileNotFoundError as exc:
        return [], f"nvidia-smi not found: {exc}"
    except subprocess.CalledProcessError as exc:
        return [], f"nvidia-smi failed: {exc.output.strip()}"

    rows: list[dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "utilization_gpu_percent": float(parts[2]),
                "utilization_memory_percent": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_total_mb": float(parts[5]),
                "temperature_c": float(parts[6]),
            }
        )
    return rows, None


def collect_system_stats(device: torch.device) -> dict:
    process = psutil.Process(os.getpid())
    process_cpu_percent = process.cpu_percent(interval=None)
    cpu_percent = psutil.cpu_percent(interval=0.25)
    process_cpu_percent = process.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process_memory = process.memory_info()
    nvidia_rows, nvidia_error = query_nvidia_smi()
    nvidia_by_index = {row["index"]: row for row in nvidia_rows}

    stats = {
        "timestamp_utc": utc_now_iso(),
        "cpu": {
            "system_percent": cpu_percent,
            "process_percent": process_cpu_percent,
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        },
        "ram": {
            "used_bytes": ram.used,
            "available_bytes": ram.available,
            "total_bytes": ram.total,
            "percent": ram.percent,
        },
        "swap": {
            "used_bytes": swap.used,
            "total_bytes": swap.total,
            "percent": swap.percent,
        },
        "process": {
            "pid": process.pid,
            "rss_bytes": process_memory.rss,
            "vms_bytes": process_memory.vms,
            "threads": process.num_threads(),
        },
        "cuda": {
            "available": bool(device.type == "cuda" and torch.cuda.is_available()),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
            "nvidia_smi_error": nvidia_error,
        },
    }

    if stats["cuda"]["available"]:
        for index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            torch_stats = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "max_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
            torch_stats.update(nvidia_by_index.get(index, {}))
            stats["cuda"]["devices"].append(torch_stats)

    return stats


def log_system_stats(system_stats: dict) -> None:
    print("System telemetry:")
    cpu_stats = system_stats["cpu"]
    print(
        f"  CPU usage: system={cpu_stats['system_percent']:.1f}% "
        f"process={cpu_stats['process_percent']:.1f}% "
        f"logical={cpu_stats['logical_count']} physical={cpu_stats['physical_count']}"
    )
    if cpu_stats["load_avg"] is not None:
        print(
            "  CPU load avg: "
            + ", ".join(f"{value:.2f}" for value in cpu_stats["load_avg"])
        )

    ram = system_stats["ram"]
    print(
        f"  RAM usage: used={human_bytes(ram['used_bytes'])} "
        f"available={human_bytes(ram['available_bytes'])} total={human_bytes(ram['total_bytes'])} "
        f"percent={ram['percent']:.1f}%"
    )

    swap = system_stats["swap"]
    print(
        f"  Swap usage: used={human_bytes(swap['used_bytes'])} total={human_bytes(swap['total_bytes'])} "
        f"percent={swap['percent']:.1f}%"
    )

    process = system_stats["process"]
    print(
        f"  Process memory: rss={human_bytes(process['rss_bytes'])} "
        f"vms={human_bytes(process['vms_bytes'])} threads={process['threads']}"
    )

    cuda = system_stats["cuda"]
    if cuda["available"]:
        if cuda["nvidia_smi_error"]:
            print(f"  CUDA usage: nvidia-smi unavailable ({cuda['nvidia_smi_error']})")
        for gpu in cuda["devices"]:
            print(
                f"  GPU {gpu['index']} {gpu['name']}: "
                f"util={gpu.get('utilization_gpu_percent', -1):.1f}% "
                f"mem_util={gpu.get('utilization_memory_percent', -1):.1f}% "
                f"temp={gpu.get('temperature_c', -1):.1f}C "
                f"used={human_bytes(gpu.get('memory_used_mb', 0) * 1024 * 1024)} "
                f"/ {human_bytes(gpu.get('memory_total_mb', 0) * 1024 * 1024)}"
            )
            print(
                f"    torch CUDA: free={human_bytes(gpu['free_bytes'])} "
                f"allocated={human_bytes(gpu['allocated_bytes'])} "
                f"reserved={human_bytes(gpu['reserved_bytes'])} "
                f"max_allocated={human_bytes(gpu['max_allocated_bytes'])} "
                f"max_reserved={human_bytes(gpu['max_reserved_bytes'])}"
            )
            torch.cuda.reset_peak_memory_stats(gpu["index"])
    else:
        print("  CUDA usage: not active")


def move_batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["img"].to(device, non_blocking=True)
    if device.type == "cuda":
        images = images.to(memory_format=torch.channels_last)
    annotations = batch["annot"].to(device, non_blocking=True)
    segmentations = batch["segmentation"].to(device, non_blocking=True)
    return images, annotations, segmentations


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    freeze_det: bool,
    freeze_seg: bool,
) -> dict:
    is_training = optimizer is not None
    num_classes = len(SEGMENTATION_CLASSES)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    totals = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "seg": 0.0}
    steps = 0
    start_time = time.perf_counter()

    model.train(mode=is_training)
    progress = tqdm(loader, desc=("Train" if is_training else "Val") + f" {epoch}/{total_epochs}", ascii=True)

    def forward_pass(autocast_enabled: bool):
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=autocast_enabled):
            cls_loss, reg_loss, seg_loss, _, _, _, segmentation_logits = model(
                images,
                annotations,
                segmentations,
                obj_list=None,
                skip_detection_loss=freeze_det,
            )
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            seg_loss = seg_loss.mean()

            if freeze_det:
                cls_loss = cls_loss * 0.0
                reg_loss = reg_loss * 0.0
            if freeze_seg:
                seg_loss = seg_loss * 0.0

            loss = cls_loss + reg_loss + seg_loss
        return loss, cls_loss, reg_loss, seg_loss, segmentation_logits

    for batch in progress:
        images, annotations, segmentations = move_batch_to_device(batch, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        loss, cls_loss, reg_loss, seg_loss, segmentation_logits = forward_pass(autocast_enabled=use_amp)

        if not torch.isfinite(loss):
            component_state = {
                "loss": float(loss.detach().cpu()),
                "cls_loss": float(cls_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "seg_loss": float(seg_loss.detach().cpu()),
            }
            if use_amp and device.type == "cuda":
                print(
                    f"Non-finite loss detected under AMP at epoch {epoch}; "
                    f"retrying batch in full precision. Components: {component_state}"
                )
                loss, cls_loss, reg_loss, seg_loss, segmentation_logits = forward_pass(autocast_enabled=False)
                component_state = {
                    "loss": float(loss.detach().cpu()),
                    "cls_loss": float(cls_loss.detach().cpu()),
                    "reg_loss": float(reg_loss.detach().cpu()),
                    "seg_loss": float(seg_loss.detach().cpu()),
                }

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Encountered non-finite loss at epoch {epoch}. "
                f"loss={component_state['loss']}, cls_loss={component_state['cls_loss']}, "
                f"reg_loss={component_state['reg_loss']}, seg_loss={component_state['seg_loss']}"
            )

        if is_training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = segmentation_logits.detach().argmax(dim=1).cpu()
        update_confusion_matrix(confusion, predictions, segmentations.detach().cpu(), num_classes)

        totals["loss"] += loss.item()
        totals["cls"] += cls_loss.item()
        totals["reg"] += reg_loss.item()
        totals["seg"] += seg_loss.item()
        steps += 1

        miou, class_iou, _, _ = compute_segmentation_metrics(confusion)
        progress.set_postfix(
            loss=f"{totals['loss'] / steps:.4f}",
            cls=f"{totals['cls'] / steps:.4f}",
            reg=f"{totals['reg'] / steps:.4f}",
            seg=f"{totals['seg'] / steps:.4f}",
            miou=f"{miou:.4f}",
            road_iou=f"{class_iou[1]:.4f}",
            lane_iou=f"{class_iou[2]:.4f}",
        )

    miou, class_iou, macc, class_acc = compute_segmentation_metrics(confusion)
    elapsed_seconds = time.perf_counter() - start_time
    class_support = confusion.sum(dim=1).tolist()
    return {
        "loss": totals["loss"] / max(steps, 1),
        "cls_loss": totals["cls"] / max(steps, 1),
        "reg_loss": totals["reg"] / max(steps, 1),
        "seg_loss": totals["seg"] / max(steps, 1),
        "miou": miou,
        "macc": macc,
        "class_iou": class_iou,
        "class_acc": class_acc,
        "class_support": class_support,
        "confusion_matrix": confusion.tolist(),
        "num_batches": steps,
        "num_samples": len(loader.dataset),
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": len(loader.dataset) / max(elapsed_seconds, 1e-8),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train HybridNets with SageMaker tar-based inputs")
    parser.add_argument("--images-root", default=first_env("SM_CHANNEL_IMAGES", "SM_CHANNEL_100K_IMAGES"))
    parser.add_argument("--images-fallback-root", default=first_env("SM_CHANNEL_IMAGES_FALLBACK", "SM_CHANNEL_100K"))
    parser.add_argument(
        "--bundle-root",
        default=first_env("SM_CHANNEL_HYBRIDNETS_DATA", "SM_CHANNEL_LABEL_BUNDLE", "SM_CHANNEL_DATA"),
        help="Channel containing hybridNets_data.tar or extracted bdd_lane_gt/bdd_seg_gt/data2 folders.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--compound-coef", type=int, default=3) # hieghlighting which EfficentNet model, we are gonna use EfficentNet-B2 
    parser.add_argument("--backbone", type=str, default=None) 
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--freeze-backbone", type=parse_bool, default=False)
    parser.add_argument("--freeze-det", type=parse_bool, default=True)
    parser.add_argument("--freeze-seg", type=parse_bool, default=False)
    parser.add_argument("--num-gpus", type=int, default=default_num_gpus())
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--save-dir", default=default_live_artifact_dir())
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/model"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "artifacts/output"))
    parser.add_argument("--debug", type=parse_bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    # Parser
    args = build_parser().parse_args()
    # Checking if there are images
    if not args.images_root:
        raise ValueError("Missing --images-root / SM_CHANNEL_IMAGES.")
    if not args.bundle_root:
        raise ValueError("Missing --bundle-root / SM_CHANNEL_HYBRIDNETS_DATA.")

    # setting cuda seed (doesn't matter )
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Resolving dataset roots and artifact directories
    images_root_arg = Path(args.images_root).expanduser().resolve()
    images_fallback_root_arg = Path(args.images_fallback_root).expanduser().resolve() if args.images_fallback_root else None
    bundle_root_arg = Path(args.bundle_root).expanduser().resolve()
    live_artifact_dir = Path(args.save_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = live_artifact_dir / "checkpoints"
    metrics_dir = live_artifact_dir / "metrics"
    output_metrics_dir = output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_metrics_dir.mkdir(parents=True, exist_ok=True)

    images_root = resolve_images_root(images_root_arg, images_fallback_root_arg, args.train_split, args.val_split)
    road_mask_root, lane_mask_root = locate_required_data(bundle_root_arg, args.train_split, args.val_split)
    args.freeze_det = True

    print("Resolved dataset roots:")
    print(f"  images:      {images_root}")
    print(f"  road masks:  {road_mask_root}")
    print(f"  lane masks:  {lane_mask_root}")
    print("Detection branch: disabled for road-only training")
    print("Segmentation remap:")
    print("  background -> 0")
    print("  all non-zero drivable-mask pixels -> road (1)")
    print("  all non-zero lane-mask pixels -> lane (2), overriding road on overlap")
    print("Artifact directories:")
    print(f"  live save dir: {live_artifact_dir}")
    print(f"  checkpoints:   {checkpoint_dir}")
    print(f"  metrics:       {metrics_dir}")
    print(f"  model dir:     {model_dir}")
    print(f"  output dir:    {output_dir}")

    train_records = build_records(images_root, road_mask_root, lane_mask_root, args.train_split)
    val_records = build_records(images_root, road_mask_root, lane_mask_root, args.val_split)

    train_dataset = HybridNetsDataset(
        records=train_records,
        image_height=args.image_height,
        image_width=args.image_width,
        train=True,
        debug=args.debug,
    )
    val_dataset = HybridNetsDataset(
        records=val_records,
        image_height=args.image_height,
        image_width=args.image_width,
        train=False,
        debug=args.debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=HybridNetsDataset.collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=HybridNetsDataset.collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    if args.num_gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")

    model = HybridNetsBackbone(
        num_classes=DETECTION_DISABLED_NUM_CLASSES,
        compound_coef=args.compound_coef,
        ratios=PAPER_ANCHOR_RATIOS,
        scales=PAPER_ANCHOR_SCALES,
        seg_classes=2,
        backbone_name=args.backbone,
        seg_mode=MULTICLASS_MODE,
    )

    checkpoint_state = {}
    if args.load_weights:
        checkpoint_state = load_weights(model, Path(args.load_weights).expanduser().resolve())
    else:
        print("Initializing model weights from scratch.")
        init_weights(model)

    if args.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        print("Backbone frozen.")
    if args.freeze_det:
        model.regressor.requires_grad_(False)
        model.classifier.requires_grad_(False)
        print("Detection heads frozen.")
    if args.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        print("Segmentation head frozen.")

    wrapped_model = ModelWithLoss(model, debug=args.debug)
    wrapped_model = wrapped_model.to(device)
    if device.type == "cuda":
        wrapped_model = wrapped_model.to(memory_format=torch.channels_last)
        if args.num_gpus > 1 and torch.cuda.device_count() > 1:
            visible_gpu_count = min(args.num_gpus, torch.cuda.device_count())
            wrapped_model = nn.DataParallel(wrapped_model, device_ids=list(range(visible_gpu_count)))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in wrapped_model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        verbose=True,
    )

    if checkpoint_state.get("optimizer"):
        optimizer.load_state_dict(checkpoint_state["optimizer"])
    if checkpoint_state.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(checkpoint_state["scaler"])

    best_miou = float(checkpoint_state.get("best_miou", 0.0))
    global_step = int(checkpoint_state.get("step", 0))
    train_history: list[dict] = []
    val_history: list[dict] = []
    system_history: list[dict] = []
    epoch_history: list[dict] = []

    run_config = {
        "timestamp_utc": utc_now_iso(),
        "args": vars(args),
        "device": str(device),
        "segmentation_classes": list(SEGMENTATION_CLASSES),
        "resolved_roots": {
            "images": images_root,
            "road_masks": road_mask_root,
            "lane_masks": lane_mask_root,
            "bundle_root": bundle_root_arg,
        },
    }
    dataset_summary = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "image_size": {"height": args.image_height, "width": args.image_width},
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "debug": bool(args.debug),
        "train_split": args.train_split,
        "val_split": args.val_split,
    }
    initial_system_stats = collect_system_stats(device)
    sync_json_targets(
        [
            metrics_dir / "run_config.json",
            output_metrics_dir / "run_config.json",
            model_dir / "run_config.json",
        ],
        run_config,
    )
    sync_json_targets(
        [
            metrics_dir / "dataset_summary.json",
            output_metrics_dir / "dataset_summary.json",
            model_dir / "dataset_summary.json",
        ],
        dataset_summary,
    )
    sync_json_targets(
        [
            metrics_dir / "system_metrics_initial.json",
            output_metrics_dir / "system_metrics_initial.json",
            model_dir / "system_metrics_initial.json",
        ],
        initial_system_stats,
    )
    print("Initial system telemetry:")
    log_system_stats(initial_system_stats)

    for epoch in range(1, args.epochs + 1):
        current_lr_before = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model=wrapped_model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=bool(args.amp and device.type == "cuda"),
            epoch=epoch,
            total_epochs=args.epochs,
            freeze_det=args.freeze_det,
            freeze_seg=args.freeze_seg,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=wrapped_model,
                loader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                use_amp=bool(args.amp and device.type == "cuda"),
                epoch=epoch,
                total_epochs=args.epochs,
                freeze_det=args.freeze_det,
                freeze_seg=args.freeze_seg,
            )

        train_metrics = enrich_epoch_metrics(train_metrics)
        val_metrics = enrich_epoch_metrics(val_metrics)
        global_step += len(train_loader)
        scheduler.step(val_metrics["miou"])
        current_lr_after = optimizer.param_groups[0]["lr"]
        current_best_miou = max(best_miou, val_metrics["miou"])

        epoch_weights = save_model_weights(
            model=wrapped_model,
            target_path=checkpoint_dir / f"hybridnets_epoch_{epoch:03d}_weights.pth",
        )
        latest_resume = save_checkpoint(
            model=wrapped_model,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            step=global_step,
            best_miou=current_best_miou,
            name="hybridnets_latest_resume.pth",
        )
        latest_weights = save_model_weights(
            model=wrapped_model,
            target_path=checkpoint_dir / "hybridnets_latest_weights.pth",
        )

        if val_metrics["miou"] >= best_miou:
            best_miou = val_metrics["miou"]
            best_resume = save_checkpoint(
                model=wrapped_model,
                optimizer=optimizer,
                scaler=scaler,
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                step=global_step,
                best_miou=best_miou,
                name="hybridnets_best_miou_resume.pth",
            )
            best_weights = save_model_weights(
                model=wrapped_model,
                target_path=checkpoint_dir / "hybridnets_best_miou_weights.pth",
            )
            print(f"Best checkpoint updated: {best_resume}")
            print(f"Best weights updated: {best_weights}")

        system_stats = collect_system_stats(device)
        system_stats["epoch"] = epoch
        system_stats["global_step"] = global_step
        train_metrics["epoch"] = epoch
        train_metrics["global_step"] = global_step
        train_metrics["lr_before_scheduler"] = current_lr_before
        train_metrics["lr_after_scheduler"] = current_lr_after
        val_metrics["epoch"] = epoch
        val_metrics["global_step"] = global_step
        val_metrics["lr_before_scheduler"] = current_lr_before
        val_metrics["lr_after_scheduler"] = current_lr_after

        epoch_summary = {
            "timestamp_utc": utc_now_iso(),
            "epoch": epoch,
            "global_step": global_step,
            "best_miou": best_miou,
            "lr_before_scheduler": current_lr_before,
            "lr_after_scheduler": current_lr_after,
            "train": train_metrics,
            "val": val_metrics,
            "system": system_stats,
            "artifacts": {
                "epoch_weights": epoch_weights,
                "latest_resume": latest_resume,
                "latest_weights": latest_weights,
                "best_weights": checkpoint_dir / "hybridnets_best_miou_weights.pth",
                "best_resume": checkpoint_dir / "hybridnets_best_miou_resume.pth",
            },
        }
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        system_history.append(system_stats)
        epoch_history.append(epoch_summary)

        sync_json_targets(
            [
                metrics_dir / f"train_metrics_epoch_{epoch:03d}.json",
                output_metrics_dir / f"train_metrics_epoch_{epoch:03d}.json",
            ],
            train_metrics,
        )
        sync_json_targets(
            [
                metrics_dir / f"val_metrics_epoch_{epoch:03d}.json",
                output_metrics_dir / f"val_metrics_epoch_{epoch:03d}.json",
            ],
            val_metrics,
        )
        sync_json_targets(
            [
                metrics_dir / f"system_metrics_epoch_{epoch:03d}.json",
                output_metrics_dir / f"system_metrics_epoch_{epoch:03d}.json",
            ],
            system_stats,
        )
        sync_json_targets(
            [
                metrics_dir / "train_metrics_history.json",
                output_metrics_dir / "train_metrics_history.json",
                model_dir / "train_metrics_history.json",
            ],
            train_history,
        )
        sync_json_targets(
            [
                metrics_dir / "val_metrics_history.json",
                output_metrics_dir / "val_metrics_history.json",
                model_dir / "val_metrics_history.json",
            ],
            val_history,
        )
        sync_json_targets(
            [
                metrics_dir / "system_metrics_history.json",
                output_metrics_dir / "system_metrics_history.json",
                model_dir / "system_metrics_history.json",
            ],
            system_history,
        )
        sync_json_targets(
            [
                metrics_dir / "scheduler_metrics.json",
                output_metrics_dir / "scheduler_metrics.json",
                model_dir / "scheduler_metrics.json",
            ],
            scheduler.state_dict(),
        )
        sync_json_targets(
            [
                metrics_dir / f"epoch_summary_{epoch:03d}.json",
                output_metrics_dir / f"epoch_summary_{epoch:03d}.json",
                model_dir / f"epoch_summary_{epoch:03d}.json",
                metrics_dir / "latest_summary.json",
                output_metrics_dir / "latest_summary.json",
                model_dir / "latest_summary.json",
            ],
            epoch_summary,
        )

        print(
            f"Epoch {epoch}/{args.epochs} summary | "
            f"train_loss={train_metrics['loss']:.4f} train_miou={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['miou']:.4f} "
            f"road_iou={val_metrics['class_iou'][1]:.4f} lane_iou={val_metrics['class_iou'][2]:.4f}"
        )
        print(
            f"  Timing: train={train_metrics['elapsed_seconds']:.2f}s "
            f"val={val_metrics['elapsed_seconds']:.2f}s "
            f"train_samples_per_sec={train_metrics['samples_per_second']:.2f} "
            f"val_samples_per_sec={val_metrics['samples_per_second']:.2f}"
        )
        print(
            f"  Optimizer LR: before_scheduler={current_lr_before:.8f} "
            f"after_scheduler={current_lr_after:.8f}"
        )
        print(
            "  Per-class IoU: "
            + ", ".join(
                f"{name}={value:.4f}" for name, value in zip(SEGMENTATION_CLASSES, val_metrics["class_iou"])
            )
        )
        print(
            "  Per-class Acc: "
            + ", ".join(
                f"{name}={value:.4f}" for name, value in zip(SEGMENTATION_CLASSES, val_metrics["class_acc"])
            )
        )
        print(
            "  Class support: "
            + ", ".join(
                f"{name}={value}" for name, value in zip(SEGMENTATION_CLASSES, val_metrics["class_support"])
            )
        )
        print(f"  Saved epoch weights: {epoch_weights}")
        print(f"  Saved latest resume checkpoint: {latest_resume}")
        print(f"  Saved latest weights: {latest_weights}")
        print(f"  Metrics JSON written under: {metrics_dir}")
        log_system_stats(system_stats)

    final_live_weights = save_model_weights(
        wrapped_model,
        checkpoint_dir / "hybridnets_final_weights.pth",
    )
    final_model_weights = save_model_weights(
        wrapped_model,
        model_dir / "hybridnets_final_weights.pth",
    )
    final_output_weights = save_model_weights(
        wrapped_model,
        output_dir / "hybridnets_final_weights.pth",
    )

    best_live_weights = checkpoint_dir / "hybridnets_best_miou_weights.pth"
    latest_live_weights = checkpoint_dir / "hybridnets_latest_weights.pth"
    if best_live_weights.exists():
        shutil.copy2(best_live_weights, model_dir / "hybridnets_best_miou_weights.pth")
        shutil.copy2(best_live_weights, output_dir / "hybridnets_best_miou_weights.pth")
    if latest_live_weights.exists():
        shutil.copy2(latest_live_weights, model_dir / "hybridnets_latest_weights.pth")
        shutil.copy2(latest_live_weights, output_dir / "hybridnets_latest_weights.pth")

    final_summary = {
        "timestamp_utc": utc_now_iso(),
        "best_miou": best_miou,
        "epochs_completed": args.epochs,
        "global_step": global_step,
        "live_checkpoint_dir": checkpoint_dir,
        "metrics_dir": metrics_dir,
        "model_dir": model_dir,
        "output_dir": output_dir,
        "final_live_weights": final_live_weights,
        "final_model_weights": final_model_weights,
        "final_output_weights": final_output_weights,
    }
    sync_json_targets(
        [
            metrics_dir / "final_summary.json",
            output_metrics_dir / "final_summary.json",
            model_dir / "final_summary.json",
        ],  
        final_summary,
    )
    print(f"Final live weights written to {final_live_weights}")
    print(f"Final model artifact weights written to {final_model_weights}")


if __name__ == "__main__":
    main()
