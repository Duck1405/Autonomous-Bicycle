from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import tarfile
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from train import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TrainConfig,
    build_model,
    build_resource_lines,
    extract_segmentation_logits,
)


DEFAULT_MODEL_TAR = Path("/home/aman/Projects/Auto/models/model.tar.gz")
DEFAULT_VIDEO = Path("/home/aman/Projects/Auto/test_videos/mp4Videos/new_video2.mp4")
DEFAULT_OUTPUT = Path("/home/aman/Projects/Auto/test_videos/mp4Videos/new_video2_deeplab_lane_border.mp4")
DEFAULT_CATEGORY_MAP = Path("/home/aman/Projects/Auto/Autonomous-Bicycle/configs/lane_border_merged_category_map.json")
DEFAULT_CACHE_ROOT = Path("/tmp/autonomous_bike_model_cache")
LANE_COLOR = (0, 215, 255)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DeepLabV3 checkpoint bundle on a video and save the overlay.")
    parser.add_argument("--model-tar", type=Path, default=DEFAULT_MODEL_TAR, help="Path to the SageMaker model.tar.gz bundle.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional direct .pth checkpoint path. Overrides --model-tar.")
    parser.add_argument("--category-map", type=Path, default=DEFAULT_CATEGORY_MAP, help="Path to category_map.json.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Annotated output video path.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="Extraction cache for model.tar.gz.")
    parser.add_argument("--backbone", type=str, default="auto", choices=["auto", "resnet50", "resnet101"], help="Checkpoint backbone.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device.")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated CUDA GPU ids, e.g. 0 or 0,1.")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per GPU per inference chunk for file input.")
    parser.add_argument("--image-width", type=int, default=1024, help="Model input width used during training.")
    parser.add_argument("--image-height", type=int, default=512, help="Model input height used during training.")
    parser.add_argument("--output-width", type=int, default=0, help="Optional output width. 0 preserves source width.")
    parser.add_argument("--output-height", type=int, default=0, help="Optional output height. 0 preserves source height.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha for lane-border pixels.")
    parser.add_argument("--stats-every", type=int, default=120, help="Print stats every N frames.")
    parser.add_argument("--save-mask-video", type=str2bool, default=False, help="Also save a mask-only video.")
    parser.add_argument("--mask-output", type=Path, default=None, help="Mask-only output path.")
    return parser


def resolve_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda:0")
    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but is not available in this environment.")
        return torch.device("mps")
    return torch.device("cpu")


def parse_gpu_ids(value: str) -> list[int]:
    ids: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        ids.append(int(piece))
    return ids


def resolve_inference_devices(device_name: str, gpu_ids: str) -> list[torch.device]:
    primary_device = resolve_device(device_name)
    if primary_device.type != "cuda":
        return [primary_device]

    visible = torch.cuda.device_count()
    if visible <= 0:
        raise RuntimeError("CUDA selected but torch reports zero visible GPUs.")

    requested_ids = parse_gpu_ids(gpu_ids) or [0]
    valid_ids = [gpu_id for gpu_id in requested_ids if 0 <= gpu_id < visible]
    if not valid_ids:
        raise RuntimeError(f"No valid GPU ids found in {requested_ids}; visible GPU count is {visible}.")
    return [torch.device(f"cuda:{gpu_id}") for gpu_id in valid_ids]


def format_device_label(devices: Sequence[torch.device]) -> str:
    if len(devices) == 1:
        device = devices[0]
        return f"CUDA:{device.index}" if device.type == "cuda" else device.type.upper()
    labels = ", ".join(f"cuda:{device.index}" for device in devices)
    return f"CUDA x{len(devices)} ({labels})"


def cache_dir_for_tar(model_tar: Path, cache_root: Path) -> Path:
    stat = model_tar.stat()
    key = f"{model_tar.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_root / digest


def extract_model_bundle(model_tar: Path, cache_root: Path) -> Path:
    if not model_tar.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_tar}")

    extract_dir = cache_dir_for_tar(model_tar, cache_root)
    extract_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = extract_dir / ".bundle_manifest.json"

    stat = model_tar.stat()
    expected_manifest = {
        "source": str(model_tar.resolve()),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = None
        if existing_manifest == expected_manifest and any(extract_dir.glob("*.pth")):
            return extract_dir

    with tarfile.open(model_tar, "r:gz") as archive:
        archive.extractall(extract_dir)
    manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    return extract_dir


def select_checkpoint_from_dir(extract_dir: Path) -> Path:
    checkpoints = sorted(extract_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found under {extract_dir}")

    finalists = [path for path in checkpoints if "_epoch_" not in path.name and not path.name.startswith("best_")]
    if finalists:
        return finalists[0]

    bests = [path for path in checkpoints if path.name.startswith("best_")]
    if bests:
        return bests[0]

    return checkpoints[0]


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


def infer_backbone_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    layer3_blocks = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("backbone.layer3.") and key.split(".")[2].isdigit()
    }
    if len(layer3_blocks) >= 20:
        return "resnet101"
    return "resnet50"


def infer_num_classes_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("classifier.4.weight", "classifier.4.bias"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise KeyError("Unable to infer num_classes from checkpoint state_dict.")


def load_class_index_metadata(category_map: Path | None, num_classes: int) -> tuple[dict[int, str], set[int]]:
    if category_map is not None and category_map.exists():
        raw = json.loads(category_map.read_text(encoding="utf-8"))
        grouped: dict[int, list[str]] = {}
        for name, index in raw.items():
            grouped.setdefault(int(index), []).append(str(name))

        idx_to_label: dict[int, str] = {}
        lane_indices: set[int] = set()
        for class_index in range(num_classes):
            names = grouped.get(class_index, [])
            if not names:
                idx_to_label[class_index] = f"class_{class_index}"
                continue
            if class_index == 0:
                idx_to_label[class_index] = "background"
            elif all(name.startswith("lane/") for name in names):
                idx_to_label[class_index] = "lane_border"
                lane_indices.add(class_index)
            elif all(name.startswith("area/") for name in names):
                idx_to_label[class_index] = "road_area"
            else:
                idx_to_label[class_index] = " / ".join(sorted(names))
                if any(name.startswith("lane/") for name in names):
                    lane_indices.add(class_index)
        if not lane_indices:
            lane_indices = set(range(1, num_classes))
        return idx_to_label, lane_indices

    idx_to_label = {0: "background"}
    for class_index in range(1, num_classes):
        idx_to_label[class_index] = "lane_border"
    return idx_to_label, set(range(1, num_classes))


def build_inference_model_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    backbone: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    config = TrainConfig(num_classes=num_classes, backbone=backbone, pretrained_backbone=False)
    model = build_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_inference_models(
    state_dict: dict[str, torch.Tensor],
    backbone: str,
    num_classes: int,
    devices: Sequence[torch.device],
) -> list[torch.nn.Module]:
    return [
        build_inference_model_from_state_dict(state_dict, backbone=backbone, num_classes=num_classes, device=device)
        for device in devices
    ]


def preprocess_frame(frame_bgr: np.ndarray, image_height: int, image_width: int) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    image_array = resized.astype(np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array).unsqueeze(0)


def read_frame_chunk(cap: cv2.VideoCapture, chunk_size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while len(frames) < chunk_size:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(frame_bgr)
    return frames


def predict_chunk(
    frames_bgr: Sequence[np.ndarray],
    models: Sequence[torch.nn.Module],
    devices: Sequence[torch.device],
    image_height: int,
    image_width: int,
    batch_size: int,
) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    cursor = 0
    per_device_batch = max(int(batch_size), 1)

    for model, device in zip(models, devices):
        frame_batch = list(frames_bgr[cursor: cursor + per_device_batch])
        cursor += per_device_batch
        if not frame_batch:
            continue

        batch_tensor = torch.cat(
            [preprocess_frame(frame, image_height=image_height, image_width=image_width) for frame in frame_batch],
            dim=0,
        ).to(device)

        logits = extract_segmentation_logits(model(batch_tensor))
        batch_prediction = torch.argmax(logits, dim=1)

        for prediction, frame_bgr in zip(batch_prediction, frame_batch):
            mask = prediction.detach().to("cpu").numpy().astype(np.uint8)
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            predictions.append(mask)

    return predictions


def overlay_lane_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    lane_indices: set[int],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.isin(mask, list(lane_indices))
    color_mask = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_mask[active] = LANE_COLOR

    overlay = frame_bgr.astype(np.float32).copy()
    overlay[active] = ((1.0 - alpha) * overlay[active]) + (alpha * color_mask[active])
    annotated = np.clip(overlay, 0, 255).astype(np.uint8)
    return annotated, color_mask


def build_hud(
    frame_bgr: np.ndarray,
    fps: float,
    device_label: str,
    lane_ratio: float,
    checkpoint_name: str,
) -> np.ndarray:
    hud = frame_bgr.copy()
    cv2.putText(hud, f"FPS: {fps:.1f} | Device: {device_label}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, f"Lane coverage: {lane_ratio * 100:.2f}%", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LANE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(hud, checkpoint_name, (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return hud


def resize_output(frame_bgr: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    if output_width <= 0 or output_height <= 0:
        return frame_bgr
    return cv2.resize(frame_bgr, (output_width, output_height), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    args = make_parser().parse_args()

    if args.weights is not None:
        checkpoint_path = args.weights
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        extract_dir = extract_model_bundle(args.model_tar, args.cache_root)
        checkpoint_path = select_checkpoint_from_dir(extract_dir)
        print(f"Extracted model bundle to: {extract_dir}")

    if not args.video.exists():
        raise FileNotFoundError(f"Input video not found: {args.video}")

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    backbone = infer_backbone_from_state_dict(state_dict) if args.backbone == "auto" else args.backbone
    num_classes = infer_num_classes_from_state_dict(state_dict)
    idx_to_label, lane_indices = load_class_index_metadata(args.category_map, num_classes=num_classes)
    devices = resolve_inference_devices(args.device, args.gpu_ids)
    device_label = format_device_label(devices)
    models = build_inference_models(
        state_dict=state_dict,
        backbone=backbone,
        num_classes=num_classes,
        devices=devices,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_output = args.mask_output
    if args.save_mask_video and mask_output is None:
        mask_output = args.output.with_name(f"{args.output.stem}_mask{args.output.suffix}")

    output_fps = cap.get(cv2.CAP_PROP_FPS)
    if output_fps <= 0:
        output_fps = 30.0

    checkpoint_name = checkpoint_path.name
    frames_per_chunk = max(int(args.batch_size), 1) * len(devices)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    last_chunk_time = time.perf_counter()
    writer = None
    mask_writer = None

    print("Starting DeepLabV3 lane-border video render.")
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Backbone: {backbone}")
    print(f"Classes: {num_classes} ({', '.join(idx_to_label[index] for index in sorted(idx_to_label))})")
    print(f"Inference devices: {device_label}")
    print(f"Frames per chunk: {frames_per_chunk}")
    for line in build_resource_lines(devices[0]):
        print(line)

    with torch.no_grad():
        while True:
            frames_bgr = read_frame_chunk(cap, frames_per_chunk)
            if not frames_bgr:
                break

            predictions = predict_chunk(
                frames_bgr=frames_bgr,
                models=models,
                devices=devices,
                image_height=args.image_height,
                image_width=args.image_width,
                batch_size=args.batch_size,
            )

            current_time = time.perf_counter()
            chunk_elapsed = max(current_time - last_chunk_time, 1e-6)
            fps = len(frames_bgr) / chunk_elapsed
            last_chunk_time = current_time

            for frame_bgr, prediction in zip(frames_bgr, predictions):
                annotated, color_mask = overlay_lane_mask(frame_bgr, prediction, lane_indices=lane_indices, alpha=args.alpha)
                lane_ratio = float(np.isin(prediction, list(lane_indices)).sum()) / float(prediction.size)
                annotated = build_hud(
                    annotated,
                    fps=fps,
                    device_label=device_label,
                    lane_ratio=lane_ratio,
                    checkpoint_name=checkpoint_name,
                )

                annotated_out = resize_output(annotated, args.output_width, args.output_height)
                mask_out = resize_output(color_mask, args.output_width, args.output_height)

                if writer is None:
                    height, width = annotated_out.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to open output writer for {args.output}")
                    print(f"Recording annotated video to: {args.output}")

                if args.save_mask_video and mask_writer is None and mask_output is not None:
                    height, width = mask_out.shape[:2]
                    mask_writer = cv2.VideoWriter(
                        str(mask_output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not mask_writer.isOpened():
                        raise RuntimeError(f"Unable to open mask writer for {mask_output}")
                    print(f"Recording mask-only video to: {mask_output}")

                writer.write(annotated_out)
                if mask_writer is not None:
                    mask_writer.write(mask_out)

                if frame_idx % max(int(args.stats_every), 1) == 0:
                    progress = ((frame_idx + 1) / total_frames * 100.0) if total_frames > 0 else 0.0
                    print(
                        f"[frame {frame_idx:06d}] progress={progress:.1f}% fps={fps:.1f} "
                        f"lane_coverage={lane_ratio * 100:.2f}%"
                    )
                    for line in build_resource_lines(devices[0]):
                        print(f"  {line}")
                frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to: {args.output}")
    if mask_writer is not None:
        mask_writer.release()
        print(f"Saved mask-only video to: {mask_output}")


if __name__ == "__main__":
    main()
