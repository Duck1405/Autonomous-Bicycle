from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

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
    get_device,
)


DEFAULT_CHECKPOINT = (
    Path("downloaded_models")
    / "test 1"
    / "autonomous-bike_smoke_test_epoch_001.pth"
)
DEFAULT_CATEGORY_MAP = Path("100k") / "category_map.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DeepLab road-segmentation smoke-test checkpoint on a webcam feed.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_CHECKPOINT, help="Path to the .pth checkpoint.")
    parser.add_argument("--category-map", type=Path, default=DEFAULT_CATEGORY_MAP, help="Path to category_map.json.")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video device path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Inference device.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"], help="Backbone used by the checkpoint.")
    parser.add_argument("--image-height", type=int, default=256, help="Model input height used during training.")
    parser.add_argument("--image-width", type=int, default=512, help="Model input width used during training.")
    parser.add_argument("--view-width", type=int, default=1280, help="Capture/display width.")
    parser.add_argument("--view-height", type=int, default=720, help="Capture/display height.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for predicted regions.")
    parser.add_argument(
        "--road-classes",
        nargs="+",
        default=["area/drivable", "area/alternative"],
        help="Class names counted as road coverage in the HUD.",
    )
    parser.add_argument("--stats-every", type=int, default=30, help="Print resource stats every N frames.")
    parser.add_argument("--project", type=Path, default=Path("runs") / "webcam_segmentation", help="Recording output directory.")
    parser.add_argument("--name", type=str, default="exp", help="Recording run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing recording directory.")
    parser.add_argument(
        "--record",
        "--save-video",
        dest="save_video",
        action="store_true",
        help="Record the annotated webcam feed.",
    )
    parser.add_argument("--record-path", type=Path, default=None, help="Optional explicit output video path.")
    parser.add_argument("--record-fps", type=float, default=0.0, help="Output FPS. Defaults to camera FPS, then 30.")
    parser.add_argument("--record-codec", type=str, default="mp4v", help="FourCC codec for recorded video.")
    return parser


def parse_source(source: str) -> int | str:
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def increment_path(path: Path, exist_ok: bool = False) -> Path:
    if exist_ok or not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.parent / f"{path.name}{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def load_category_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        class_to_idx = {str(name): int(index) for name, index in json.load(handle).items()}
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


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
    config = TrainConfig(num_classes=num_classes, backbone=backbone, pretrained_backbone=False)
    model = build_model(config)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path))
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame_bgr: np.ndarray, image_height: int, image_width: int) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    image_array = resized.astype(np.float32) / 255.0
    image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array).unsqueeze(0)


def class_color(class_name: str, class_index: int) -> tuple[int, int, int]:
    named_colors = {
        "background": (0, 0, 0),
        "area/alternative": (0, 191, 255),
        "area/drivable": (0, 200, 0),
        "area/unknown": (0, 0, 255),
        "lane/road curb": (255, 255, 0),
        "lane/single white": (255, 255, 255),
        "lane/single yellow": (0, 215, 255),
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


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, idx_to_class: dict[int, str], alpha: float) -> np.ndarray:
    color_mask = colorize_mask(mask, idx_to_class)
    overlay = frame_bgr.astype(np.float32).copy()
    active = mask > 0
    overlay[active] = ((1.0 - alpha) * overlay[active]) + (alpha * color_mask[active])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def summarize_mask(mask: np.ndarray, idx_to_class: dict[int, str]) -> list[tuple[str, float]]:
    total_pixels = float(mask.size)
    summary: list[tuple[str, float]] = []
    unique_values, counts = np.unique(mask, return_counts=True)
    count_lookup = {int(value): int(count) for value, count in zip(unique_values, counts)}
    for class_index, class_name in idx_to_class.items():
        if class_index == 0:
            continue
        ratio = count_lookup.get(class_index, 0) / total_pixels if total_pixels > 0 else 0.0
        if ratio > 0:
            summary.append((class_name, ratio))
    summary.sort(key=lambda item: item[1], reverse=True)
    return summary


def build_hud(
    frame: np.ndarray,
    fps: float,
    device: torch.device,
    drivable_ratio: float,
    top_classes: list[tuple[str, float]],
) -> np.ndarray:
    hud = frame.copy()
    cv2.putText(
        hud,
        f"FPS: {fps:.1f} | Device: {device.type.upper()}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        hud,
        f"Road coverage: {drivable_ratio * 100:.1f}%",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if drivable_ratio > 0.05 else (0, 140, 255),
        2,
        cv2.LINE_AA,
    )
    for index, (class_name, ratio) in enumerate(top_classes[:3], start=1):
        cv2.putText(
            hud,
            f"{index}. {class_name}: {ratio * 100:.1f}%",
            (20, 68 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return hud


def resolve_recording_params(
    record_path: Path | None,
    project: Path,
    name: str,
    exist_ok: bool,
    cap: cv2.VideoCapture,
    record_fps: float,
    record_codec: str,
) -> tuple[Path, float, str]:
    if record_path is None:
        save_dir = increment_path(project / name, exist_ok=exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "webcam_segmentation.mp4"
    else:
        output_path = record_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = record_fps if record_fps > 0 else cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    codec = record_codec.strip() or "mp4v"
    if len(codec) != 4:
        raise ValueError(f"--record-codec must be exactly 4 characters, got: {codec!r}")

    return output_path, fps, codec


def main() -> None:
    args = make_parser().parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if not args.category_map.exists():
        raise FileNotFoundError(f"Category map not found: {args.category_map}")

    class_to_idx, idx_to_class = load_category_map(args.category_map)
    missing_road_classes = [name for name in args.road_classes if name not in class_to_idx]
    if missing_road_classes:
        raise ValueError(
            "Unknown road classes: "
            + ", ".join(missing_road_classes)
            + f". Available classes include: {', '.join(sorted(class_to_idx))}"
        )

    device = get_device(args.device)
    model = build_inference_model(
        checkpoint_path=args.weights,
        backbone=args.backbone,
        num_classes=len(class_to_idx),
        device=device,
    )

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.view_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.view_height)

    video_writer = None
    output_path = None
    if args.save_video:
        output_path, output_fps, output_codec = resolve_recording_params(
            record_path=args.record_path,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            cap=cap,
            record_fps=args.record_fps,
            record_codec=args.record_codec,
        )

    print("Starting segmentation webcam demo. Press 'q' to quit.")
    print(f"Checkpoint: {args.weights}")
    print(f"Category map: {args.category_map}")
    print(f"Device: {device}")
    print(f"Input size: {args.image_height}x{args.image_width}")
    print(f"Road classes: {', '.join(args.road_classes)}")
    for line in build_resource_lines(device):
        print(line)

    road_indices = {class_to_idx[name] for name in args.road_classes}
    window_name = "DeepLab Road Segmentation"
    frame_idx = 0
    last_frame_time = time.perf_counter()

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Camera frame read failed, stopping.")
                break

            input_tensor = preprocess_frame(
                frame_bgr,
                image_height=args.image_height,
                image_width=args.image_width,
            ).to(device)

            logits = extract_segmentation_logits(model(input_tensor))
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            prediction = cv2.resize(
                prediction,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            overlay = overlay_mask(frame_bgr, prediction, idx_to_class=idx_to_class, alpha=args.alpha)

            top_classes = summarize_mask(prediction, idx_to_class=idx_to_class)
            drivable_pixels = sum(int((prediction == class_index).sum()) for class_index in road_indices)
            drivable_ratio = drivable_pixels / float(prediction.size)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - last_frame_time, 1e-6)
            last_frame_time = current_time

            annotated = build_hud(
                overlay,
                fps=fps,
                device=device,
                drivable_ratio=drivable_ratio,
                top_classes=top_classes,
            )

            if video_writer is None and args.save_video:
                output_height, output_width = annotated.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*output_codec),
                    output_fps,
                    (output_width, output_height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Unable to open video writer for: {output_path}")
                print(f"Recording webcam video to: {output_path}")

            cv2.imshow(window_name, annotated)
            if video_writer is not None:
                video_writer.write(annotated)

            if frame_idx % max(args.stats_every, 1) == 0:
                top_summary = ", ".join(f"{name}={ratio * 100:.1f}%" for name, ratio in top_classes[:5]) or "none"
                print(
                    f"[frame {frame_idx:06d}] fps={fps:.1f} road_coverage={drivable_ratio * 100:.1f}% "
                    f"top_classes={top_summary}"
                )
                for line in build_resource_lines(device):
                    print(f"  {line}")

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Saved webcam video to: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
