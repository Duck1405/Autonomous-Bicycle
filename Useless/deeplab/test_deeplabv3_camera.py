from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from train import IMAGENET_MEAN, IMAGENET_STD, TrainConfig, build_model, get_device


DRIVABLE_CATEGORY = "area/drivable"
ALTERNATIVE_CATEGORY = "area/alternative"


def parse_source(source: str) -> int | str:
    source = str(source).strip()
    if source.isdigit():
        return int(source)
    return source


def load_class_map(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(key): int(value) for key, value in raw.items()}


def preprocess_frame(frame: np.ndarray, image_height: int, image_width: int) -> tuple[torch.Tensor, np.ndarray]:
    resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(rgb).float().unsqueeze(0)
    return tensor, resized


def build_overlay(
    frame: np.ndarray,
    prediction: np.ndarray,
    drivable_idx: int,
    alternative_idx: int | None,
    alpha: float,
) -> tuple[np.ndarray, float]:
    overlay = frame.copy()

    drivable_mask = prediction == drivable_idx
    alternative_mask = prediction == alternative_idx if alternative_idx is not None else None

    road_pixels = int(drivable_mask.sum())
    road_ratio = road_pixels / float(prediction.size)

    if alternative_mask is not None and np.any(alternative_mask):
        overlay[alternative_mask] = (
            overlay[alternative_mask].astype(np.float32) * (1.0 - alpha)
            + np.array([0, 165, 255], dtype=np.float32) * alpha
        ).astype(np.uint8)

    if np.any(drivable_mask):
        overlay[drivable_mask] = (
            overlay[drivable_mask].astype(np.float32) * (1.0 - alpha)
            + np.array([0, 255, 0], dtype=np.float32) * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(drivable_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                cv2.drawContours(overlay, [largest], -1, (255, 255, 255), 2)

    return overlay, road_ratio


def add_hud(
    frame: np.ndarray,
    fps: float,
    road_ratio: float,
    device_name: str,
    drivable_found: bool,
) -> None:
    state_text = "Road detected" if drivable_found else "Searching for road"
    state_color = (80, 220, 80) if drivable_found else (180, 180, 180)

    panel_left, panel_top = 18, 18
    panel_right, panel_bottom = 380, 140
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), state_color, 2)

    cv2.putText(frame, "DeepLabV3 Road Test", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, state_text, (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.70, state_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Drivable coverage: {road_ratio * 100.0:.1f}%",
        (34, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Device: {device_name.upper()}",
        (34, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(frame, "Green: drivable  Orange: alternative", (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained DeepLabV3 drivable-area model on a camera feed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/Users/amannindra/Projects/Autonomous-Bicycle/best_deeplabv3_resnet50_drivable_small.pth",
        help="Path to the trained model state_dict.",
    )
    parser.add_argument(
        "--category-map",
        type=str,
        default="/Users/amannindra/Projects/Autonomous-Bicycle/configs/drivable_only_category_map.json",
        help="Path to the class-to-index JSON used during training.",
    )
    parser.add_argument("--source", type=str, default="0", help="Camera index or video device path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"], help="Preferred device.")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"], help="Backbone used for training.")
    parser.add_argument("--output-stride", type=int, default=16, choices=[8, 16], help="Model output stride.")
    parser.add_argument("--image-height", type=int, default=513, help="Inference image height.")
    parser.add_argument("--image-width", type=int, default=513, help="Inference image width.")
    parser.add_argument("--view-width", type=int, default=1280, help="Display/capture width.")
    parser.add_argument("--view-height", type=int, default=720, help="Display/capture height.")
    parser.add_argument("--alpha", type=float, default=0.40, help="Overlay strength.")
    parser.add_argument("--save-video", action="store_true", help="Save the annotated camera feed.")
    parser.add_argument("--output-path", type=str, default="", help="Optional explicit output video path.")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2.imshow for headless runs.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on processed frames; 0 means unlimited.")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    class_to_idx = load_class_map(args.category_map)
    if DRIVABLE_CATEGORY not in class_to_idx:
        raise RuntimeError(f"'{DRIVABLE_CATEGORY}' was not found in {args.category_map}")

    drivable_idx = class_to_idx[DRIVABLE_CATEGORY]
    alternative_idx = class_to_idx.get(ALTERNATIVE_CATEGORY)

    device = get_device(args.device)
    config = TrainConfig(
        num_classes=len(class_to_idx),
        backbone=args.backbone,
        output_stride=args.output_stride,
        crop_size=args.image_height,
        device=str(device),
    )
    model = build_model(config)

    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.view_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.view_height)

    video_writer = None
    if args.save_video:
        output_path = Path(args.output_path) if args.output_path else Path("runs/deeplabv3_camera.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (args.view_width, args.view_height),
        )

    if device.type != "cpu":
        warmup = torch.zeros(1, 3, args.image_height, args.image_width, device=device)
        model(warmup)

    window_name = "DeepLabV3 Road Detection"
    print("Starting DeepLabV3 camera inference. Press 'q' to quit.")
    frame_count = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed, stopping.")
                break

            frame = cv2.resize(frame, (args.view_width, args.view_height), interpolation=cv2.INTER_LINEAR)
            inputs, _ = preprocess_frame(frame, args.image_height, args.image_width)
            inputs = inputs.to(device)

            t0 = cv2.getTickCount()
            logits = model(inputs)
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            t1 = cv2.getTickCount()

            prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay, road_ratio = build_overlay(frame, prediction, drivable_idx, alternative_idx, args.alpha)

            elapsed = (t1 - t0) / cv2.getTickFrequency()
            fps = 1.0 / max(elapsed, 1e-6)
            add_hud(
                overlay,
                fps=fps,
                road_ratio=road_ratio,
                device_name=device.type,
                drivable_found=road_ratio > 0.01,
            )

            if not args.no_display:
                cv2.imshow(window_name, overlay)
            if video_writer is not None:
                video_writer.write(overlay)

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

            if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Saved annotated video to: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
