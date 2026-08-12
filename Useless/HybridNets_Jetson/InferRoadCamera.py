from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch

from InferRoadVideo import (
    DEFAULT_WEIGHTS,
    annotate_frame,
    build_guidance_config_from_args,
    build_parser as build_video_parser,
    load_model,
    parse_bool,
    parse_gpu_ids,
    pick_device,
    preprocess_frame,
    run_segmentation_batch,
    segmentation_to_overlay,
)
from road_guidance import RoadGuidanceEstimator, draw_guidance_overlay, format_direction


ROOT = Path(__file__).resolve().parent


def parse_source(source_value: str):
    source_value = str(source_value).strip()
    if source_value.isdigit():
        return int(source_value)
    return source_value


def build_parser() -> argparse.ArgumentParser:
    video_parser = build_video_parser()
    parser = argparse.ArgumentParser("Run the Jetson runtime HybridNets guidance model on a live camera or stream")
    for action in video_parser._actions:
        if action.dest in {"video", "output", "write_mask_video", "mask_output"}:
            continue
        if action.dest == "help":
            continue
        parser._add_action(action)

    parser.set_defaults(weights=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or video path")
    parser.add_argument("--display", type=parse_bool, default=True, help="Show the live annotated stream window")
    parser.add_argument("--save-video", type=parse_bool, default=True, help="Save the annotated stream to a video file")
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(ROOT / "road_guidance_camera_output.mp4"),
        help="Saved video path for camera or stream inference",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested capture width for live cameras")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested capture height for live cameras")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested capture FPS for live cameras")
    parser.add_argument("--status-json", type=str, default="road_guidance_latest.json", help="Latest per-frame guidance status JSON")
    parser.add_argument("--json-indent", type=int, default=2, help="Indentation level for the latest-status JSON")
    return parser


def open_capture(source, camera_width: int, camera_height: int, camera_fps: float) -> cv2.VideoCapture:
    backend = cv2.CAP_ANY
    if isinstance(source, int):
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend = cv2.CAP_AVFOUNDATION
        elif platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
            backend = cv2.CAP_V4L2

    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture = cv2.VideoCapture(source)

    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        capture.set(cv2.CAP_PROP_FPS, camera_fps)
    return capture


def write_json_atomic(path: Path, payload: dict, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=indent, sort_keys=False), encoding="utf-8")
    tmp_path.replace(path)


def build_status_payload(
    *,
    frame_index: int,
    source,
    weights_path: Path,
    device: torch.device,
    processing_fps: float,
    frame_shape: tuple[int, int],
    road_pixels: int,
    lane_pixels: int,
    guidance,
) -> dict:
    direction_label, action_label, _ = format_direction(None if guidance is None else guidance.steering_angle_deg)
    height, width = frame_shape
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frame_index": frame_index,
        "source": str(source),
        "weights": str(weights_path),
        "device": str(device),
        "processing_fps": processing_fps,
        "frame_width": width,
        "frame_height": height,
        "road_pixel_count": road_pixels,
        "lane_pixel_count": lane_pixels,
        "direction": direction_label,
        "action": action_label,
    }

    if guidance is None:
        payload.update(
            {
                "heading_error_deg": None,
                "raw_heading_error_deg": None,
                "path_heading_deg": None,
                "raw_path_heading_deg": None,
                "steering_angle_deg": None,
                "raw_steering_angle_deg": None,
                "stanley_term_deg": None,
                "confidence": 0.0,
                "method": "none",
                "controller": "none",
                "lookahead_point": None,
                "image_center_x": width // 2,
                "y_near": None,
                "y_lookahead": None,
                "x_near": None,
                "x_lookahead": None,
                "cross_track_error_px": None,
                "cross_track_error_norm": None,
                "corridor_width_px": None,
                "lane_support_ratio": 0.0,
                "valid_row_ratio": 0.0,
            }
        )
        return payload

    payload.update(
        {
            "heading_error_deg": guidance.heading_error_deg,
            "raw_heading_error_deg": guidance.raw_heading_error_deg,
            "path_heading_deg": guidance.path_heading_deg,
            "raw_path_heading_deg": guidance.raw_path_heading_deg,
            "steering_angle_deg": guidance.steering_angle_deg,
            "raw_steering_angle_deg": guidance.raw_steering_angle_deg,
            "stanley_term_deg": guidance.stanley_term_deg,
            "confidence": guidance.confidence,
            "method": guidance.method,
            "controller": guidance.controller,
            "lookahead_point": {"x": guidance.lookahead_point[0], "y": guidance.lookahead_point[1]},
            "image_center_x": guidance.image_center_x,
            "y_near": guidance.y_near,
            "y_lookahead": guidance.y_lookahead,
            "x_near": guidance.x_near,
            "x_lookahead": guidance.x_lookahead,
            "cross_track_error_px": guidance.cross_track_error_px,
            "cross_track_error_norm": guidance.cross_track_error_norm,
            "corridor_width_px": guidance.corridor_width_px,
            "lane_support_ratio": guidance.lane_support_ratio,
            "valid_row_ratio": guidance.valid_row_ratio,
            "sample_counts": {
                "left": len(guidance.left_points),
                "right": len(guidance.right_points),
                "center": len(guidance.center_points),
                "fitted": len(guidance.fitted_points),
            },
        }
    )
    return payload


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    output_video_path = Path(args.output_video).expanduser().resolve()
    status_json_path = Path(args.status_json).expanduser().resolve()
    source = parse_source(args.source)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    gpu_ids = parse_gpu_ids(args.gpu_ids) if device.type == "cuda" else []
    if device.type == "cuda":
        available_gpu_count = torch.cuda.device_count()
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpu_count]
        if not gpu_ids:
            gpu_ids = [0]

    print(f"Loading weights: {weights_path}")
    print(f"Source: {source}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU ids: {gpu_ids}")
    print("Batch size: 1")
    print(f"AMP enabled: {use_amp}")
    print(f"Status JSON: {status_json_path}")
    if args.save_video:
        print(f"Output video: {output_video_path}")
    else:
        print("Output video saving: disabled")

    model = load_model(weights_path, args.compound_coef, args.backbone, device, gpu_ids)
    guidance_estimator = RoadGuidanceEstimator(build_guidance_config_from_args(args))

    capture = open_capture(source, args.camera_width, args.camera_height, args.camera_fps)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = args.camera_fps

    writer = None
    if args.save_video:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.output_width, args.output_height))

    frame_index = 0
    start_time = time.perf_counter()
    window_name = "HybridNets Road Guidance"

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            print("Stream read failed or ended, stopping.")
            break

        tensor, resized_output = preprocess_frame(frame_bgr, args.image_width, args.image_height, args.output_width, args.output_height)
        batch_tensor = tensor.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            batch_tensor = batch_tensor.to(memory_format=torch.channels_last)

        segmentation_logits = run_segmentation_batch(model, batch_tensor, use_amp)
        segmentation_mask = segmentation_logits.argmax(dim=1).detach().cpu().numpy().astype("uint8")[0]
        mask_resized, color_mask = segmentation_to_overlay(segmentation_mask, resized_output.shape[:2])
        guidance = guidance_estimator.update(segmentation_mask)

        annotated = annotate_frame(resized_output, color_mask, args.alpha, args.ui_scale)
        annotated = draw_guidance_overlay(annotated, guidance, args.max_angle, args.ui_scale)

        frame_index += 1
        elapsed = max(time.perf_counter() - start_time, 1e-8)
        processing_fps = frame_index / elapsed
        road_pixels = int((mask_resized == 1).sum())
        lane_pixels = int((mask_resized == 2).sum())
        payload = build_status_payload(
            frame_index=frame_index,
            source=source,
            weights_path=weights_path,
            device=device,
            processing_fps=processing_fps,
            frame_shape=annotated.shape[:2],
            road_pixels=road_pixels,
            lane_pixels=lane_pixels,
            guidance=guidance,
        )
        write_json_atomic(status_json_path, payload, args.json_indent)

        if writer is not None:
            writer.write(annotated)
        if args.display:
            cv2.imshow(window_name, annotated)

        if frame_index % 30 == 0:
            print(f"Processed {frame_index} frames ({processing_fps:.2f} fps)")

        key = cv2.waitKey(1) if args.display else -1
        if key & 0xFF in {ord('q'), 27}:
            print("Stopping on user request.")
            break

    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved camera output to {output_video_path}")
    if args.display:
        cv2.destroyAllWindows()
    print(f"Updated latest status JSON at {status_json_path}")


if __name__ == "__main__":
    main()
