from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from road_guidance import RoadGuidanceEstimator, draw_guidance_overlay
from road_segmentation_ort import OnnxRoadSegmentationSession
from video_common import (
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO,
    annotate_frame,
    build_guidance_config_from_args,
    parse_bool,
    preprocess_frame,
    segmentation_to_overlay,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run road-only HybridNets ONNX inference, preferring TensorRT on Jetson")
    parser.add_argument("--onnx", required=True, help="Path to the segmentation-only ONNX model")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_trt.mp4")), help="Output video path")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--output-width", type=int, default=640, help="Output video width")
    parser.add_argument("--output-height", type=int, default=384, help="Output video height")
    parser.add_argument("--provider", choices=("auto", "tensorrt", "cuda", "cpu"), default="auto")
    parser.add_argument("--trt-fp16", type=parse_bool, default=True, help="Enable FP16 kernels when TensorRT is available")
    parser.add_argument("--trt-workspace-mib", type=int, default=1024, help="TensorRT builder workspace size in MiB")
    parser.add_argument("--engine-cache-dir", type=str, default=None, help="TensorRT engine cache directory")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup inferences used to build/cache the TensorRT engine")
    parser.add_argument("--intra-op-threads", type=int, default=0, help="Optional onnxruntime intra-op thread count")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--ui-scale", type=float, default=1.0, help="Scale factor for the on-video HUD and angle gauge")
    parser.add_argument("--smooth-alpha", type=float, default=0.18, help="EMA smoothing factor for path-heading signals")
    parser.add_argument("--steering-smooth-alpha", type=float, default=0.22, help="EMA smoothing factor for the final Stanley steering command")
    parser.add_argument("--lookahead-ratio", type=float, default=0.62, help="Row ratio used for the center-path lookahead point")
    parser.add_argument("--roi-top-ratio", type=float, default=0.52, help="Top row ratio of the guidance ROI")
    parser.add_argument("--sample-step", type=int, default=6, help="Vertical sampling stride for road-center extraction")
    parser.add_argument("--stanley-gain", type=float, default=1.2, help="Cross-track gain used by the Stanley steering controller")
    parser.add_argument("--stanley-softening", type=float, default=1.0, help="Softening term added to the Stanley speed denominator")
    parser.add_argument("--vehicle-speed-mps", type=float, default=3.0, help="Nominal vehicle speed used by Stanley when live speed is unavailable")
    parser.add_argument("--max-angle", type=float, default=45.0, help="Maximum steering angle shown in the guidance gauge")
    parser.add_argument("--write-mask-video", type=parse_bool, default=False)
    parser.add_argument("--mask-output", type=str, default=None, help="Optional path for a color-mask-only video")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    mask_output_path = Path(args.mask_output).expanduser().resolve() if args.mask_output else output_path.with_name(output_path.stem + "_mask.mp4")
    engine_cache_dir = Path(args.engine_cache_dir).expanduser().resolve() if args.engine_cache_dir else onnx_path.with_name(onnx_path.stem + "_trt_cache")

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    session = OnnxRoadSegmentationSession(
        model_path=onnx_path,
        provider=args.provider,
        engine_cache_dir=engine_cache_dir,
        enable_trt_fp16=args.trt_fp16,
        trt_workspace_size_bytes=args.trt_workspace_mib * 1024 * 1024,
        intra_op_threads=args.intra_op_threads,
    )

    model_hw = session.get_static_input_hw()
    if model_hw is not None:
        model_height, model_width = model_hw
        if (args.image_height, args.image_width) != (model_height, model_width):
            raise RuntimeError(
                f"CLI input size {(args.image_height, args.image_width)} does not match the ONNX model input size "
                f"{(model_height, model_width)}."
            )

    print(f"Loading ONNX model: {onnx_path}")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"ONNX Runtime: {session.describe()}")
    print(f"TensorRT cache dir: {engine_cache_dir}")
    if args.provider in {"auto", "tensorrt"}:
        print(f"TensorRT FP16 enabled: {args.trt_fp16}")

    input_shape = (
        int(session.input_shape[0]) if isinstance(session.input_shape[0], int) else 1,
        int(session.input_shape[1]) if isinstance(session.input_shape[1], int) else 3,
        args.image_height,
        args.image_width,
    )
    session.warmup(input_shape=input_shape, runs=args.warmup_runs)

    guidance_estimator = RoadGuidanceEstimator(build_guidance_config_from_args(args))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.output_width, args.output_height))
    mask_writer = None
    if args.write_mask_video:
        mask_writer = cv2.VideoWriter(str(mask_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.output_width, args.output_height))

    frame_index = 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.perf_counter()

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        input_array, resized_output = preprocess_frame(
            frame_bgr,
            args.image_width,
            args.image_height,
            args.output_width,
            args.output_height,
        )
        segmentation_logits = session.infer(np.expand_dims(input_array, axis=0))
        segmentation_mask = segmentation_logits.argmax(axis=1).astype(np.uint8)[0]
        _, color_mask = segmentation_to_overlay(segmentation_mask, resized_output.shape[:2])
        guidance = guidance_estimator.update(segmentation_mask)
        annotated = annotate_frame(resized_output, color_mask, args.alpha, args.ui_scale)
        annotated = draw_guidance_overlay(annotated, guidance, args.max_angle, args.ui_scale)
        writer.write(annotated)
        if mask_writer is not None:
            mask_writer.write(color_mask)

        frame_index += 1
        if frame_index % 30 == 0 or frame_index == total_frames:
            elapsed = max(time.perf_counter() - start_time, 1e-8)
            print(
                f"Processed {frame_index}/{total_frames if total_frames > 0 else '?'} frames "
                f"({frame_index / elapsed:.2f} fps)"
            )

    elapsed = max(time.perf_counter() - start_time, 1e-8)
    print(f"Finished {frame_index} frames in {elapsed:.2f}s ({frame_index / elapsed:.2f} fps)")
    print(f"Saved annotated video to {output_path}")
    if mask_writer is not None:
        print(f"Saved mask-only video to {mask_output_path}")

    capture.release()
    writer.release()
    if mask_writer is not None:
        mask_writer.release()


if __name__ == "__main__":
    main()
