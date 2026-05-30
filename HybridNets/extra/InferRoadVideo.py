import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from road_segmentation_model import build_road_segmentation_model
from road_guidance import RoadGuidanceConfig, RoadGuidanceEstimator, draw_guidance_overlay
from utils.constants import MULTICLASS_MODE
from utils.segmentation import segmentation_logits_to_probabilities, segmentation_probabilities_to_predictions, \
    resize_segmentation_probabilities
from utils.utils import letterbox


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_VIDEO = Path("/home/aman/Projects/Auto/test_videos/mp4Videos/new_video2.mp4")
DEFAULT_OUTPUT = Path("/home/aman/Projects/Auto/test_videos/mp4Videos/new_video2_hybridnets_output.mp4")
OVERLAY_COLORS_BGR = {
    "road": np.array([0, 180, 0], dtype=np.uint8),
    "lane": np.array([0, 255, 255], dtype=np.uint8),
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run road-only HybridNets inference on a video")
    parser.add_argument("--weights", required=True, help="Path to trained road-only HybridNets weights (.pth)")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output video path")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--output-width", type=int, default=640, help="Output video width")
    parser.add_argument("--output-height", type=int, default=384, help="Output video height")
    parser.add_argument("--compound-coef", type=int, default=3)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--gpu-ids", type=str, default="0,1", help="Comma-separated CUDA device ids for inference")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per inference batch")
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--ui-scale", type=float, default=1.0, help="Scale factor for the on-video HUD and angle gauge")
    parser.add_argument("--smooth-alpha", type=float, default=0.18, help="EMA smoothing factor for heading guidance")
    parser.add_argument("--lookahead-ratio", type=float, default=0.62, help="Row ratio used for the center-path lookahead point")
    parser.add_argument("--roi-top-ratio", type=float, default=0.52, help="Top row ratio of the guidance ROI")
    parser.add_argument("--sample-step", type=int, default=6, help="Vertical sampling stride for road-center extraction")
    parser.add_argument("--max-angle", type=float, default=45.0, help="Maximum steering angle shown in the guidance gauge")
    parser.add_argument("--write-mask-video", type=parse_bool, default=False)
    parser.add_argument("--mask-output", type=str, default=None, help="Optional path for a color-mask-only video")
    parser.add_argument("--confidence-output-dir", type=str, default=None,
                        help="Optional directory for per-frame per-pixel confidence .npz files")
    parser.add_argument("--confidence-stride", type=int, default=1,
                        help="Save confidence for every Nth frame when --confidence-output-dir is set")
    return parser


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = []
    for item in value.split(","):
        item = item.strip()
        if item:
            gpu_ids.append(int(item))
    return gpu_ids


def load_model(
    weights_path: Path,
    compound_coef: int,
    backbone_name: str | None,
    device: torch.device,
    gpu_ids: list[int],
) -> torch.nn.Module:
    return build_road_segmentation_model(
        weights_path=weights_path,
        compound_coef=compound_coef,
        backbone_name=backbone_name,
        device=device,
        gpu_ids=gpu_ids,
    )


def preprocess_frame(
    frame_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    output_width: int,
    output_height: int,
) -> tuple[torch.Tensor, np.ndarray]:
    resized_output = cv2.resize(frame_bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
    original_height, original_width = resized_output.shape[:2]
    rgb = cv2.cvtColor(resized_output, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_width, image_height), interpolation=cv2.INTER_AREA)
    (padded, _), _, pad = letterbox((resized, None), new_shape=(image_height, image_width), auto=False, scaleup=True)
    padded = padded.astype(np.float32) / 255.0
    padded = (padded - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    return tensor, resized_output


def segmentation_to_overlay(segmentation_mask: np.ndarray, original_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    original_height, original_width = original_shape
    mask_resized = cv2.resize(segmentation_mask.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    color_mask[mask_resized == 1] = OVERLAY_COLORS_BGR["road"]
    color_mask[mask_resized == 2] = OVERLAY_COLORS_BGR["lane"]
    return mask_resized, color_mask


def annotate_frame(frame_bgr: np.ndarray, color_mask: np.ndarray, alpha: float, ui_scale: float) -> np.ndarray:
    output = frame_bgr.copy()
    non_background = np.any(color_mask != 0, axis=2)
    if np.any(non_background):
        blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_mask, alpha, 0.0)
        output[non_background] = blended[non_background]
    legend_x = max(8, int(round(20 * ui_scale)))
    legend_y = max(18, int(round(30 * ui_scale)))
    legend_scale = max(0.30, 0.56 * ui_scale)
    legend_thickness = max(1, int(round(2 * ui_scale)))
    cv2.putText(
        output,
        "road=green lane=yellow centerline=cyan",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        legend_scale,
        (255, 255, 255),
        legend_thickness,
    )
    return output


def run_segmentation_batch(model: torch.nn.Module, batch_tensor: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if batch_tensor.device.type == "cuda" else "cpu", enabled=use_amp):
            segmentation_logits = model(batch_tensor)
    return segmentation_logits


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    mask_output_path = Path(args.mask_output).expanduser().resolve() if args.mask_output else output_path.with_name(output_path.stem + "_mask.mp4")
    confidence_output_dir = Path(args.confidence_output_dir).expanduser().resolve() if args.confidence_output_dir else None

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device = pick_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    gpu_ids = parse_gpu_ids(args.gpu_ids) if device.type == "cuda" else []
    if device.type == "cuda":
        available_gpu_count = torch.cuda.device_count()
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpu_count]
        if not gpu_ids:
            gpu_ids = [0]

    print(f"Loading weights: {weights_path}")
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU ids: {gpu_ids}")
    print(f"Batch size: {args.batch_size}")
    print(f"AMP enabled: {use_amp}")

    model = load_model(weights_path, args.compound_coef, args.backbone, device, gpu_ids)
    guidance_estimator = RoadGuidanceEstimator(
        RoadGuidanceConfig(
            roi_top_ratio=args.roi_top_ratio,
            sample_step=args.sample_step,
            lookahead_ratio=args.lookahead_ratio,
            smoothing_alpha=args.smooth_alpha,
        )
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = args.output_width
    frame_height = args.output_height
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if confidence_output_dir is not None:
        confidence_output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    mask_writer = None
    if args.write_mask_video:
        mask_writer = cv2.VideoWriter(
            str(mask_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

    frame_index = 0
    source_frame_index = 0
    start_time = time.perf_counter()
    pending_tensors: list[torch.Tensor] = []
    pending_frames: list[np.ndarray] = []
    pending_frame_indices: list[int] = []

    def flush_batch():
        nonlocal frame_index, pending_tensors, pending_frames, pending_frame_indices
        if not pending_tensors:
            return
        batch_tensor = torch.stack(pending_tensors, dim=0).to(device, non_blocking=True)
        if device.type == "cuda":
            batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
        segmentation_logits = run_segmentation_batch(model, batch_tensor, use_amp)
        segmentation_probabilities = segmentation_logits_to_probabilities(segmentation_logits.float(), MULTICLASS_MODE)
        segmentation_masks = segmentation_probabilities.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)

        for batch_offset, (source_frame_index, frame_bgr, segmentation_mask) in enumerate(
            zip(pending_frame_indices, pending_frames, segmentation_masks)
        ):
            if confidence_output_dir is not None and source_frame_index % max(args.confidence_stride, 1) == 0:
                frame_probabilities = resize_segmentation_probabilities(
                    segmentation_probabilities[batch_offset:batch_offset + 1],
                    size=frame_bgr.shape[:2],
                )
                frame_probabilities = frame_probabilities / frame_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
                frame_prediction, frame_confidence = segmentation_probabilities_to_predictions(
                    frame_probabilities,
                    MULTICLASS_MODE,
                )
                np.savez_compressed(
                    confidence_output_dir / f'frame_{source_frame_index:06d}_seg_confidence.npz',
                    probabilities=frame_probabilities.squeeze(0).detach().cpu().numpy().astype(np.float32),
                    predicted_class=frame_prediction.squeeze(0).detach().cpu().numpy().astype(np.uint8),
                    confidence=frame_confidence.squeeze(0).detach().cpu().numpy().astype(np.float32),
                    class_names=np.array(['background', 'road', 'lane']),
                    seg_mode=np.array(MULTICLASS_MODE),
                )
            _, color_mask = segmentation_to_overlay(segmentation_mask, frame_bgr.shape[:2])
            guidance = guidance_estimator.update(segmentation_mask)
            annotated = annotate_frame(frame_bgr, color_mask, args.alpha, args.ui_scale)
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

        pending_tensors = []
        pending_frames = []
        pending_frame_indices = []

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        tensor, resized_output = preprocess_frame(
            frame_bgr,
            args.image_width,
            args.image_height,
            args.output_width,
            args.output_height,
        )
        pending_tensors.append(tensor)
        pending_frames.append(resized_output)
        pending_frame_indices.append(source_frame_index)
        source_frame_index += 1
        if len(pending_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

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
