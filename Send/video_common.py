from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from road_guidance import RoadGuidanceConfig
from runtime_utils import letterbox


ROOT = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = ROOT / "hybridnets-d3_19_43740_best.pth"
DEFAULT_VIDEO = "/home/rayan/aman/Autonomous-Bicycle/HybridNets/demo/video/new_video(2min).mp4"
DEFAULT_OUTPUT = ROOT / "output_stanley.mp4"
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
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


def preprocess_frame(
    frame_bgr: np.ndarray,
    image_width: int,
    image_height: int,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    resized_output = cv2.resize(frame_bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_output, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_width, image_height), interpolation=cv2.INTER_AREA)
    (padded, _), _, _ = letterbox((resized, None), new_shape=(image_height, image_width), auto=False, scaleup=True)
    padded = padded.astype(np.float32) / 255.0
    padded = (padded - MEAN) / STD
    return np.ascontiguousarray(padded.transpose(2, 0, 1)), resized_output


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


def build_guidance_config_from_args(args) -> RoadGuidanceConfig:
    return RoadGuidanceConfig(
        roi_top_ratio=args.roi_top_ratio,
        sample_step=args.sample_step,
        lookahead_ratio=args.lookahead_ratio,
        smoothing_alpha=args.smooth_alpha,
        steering_smoothing_alpha=args.steering_smooth_alpha,
        stanley_gain=args.stanley_gain,
        stanley_softening=args.stanley_softening,
        stanley_speed_mps=args.vehicle_speed_mps,
    )
