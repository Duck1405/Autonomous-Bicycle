from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np


GUIDANCE_COLORS_BGR = {
    "left": (255, 180, 0),
    "right": (255, 180, 0),
    "center": (255, 255, 0),
    "lookahead": (0, 0, 255),
    "reference": (255, 255, 255),
}


@dataclass
class RoadGuidanceConfig:
    roi_top_ratio: float = 0.52
    roi_bottom_ratio: float = 0.97
    sample_step: int = 6
    min_corridor_width_ratio: float = 0.16
    max_corridor_width_ratio: float = 0.92
    edge_refine_tolerance_ratio: float = 0.06
    max_width_change_ratio: float = 0.22
    max_center_jump_ratio: float = 0.12
    min_center_points: int = 10
    fit_point_count: int = 24
    lookahead_ratio: float = 0.62
    y_near_ratio: float = 0.92
    smoothing_alpha: float = 0.18
    steering_smoothing_alpha: float = 0.22
    hold_confidence_decay: float = 0.82
    min_hold_confidence: float = 0.35
    stanley_gain: float = 1.2
    stanley_softening: float = 1.0
    stanley_speed_mps: float = 3.0
    max_cross_track_norm: float = 1.5


@dataclass
class RoadGuidanceResult:
    heading_error_deg: float
    raw_heading_error_deg: float
    path_heading_deg: float
    raw_path_heading_deg: float
    steering_angle_deg: float
    raw_steering_angle_deg: float
    stanley_term_deg: float
    confidence: float
    method: str
    controller: str
    left_points: list[tuple[int, int]]
    right_points: list[tuple[int, int]]
    center_points: list[tuple[int, int]]
    fitted_points: list[tuple[int, int]]
    lookahead_point: tuple[int, int]
    image_center_x: int
    y_near: int
    y_lookahead: int
    x_near: float
    x_lookahead: float
    cross_track_error_px: float
    cross_track_error_norm: float
    corridor_width_px: float
    lane_support_ratio: float
    valid_row_ratio: float


@dataclass
class RoadGuidanceState:
    previous_result: RoadGuidanceResult | None = None
    previous_heading_deg: float | None = None
    previous_path_heading_deg: float | None = None
    previous_steering_deg: float | None = None
    previous_lookahead_x: float | None = None


def smooth_scalar(current_value: float | None, previous_value: float | None, alpha: float) -> float | None:
    if current_value is None:
        return previous_value
    if previous_value is None:
        return current_value
    return (1.0 - alpha) * previous_value + alpha * current_value


def _connected_runs(xs: np.ndarray) -> list[np.ndarray]:
    if xs.size == 0:
        return []
    split_indices = np.where(np.diff(xs) > 1)[0] + 1
    return [segment for segment in np.split(xs, split_indices) if segment.size > 0]


def _select_primary_corridor(corridor_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corridor_mask, connectivity=8)
    if num_labels <= 2:
        return corridor_mask

    height, width = corridor_mask.shape
    anchor_x = width // 2
    anchor_y = min(height - 1, int(height * 0.92))

    anchor_label = labels[anchor_y, anchor_x]
    if anchor_label > 0:
        return (labels == anchor_label).astype(np.uint8) * 255

    best_label = 0
    best_score = float("-inf")
    image_area = float(height * width)
    for label_index in range(1, num_labels):
        x, y, component_width, component_height, area = stats[label_index]
        centroid_x, _ = centroids[label_index]
        bottom_ratio = float(y + component_height) / max(height, 1)
        center_distance = abs(float(centroid_x) - anchor_x) / max(width, 1)
        area_ratio = float(area) / max(image_area, 1.0)
        score = area_ratio * 2.0 + bottom_ratio - center_distance
        if score > best_score:
            best_score = score
            best_label = label_index

    if best_label <= 0:
        return corridor_mask
    return (labels == best_label).astype(np.uint8) * 255


def _extract_row_bounds(
    corridor_mask: np.ndarray,
    y: int,
    reference_x: float,
    min_width: int,
    max_width: int,
) -> tuple[int, int] | None:
    xs = np.flatnonzero(corridor_mask[y] > 0)
    if xs.size == 0:
        return None

    best_bounds = None
    best_score = None
    for segment in _connected_runs(xs):
        left_x = int(segment[0])
        right_x = int(segment[-1])
        width_now = right_x - left_x
        if width_now < min_width or width_now > max_width:
            continue

        segment_center = 0.5 * (left_x + right_x)
        contains_reference = left_x <= reference_x <= right_x
        score = (
            0 if contains_reference else 1,
            abs(segment_center - reference_x),
            -width_now,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_bounds = (left_x, right_x)

    return best_bounds


def _refine_boundary_with_lane(
    lane_mask: np.ndarray,
    y: int,
    boundary_x: int,
    side: str,
    tolerance: int,
    corridor_left: int,
    corridor_right: int,
) -> int | None:
    xs = np.flatnonzero(lane_mask[y] > 0)
    if xs.size == 0:
        return None

    if side == "left":
        candidates = xs[(xs >= corridor_left) & (xs <= min(boundary_x + tolerance, corridor_right))]
    else:
        candidates = xs[(xs <= corridor_right) & (xs >= max(boundary_x - tolerance, corridor_left))]

    if candidates.size == 0:
        return None

    best_index = int(np.argmin(np.abs(candidates - boundary_x)))
    return int(candidates[best_index])


def _compute_confidence(collected_rows: int, expected_rows: int, widths: list[float], lane_rows: int) -> float:
    if collected_rows <= 0 or not widths:
        return 0.0

    valid_row_ratio = min(collected_rows / max(expected_rows, 1), 1.0)
    lane_support_ratio = lane_rows / max(collected_rows, 1)
    width_mean = float(np.mean(widths))
    width_std = float(np.std(widths))
    width_stability = 1.0 - min(width_std / max(width_mean, 1.0), 1.0)
    confidence = 0.5 * valid_row_ratio + 0.3 * width_stability + 0.2 * lane_support_ratio
    return float(np.clip(confidence, 0.0, 1.0))


def _compute_path_heading(fit: np.poly1d, y_near: int, min_y: int, height: int) -> tuple[float, int, float]:
    tangent_delta_y = max(int(height * 0.06), 12)
    y_heading = max(min_y, y_near - tangent_delta_y)
    if y_heading >= y_near:
        y_heading = max(min_y, y_near - 1)
    x_near = float(fit(y_near))
    x_heading = float(fit(y_heading))
    raw_path_heading_deg = float(np.degrees(np.arctan2(x_heading - x_near, y_near - y_heading)))
    return raw_path_heading_deg, int(y_heading), x_heading


def _compute_stanley_steering(
    *,
    path_heading_deg: float,
    cross_track_error_px: float,
    corridor_width_px: float,
    image_width: int,
    config: RoadGuidanceConfig,
) -> tuple[float, float]:
    corridor_half_width = max(corridor_width_px * 0.5, image_width * 0.12, 1.0)
    cross_track_error_norm = float(
        np.clip(
            cross_track_error_px / corridor_half_width,
            -config.max_cross_track_norm,
            config.max_cross_track_norm,
        )
    )
    effective_speed = max(config.stanley_speed_mps + config.stanley_softening, 1e-3)
    stanley_term_deg = float(np.degrees(np.arctan2(config.stanley_gain * cross_track_error_norm, effective_speed)))
    raw_steering_angle_deg = float(path_heading_deg + stanley_term_deg)
    return raw_steering_angle_deg, stanley_term_deg


def _fit_center_path(
    center_points: list[tuple[int, int]],
    image_shape: tuple[int, int],
    config: RoadGuidanceConfig,
    state: RoadGuidanceState,
    confidence: float,
    corridor_width_px: float,
) -> RoadGuidanceResult | None:
    if len(center_points) < config.min_center_points:
        return None

    points = np.array(center_points, dtype=np.float32)
    xs = points[:, 0]
    ys = points[:, 1]
    weights = np.interp(ys, (float(ys.min()), float(ys.max())), (0.65, 1.4))
    fit_degree = 2 if len(points) >= 12 else 1
    coefficients = np.polyfit(ys, xs, deg=fit_degree, w=weights)
    fit = np.poly1d(coefficients)

    height, width = image_shape
    y_near = min(int(height * config.y_near_ratio), int(ys.max()))
    y_lookahead = max(int(height * config.lookahead_ratio), int(ys.min()))
    if y_near <= y_lookahead:
        return None

    x_near = float(fit(y_near))
    x_lookahead_raw = float(fit(y_lookahead))
    x_lookahead = smooth_scalar(x_lookahead_raw, state.previous_lookahead_x, config.smoothing_alpha)
    if x_lookahead is None:
        x_lookahead = x_lookahead_raw

    image_center_x = width // 2
    raw_heading_error_deg = float(np.degrees(np.arctan2(x_lookahead_raw - image_center_x, y_near - y_lookahead)))
    heading_error_deg = smooth_scalar(raw_heading_error_deg, state.previous_heading_deg, config.smoothing_alpha)
    if heading_error_deg is None:
        heading_error_deg = raw_heading_error_deg

    raw_path_heading_deg, _, _ = _compute_path_heading(
        fit=fit,
        y_near=y_near,
        min_y=int(ys.min()),
        height=height,
    )
    path_heading_deg = smooth_scalar(raw_path_heading_deg, state.previous_path_heading_deg, config.smoothing_alpha)
    if path_heading_deg is None:
        path_heading_deg = raw_path_heading_deg

    cross_track_error_px = float(x_near - image_center_x)
    raw_steering_angle_deg, stanley_term_deg = _compute_stanley_steering(
        path_heading_deg=raw_path_heading_deg,
        cross_track_error_px=cross_track_error_px,
        corridor_width_px=corridor_width_px,
        image_width=width,
        config=config,
    )
    steering_angle_deg = smooth_scalar(raw_steering_angle_deg, state.previous_steering_deg, config.steering_smoothing_alpha)
    if steering_angle_deg is None:
        steering_angle_deg = raw_steering_angle_deg

    corridor_half_width = max(corridor_width_px * 0.5, width * 0.12, 1.0)
    cross_track_error_norm = float(
        np.clip(
            cross_track_error_px / corridor_half_width,
            -config.max_cross_track_norm,
            config.max_cross_track_norm,
        )
    )

    sample_ys = np.linspace(y_near, y_lookahead, num=config.fit_point_count)
    fitted_points = []
    for sample_y in sample_ys:
        sample_x = int(np.clip(fit(sample_y), 0, width - 1))
        fitted_points.append((sample_x, int(sample_y)))

    return RoadGuidanceResult(
        heading_error_deg=float(heading_error_deg),
        raw_heading_error_deg=float(raw_heading_error_deg),
        path_heading_deg=float(path_heading_deg),
        raw_path_heading_deg=float(raw_path_heading_deg),
        steering_angle_deg=float(steering_angle_deg),
        raw_steering_angle_deg=float(raw_steering_angle_deg),
        stanley_term_deg=float(stanley_term_deg),
        confidence=float(confidence),
        method="road",
        controller="stanley",
        left_points=[],
        right_points=[],
        center_points=center_points,
        fitted_points=fitted_points,
        lookahead_point=(int(np.clip(x_lookahead, 0, width - 1)), int(y_lookahead)),
        image_center_x=image_center_x,
        y_near=int(y_near),
        y_lookahead=int(y_lookahead),
        x_near=float(x_near),
        x_lookahead=float(x_lookahead),
        cross_track_error_px=float(cross_track_error_px),
        cross_track_error_norm=float(cross_track_error_norm),
        corridor_width_px=float(corridor_width_px),
        lane_support_ratio=0.0,
        valid_row_ratio=0.0,
    )


def _hold_previous_result(state: RoadGuidanceState, config: RoadGuidanceConfig) -> RoadGuidanceResult | None:
    previous_result = state.previous_result
    if previous_result is None or previous_result.confidence < config.min_hold_confidence:
        return None

    held_result = deepcopy(previous_result)
    held_result.method = "hold"
    held_result.confidence = float(previous_result.confidence * config.hold_confidence_decay)
    held_result.heading_error_deg = float(previous_result.heading_error_deg)
    held_result.raw_heading_error_deg = float(previous_result.raw_heading_error_deg)
    return held_result


class RoadGuidanceEstimator:
    def __init__(self, config: RoadGuidanceConfig | None = None):
        self.config = config or RoadGuidanceConfig()
        self.state = RoadGuidanceState()

    def update(self, segmentation_mask: np.ndarray) -> RoadGuidanceResult | None:
        if segmentation_mask.ndim != 2:
            raise ValueError("Expected a 2D segmentation mask.")

        height, width = segmentation_mask.shape
        lane_mask = (segmentation_mask == 2).astype(np.uint8) * 255
        corridor_mask = np.isin(segmentation_mask, (1, 2)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        corridor_mask = cv2.morphologyEx(corridor_mask, cv2.MORPH_CLOSE, kernel)
        corridor_mask = cv2.medianBlur(corridor_mask, 5)
        corridor_mask = _select_primary_corridor(corridor_mask)
        lane_mask = cv2.bitwise_and(lane_mask, lane_mask, mask=corridor_mask)

        y_bottom = min(height - 1, int(height * self.config.roi_bottom_ratio))
        y_top = max(0, int(height * self.config.roi_top_ratio))
        expected_rows = max((y_bottom - y_top) // max(self.config.sample_step, 1), 1)
        min_width = max(int(width * self.config.min_corridor_width_ratio), 24)
        max_width = max(int(width * self.config.max_corridor_width_ratio), min_width + 1)
        refine_tolerance = max(int(width * self.config.edge_refine_tolerance_ratio), 6)
        max_width_delta = int(width * self.config.max_width_change_ratio)
        max_center_jump = int(width * self.config.max_center_jump_ratio)

        reference_x = float(self.state.previous_lookahead_x if self.state.previous_lookahead_x is not None else width // 2)
        previous_width = None

        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        center_points: list[tuple[int, int]] = []
        widths: list[float] = []
        lane_rows = 0

        for y in range(y_bottom, y_top - 1, -self.config.sample_step):
            bounds = _extract_row_bounds(corridor_mask, y, reference_x, min_width, max_width)
            if bounds is None:
                continue

            corridor_left, corridor_right = bounds
            left_x = corridor_left
            right_x = corridor_right

            left_lane_x = _refine_boundary_with_lane(
                lane_mask,
                y,
                corridor_left,
                "left",
                refine_tolerance,
                corridor_left,
                corridor_right,
            )
            right_lane_x = _refine_boundary_with_lane(
                lane_mask,
                y,
                corridor_right,
                "right",
                refine_tolerance,
                corridor_left,
                corridor_right,
            )
            if left_lane_x is not None:
                left_x = left_lane_x
            if right_lane_x is not None:
                right_x = right_lane_x
            if left_lane_x is not None or right_lane_x is not None:
                lane_rows += 1

            width_now = right_x - left_x
            if width_now < min_width or width_now > max_width:
                continue

            if previous_width is not None and abs(width_now - previous_width) > max_width_delta:
                continue

            center_x = 0.5 * (left_x + right_x)
            if center_points and abs(center_x - center_points[-1][0]) > max_center_jump:
                continue

            left_points.append((int(left_x), y))
            right_points.append((int(right_x), y))
            center_points.append((int(center_x), y))
            widths.append(float(width_now))
            previous_width = width_now
            reference_x = center_x

        confidence = _compute_confidence(len(center_points), expected_rows, widths, lane_rows)
        corridor_width_px = float(np.mean(widths)) if widths else float(width * self.config.min_corridor_width_ratio)
        result = _fit_center_path(center_points, (height, width), self.config, self.state, confidence, corridor_width_px)
        if result is None:
            held_result = _hold_previous_result(self.state, self.config)
            if held_result is not None:
                self.state.previous_result = held_result
                self.state.previous_heading_deg = held_result.heading_error_deg
                self.state.previous_path_heading_deg = held_result.path_heading_deg
                self.state.previous_steering_deg = held_result.steering_angle_deg
                self.state.previous_lookahead_x = held_result.x_lookahead
            return held_result

        result.left_points = left_points
        result.right_points = right_points
        result.lane_support_ratio = lane_rows / max(len(center_points), 1)
        result.valid_row_ratio = len(center_points) / max(expected_rows, 1)
        result.method = "lane+road" if result.lane_support_ratio >= 0.25 else "road"

        self.state.previous_result = result
        self.state.previous_heading_deg = result.heading_error_deg
        self.state.previous_path_heading_deg = result.path_heading_deg
        self.state.previous_steering_deg = result.steering_angle_deg
        self.state.previous_lookahead_x = result.x_lookahead
        return result
    def new_upate(self, segmentation_mask: np.ndarray) -> RoadGuidanceResult | None:
        return self.update(segmentation_mask)


def format_direction(angle_deg: float | None) -> tuple[str, str, tuple[int, int, int]]:
    if angle_deg is None:
        return "Searching", "Road center unavailable", (180, 180, 180)
    if abs(angle_deg) < 2.0:
        return "Straight", "Hold current heading", (80, 220, 80)
    if angle_deg > 0:
        return "Right", f"Steer right {abs(angle_deg):.1f} deg", (0, 210, 255)
    return "Left", f"Steer left {abs(angle_deg):.1f} deg", (0, 165, 255)


def _scaled(value: int, ui_scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * ui_scale)))


def draw_guidance_overlay(frame_bgr: np.ndarray, guidance: RoadGuidanceResult | None, max_angle: float, ui_scale: float = 1.0) -> np.ndarray:
    frame = frame_bgr.copy()
    if guidance is not None:
        overlay = frame.copy()
        point_radius = _scaled(2, ui_scale)
        line_thickness = _scaled(3, ui_scale)
        marker_radius = _scaled(6, ui_scale)
        ref_thickness = _scaled(2, ui_scale)

        for x, y in guidance.left_points:
            cv2.circle(overlay, (x, y), point_radius, GUIDANCE_COLORS_BGR["left"], -1)
        for x, y in guidance.right_points:
            cv2.circle(overlay, (x, y), point_radius, GUIDANCE_COLORS_BGR["right"], -1)
        if guidance.fitted_points:
            cv2.polylines(
                overlay,
                [np.array(guidance.fitted_points, dtype=np.int32)],
                isClosed=False,
                color=GUIDANCE_COLORS_BGR["center"],
                thickness=line_thickness,
            )
        cv2.circle(overlay, guidance.lookahead_point, marker_radius, GUIDANCE_COLORS_BGR["lookahead"], -1)
        cv2.line(
            overlay,
            (guidance.image_center_x, guidance.y_near),
            (guidance.image_center_x, guidance.y_lookahead),
            GUIDANCE_COLORS_BGR["reference"],
            ref_thickness,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    displayed_angle = 0.0 if guidance is None else guidance.steering_angle_deg
    direction_label, action_label, color = format_direction(None if guidance is None else displayed_angle)
    path_heading = 0.0 if guidance is None else guidance.path_heading_deg
    cross_track_px = 0.0 if guidance is None else guidance.cross_track_error_px
    cross_track_norm = 0.0 if guidance is None else guidance.cross_track_error_norm
    confidence = 0.0 if guidance is None else guidance.confidence
    method = "none" if guidance is None else guidance.method
    controller = "none" if guidance is None else guidance.controller

    panel_left, panel_top = _scaled(20, ui_scale), _scaled(20, ui_scale)
    panel_width, panel_height = _scaled(340, ui_scale), _scaled(136, ui_scale)
    panel_right, panel_bottom = panel_left + panel_width, panel_top + panel_height
    border = _scaled(2, ui_scale)
    title_y = panel_top + _scaled(20, ui_scale)
    row1_y = panel_top + _scaled(42, ui_scale)
    row2_y = panel_top + _scaled(62, ui_scale)
    row3_y = panel_top + _scaled(82, ui_scale)
    row4_y = panel_top + _scaled(102, ui_scale)
    row5_y = panel_top + _scaled(120, ui_scale)
    text_x = panel_left + _scaled(12, ui_scale)
    title_scale = max(0.35, 0.62 * ui_scale)
    body_scale = max(0.32, 0.50 * ui_scale)
    meta_scale = max(0.28, 0.42 * ui_scale)
    body_thickness = _scaled(2, ui_scale)
    meta_thickness = _scaled(1, ui_scale)

    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (20, 20, 20), -1)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), color, border)
    cv2.putText(frame, "Road Guidance", (text_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (240, 240, 240), body_thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Steering Cmd: {displayed_angle:+.1f} deg", (text_x, row1_y), cv2.FONT_HERSHEY_SIMPLEX, body_scale, color, body_thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Path Heading: {path_heading:+.1f} deg", (text_x, row2_y), cv2.FONT_HERSHEY_SIMPLEX, body_scale, (240, 240, 240), body_thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Cross-track: {cross_track_px:+.0f} px ({cross_track_norm:+.2f})",
        (text_x, row3_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        meta_scale,
        (220, 220, 220),
        meta_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(frame, f"{direction_label}: {action_label}", (text_x, row4_y), cv2.FONT_HERSHEY_SIMPLEX, meta_scale, (220, 220, 220), meta_thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Conf: {confidence:.2f}  Ctrl: {controller}  Method: {method}",
        (text_x, row5_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        meta_scale,
        (220, 220, 220),
        meta_thickness,
        cv2.LINE_AA,
    )

    gauge_left = _scaled(22, ui_scale)
    gauge_top = frame.shape[0] - _scaled(42, ui_scale)
    gauge_width, gauge_height = _scaled(165, ui_scale), _scaled(16, ui_scale)
    cv2.rectangle(frame, (gauge_left, gauge_top), (gauge_left + gauge_width, gauge_top + gauge_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (gauge_left, gauge_top), (gauge_left + gauge_width, gauge_top + gauge_height), (220, 220, 220), 1)
    gauge_center_x = gauge_left + gauge_width // 2
    cv2.line(
        frame,
        (gauge_center_x, gauge_top - _scaled(4, ui_scale)),
        (gauge_center_x, gauge_top + gauge_height + _scaled(4, ui_scale)),
        (255, 255, 255),
        border,
    )

    clamped = float(np.clip(displayed_angle, -max_angle, max_angle))
    offset = int((clamped / max_angle) * (gauge_width // 2 - _scaled(6, ui_scale)))
    cv2.rectangle(
        frame,
        (gauge_center_x, gauge_top + _scaled(3, ui_scale)),
        (gauge_center_x + offset, gauge_top + gauge_height - _scaled(3, ui_scale)),
        color,
        -1,
    )
    label_scale = max(0.26, 0.38 * ui_scale)
    cv2.putText(frame, "L", (gauge_left, gauge_top - _scaled(6, ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "R",
        (gauge_left + gauge_width - _scaled(10, ui_scale), gauge_top - _scaled(6, ui_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame
