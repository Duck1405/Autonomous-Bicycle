import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Placeholder: the camera has never been calibrated (no checkerboard run, no
# measured focal length anywhere in this repo). 70 degrees is a typical wide
# webcam/action-cam horizontal FOV. Every degree this is wrong scales the
# cross-track term by the same factor — replace with the measured HFOV from
# cv2.calibrateCamera before trusting absolute angles.
ASSUMED_HFOV_DEG = 90.0

# Standard US/CA travel-lane width. Dividing it by the lane's pixel width at a
# given image row yields meters-per-pixel at that row — a per-frame, per-row
# pixel->meter conversion that needs no camera calibration. Wrong by whatever
# factor the real lane differs from 3.7 m (residential lanes run ~3.0-3.4 m).
ASSUMED_LANE_WIDTH_M = 3.7


@dataclass
class SteeringResult:
    steering_angle_deg: float   # + = steer right, - = steer left
    heading_error_deg: float    # lane direction vs straight-ahead, same sign
    cross_track_px: float       # bottom midpoint offset from image center
    cross_track_deg: float      # cross_track_px through the (assumed) HFOV
    confidence: float           # 1.0 fresh, decays by hold_decay while holding
    held: bool                  # True when this frame reused the last command
    cross_track_m: Optional[float] = None  # meters, via lane-width scale; None
                                           # when edges were unavailable


@dataclass
class EgoVehicleResult:
    box_xyxy: tuple             # (x1, y1, x2, y2) pixels
    bottom_center: tuple        # (cx, y2) — the point tested against the lane
    confidence: float
    track_id: Optional[int]    # None until yolo runs model.track(persist=True)


class Angle:
    """Steering + lead-vehicle (CIPV) layer over get_ego_lanes() and YOLO.

    Coordinates match the rest of lib/: origin top-left, x right, y down,
    pixels — the same space as get_ego_lanes() output and boxes.xyxy.
    Positive steering means steer right.
    """

    def __init__(self, assumed_hfov_deg=ASSUMED_HFOV_DEG, stanley_gain=1.0,
                 nominal_speed=3.0, heading_window_px=None, smoothing_alpha=0.2,
                 hold_decay=0.85, max_extrapolation_px=60.0, vehicle_class_id=1,
                 lane_width_m=ASSUMED_LANE_WIDTH_M):
        self.assumed_hfov_deg = assumed_hfov_deg
        self.lane_width_m = lane_width_m
        self.stanley_gain = stanley_gain
        # Fallback only — pass the trike's measured speed into
        # compute_steering(speed=...) once telemetry is wired up.
        self.nominal_speed = nominal_speed
        # None -> per-frame default of max(8% of img_h, 12px).
        self.heading_window_px = heading_window_px
        self.smoothing_alpha = smoothing_alpha
        self.hold_decay = hold_decay
        self.max_extrapolation_px = max_extrapolation_px
        # 4-class checkpoint: {0: person, 1: vehicle, 2: traffic-light,
        # 3: stop-sign} — car/moto/bus/truck are all one 'vehicle' class.
        self.vehicle_class_id = vehicle_class_id
        self.reset_video_state()

    def reset_video_state(self):
        """Forget per-video temporal state (same convention as LaneATTInference)."""
        self._prev_steering_deg = None
        self._prev_heading_deg = None
        self._confidence = 0.0

    # ------------------------------------------------------------- steering

    def compute_steering(self, mid_points, img_w, img_h, speed=None,
                         left_points=None, right_points=None):
        """Stanley-style steering from lane midpoints. Returns SteeringResult,
        or None when there is nothing to steer from and no held value left.

        When left_points/right_points are given, cross-track is converted to
        METERS via the lane-width scale (lane_width_m / lane pixel width at
        the bottom row) and Stanley runs in its original real units; otherwise
        it falls back to the HFOV pseudo-degrees approximation."""
        if mid_points is None or len(mid_points) == 0:
            return self._hold()

        pts = np.asarray(mid_points, dtype=float)
        pts = pts[np.argsort(pts[:, 1])]          # ascending y: far -> near
        x_near, y_near = pts[-1]

        window = self.heading_window_px if self.heading_window_px is not None \
            else max(img_h * 0.08, 12.0)
        near = pts[pts[:, 1] >= y_near - window]

        # Heading: x = a*y + b fit through the near-car window. The slope is
        # dx/dy, a pure ratio, so this term needs no FOV assumption.
        heading_deg = None
        if len(near) >= 2 and (near[-1, 1] - near[0, 1]) > 2.0:
            a, b = np.polyfit(near[:, 1], near[:, 0], 1)
            y_far = near[0, 1]
            heading_deg = math.degrees(math.atan2(
                (a * y_far + b) - (a * y_near + b), y_near - y_far))
        if heading_deg is None:
            # Too few points / no y-spread: reuse the last heading so the
            # cross-track term still steers this frame.
            heading_deg = self._prev_heading_deg if self._prev_heading_deg is not None else 0.0

        cross_track_px = x_near - img_w / 2.0
        cross_track_deg = cross_track_px * (self.assumed_hfov_deg / img_w)
        cross_track_m = self._cross_track_meters(cross_track_px, y_near,
                                                 left_points, right_points)

        # Stanley: heading + atan(k * cross_track / v). With lane edges
        # available, cross_track is in real meters (pixels scaled by
        # lane_width_m / lane pixel width at this row) — the original
        # formula's units, so k ~ 1 is a sane start. Without edges we fall
        # back to HFOV pseudo-degrees, where k must absorb the unknown scale.
        v = self.nominal_speed if speed is None else max(float(speed), 0.1)
        error = cross_track_m if cross_track_m is not None else cross_track_deg
        steering_deg = heading_deg + math.degrees(
            math.atan2(self.stanley_gain * error, v))

        # EMA smoothing disabled to keep the controller as simple as possible;
        # re-enable if the raw command jitters too much on real footage.
        # alpha = self.smoothing_alpha
        # if self._prev_steering_deg is not None:
        #     steering_deg = (1 - alpha) * self._prev_steering_deg + alpha * steering_deg
        # if self._prev_heading_deg is not None:
        #     heading_deg = (1 - alpha) * self._prev_heading_deg + alpha * heading_deg
        self._prev_steering_deg = steering_deg
        self._prev_heading_deg = heading_deg
        self._confidence = 1.0

        return SteeringResult(steering_deg, heading_deg, cross_track_px,
                              cross_track_deg, 1.0, held=False,
                              cross_track_m=cross_track_m)

    def _cross_track_meters(self, cross_track_px, y_near, left_points,
                            right_points):
        """meters-per-pixel at row y_near from the lane's pixel width there:
        m_per_px = lane_width_m / (right_x - left_x). Returns None when the
        edges are missing or the width there is degenerate."""
        if left_points is None or right_points is None \
                or len(left_points) == 0 or len(right_points) == 0:
            return None
        left_ys, left_xs = self._edge_interp(left_points)
        right_ys, right_xs = self._edge_interp(right_points)
        if not (self._in_range(left_ys, y_near) and self._in_range(right_ys, y_near)):
            return None
        width_px = np.interp(y_near, right_ys, right_xs) \
            - np.interp(y_near, left_ys, left_xs)
        if width_px < 20.0:      # crossed/degenerate edges — don't divide by it
            return None
        return cross_track_px * (self.lane_width_m / width_px)

    def _hold(self):
        """No midpoints this frame: fade the held command toward straight-ahead
        so a long dropout never locks in a hard turn."""
        if self._prev_steering_deg is None:
            return None
        self._prev_steering_deg *= self.hold_decay
        self._confidence *= self.hold_decay
        heading = self._prev_heading_deg if self._prev_heading_deg is not None else 0.0
        return SteeringResult(self._prev_steering_deg, heading, 0.0, 0.0,
                              self._confidence, held=True)

    # --------------------------------------------------------- lead vehicle

    def select_ego_vehicle(self, yolo_results, left_points, right_points):
        """CIPV pick: vehicles whose bottom-center sits inside the ego lane,
        nearest one (largest bottom y) wins. Returns EgoVehicleResult or None."""
        if yolo_results is None or len(yolo_results) == 0:
            return None
        if left_points is None or right_points is None:
            return None
        boxes = yolo_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        left_ys, left_xs = self._edge_interp(left_points)
        right_ys, right_xs = self._edge_interp(right_points)

        best = None
        for k in np.flatnonzero(cls == self.vehicle_class_id):
            x1, y1, x2, y2 = xyxy[k]
            cx = (x1 + x2) / 2.0
            # np.interp clamps outside the observed rows (flat extrapolation);
            # a clamp further than max_extrapolation_px from both edges' data
            # is a guess, not a measurement — reject it.
            if not (self._in_range(left_ys, y2) or self._in_range(right_ys, y2)):
                continue
            if not (np.interp(y2, left_ys, left_xs) < cx < np.interp(y2, right_ys, right_xs)):
                continue
            if best is None or y2 > best.bottom_center[1]:
                best = EgoVehicleResult(
                    box_xyxy=tuple(float(v) for v in xyxy[k]),
                    bottom_center=(cx, float(y2)),
                    confidence=float(conf[k]),
                    track_id=int(ids[k]) if ids is not None else None)
        return best

    def _in_range(self, edge_ys, y):
        return edge_ys[0] - self.max_extrapolation_px <= y <= edge_ys[-1] + self.max_extrapolation_px

    @staticmethod
    def _edge_interp(points):
        """(x, y) points -> (sorted unique ys, mean x per y) for np.interp.
        Edges from get_ego_lanes() aren't guaranteed sorted or duplicate-free."""
        pts = np.asarray(points, dtype=float)
        ys, inverse = np.unique(pts[:, 1], return_inverse=True)
        xs = np.bincount(inverse, weights=pts[:, 0]) / np.bincount(inverse)
        return ys, xs

    def distance_to_ego_vehicle(self, ego_vehicle):
        """Not implemented yet. Needs either a calibrated camera + measured
        mounting height for ground-plane projection (d = f*h/dy below horizon)
        or a monocular depth model; neither exists in this repo yet."""
        return None

    # -------------------------------------------------------------- drawing

    def draw_overlay(self, frame, steering, ego_vehicle):
        """Steering panel + gauge + direction arrow, and the CIPV box in solid
        red on top of YOLO's own drawing. Frame is BGR, returned modified."""
        h, w = frame.shape[:2]

        if ego_vehicle is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in ego_vehicle.box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = "LEAD (CIPV)"
            if ego_vehicle.track_id is not None:
                label += f" #{ego_vehicle.track_id}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Darkened backing so the text reads over any road surface.
        x0, y0, pw, ph = 8, 8, 252, 108
        panel = frame[y0:y0 + ph, x0:x0 + pw]
        frame[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(
            panel, 0.45, np.zeros_like(panel), 0.55, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if steering is None:
            cv2.putText(frame, "STEER --", (x0 + 8, y0 + 24), font, 0.55,
                        (200, 200, 200), 2)
        else:
            color = (0, 200, 255) if steering.held else (255, 255, 255)
            lines = [
                f"STEER {steering.steering_angle_deg:+6.1f} deg"
                + (" (held)" if steering.held else ""),
                f"HEAD  {steering.heading_error_deg:+6.1f} deg",
                f"XTRK  {steering.cross_track_m:+6.2f} m"
                if steering.cross_track_m is not None
                else f"XTRK  {steering.cross_track_px:+6.0f} px",
                f"CONF  {steering.confidence:4.2f}",
            ]
            for j, line in enumerate(lines):
                cv2.putText(frame, line, (x0 + 8, y0 + 24 + 18 * j), font, 0.5,
                            color, 1, cv2.LINE_AA)

            # Gauge: marker position = steering clamped to +-25 deg.
            gx0, gx1, gy = x0 + 8, x0 + pw - 8, y0 + ph - 10
            cv2.line(frame, (gx0, gy), (gx1, gy), (120, 120, 120), 2)
            center = (gx0 + gx1) // 2
            cv2.line(frame, (center, gy - 5), (center, gy + 5), (200, 200, 200), 1)
            frac = max(-1.0, min(1.0, steering.steering_angle_deg / 25.0))
            mx = int(center + frac * (gx1 - center))
            cv2.circle(frame, (mx, gy), 5, (0, 0, 255) if steering.held else (0, 255, 0), -1)

            # Direction arrow from the bottom-center of the frame.
            rad = math.radians(steering.steering_angle_deg)
            base = (w // 2, h - 16)
            tip = (int(base[0] + 90 * math.sin(rad)), int(base[1] - 90 * math.cos(rad)))
            cv2.arrowedLine(frame, base, tip,
                            (0, 200, 255) if steering.held else (0, 255, 0),
                            3, tipLength=0.25)

        return frame
