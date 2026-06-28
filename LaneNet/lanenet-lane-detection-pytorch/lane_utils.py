"""Lane detection utilities: clustering, polynomial fitting, drawing, ego-lane tracking."""

import warnings

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation

# np.RankWarning moved to np.exceptions.RankWarning in numpy >= 2.0.
try:
    _RANK_WARNING = np.exceptions.RankWarning
except AttributeError:
    _RANK_WARNING = np.RankWarning

from model.lanenet.backbone.H_Net import build_H
from model.lanenet.hnet_loss import fit_lane_bev


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EGO_LEFT_COLOR_BGR  = (255, 180,   0)
EGO_RIGHT_COLOR_BGR = (  0, 180, 255)

LANE_COLORS_BGR = np.array([
    [0, 0, 255], [0, 255, 0], [255, 128, 0], [0, 255, 255],
    [255, 0, 255], [255, 255, 0], [128, 255, 0], [128, 128, 0],
    [255, 255, 128], [128, 128, 128],
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def mean_shift_center(embeddings, center, bandwidth, max_iters):
    center = center.astype(np.float32, copy=True)
    for _ in range(max_iters):
        distances = np.linalg.norm(embeddings - center, axis=1)
        neighbors = embeddings[distances <= bandwidth]
        if len(neighbors) == 0:
            break
        next_center = np.mean(neighbors, axis=0)
        if np.linalg.norm(next_center - center) < 1e-3:
            center = next_center
            break
        center = next_center
    return center


def cluster_lane_embeddings(binary_pred, instance_embedding, delta_v=0.5,
                            cluster_radius=None, mean_shift_bandwidth=None,
                            mean_shift_iters=10, min_cluster_size=500,
                            max_lanes=10):
    lane_mask = binary_pred == 1
    cluster_labels = np.zeros(binary_pred.shape, dtype=np.int32)
    if not np.any(lane_mask):
        return cluster_labels

    if instance_embedding.shape[1:] != binary_pred.shape:
        raise ValueError("Instance embedding shape {} does not match binary mask shape {}".format(
            instance_embedding.shape, binary_pred.shape))

    radius = 2.0 * delta_v if cluster_radius is None else cluster_radius
    bandwidth = radius if mean_shift_bandwidth is None else mean_shift_bandwidth

    lane_ys, lane_xs = np.where(lane_mask)
    lane_embeddings = instance_embedding[:, lane_ys, lane_xs].T
    remaining = np.ones(lane_embeddings.shape[0], dtype=bool)
    rng = np.random.default_rng(0)
    cluster_id = 1

    while np.any(remaining) and cluster_id <= max_lanes:
        remaining_indices = np.flatnonzero(remaining)
        seed_index = rng.choice(remaining_indices)
        center = mean_shift_center(
            lane_embeddings[remaining], lane_embeddings[seed_index],
            bandwidth, mean_shift_iters,
        )
        distances = np.linalg.norm(lane_embeddings - center, axis=1)
        cluster = (distances <= radius) & remaining
        cluster_size = np.count_nonzero(cluster)

        if cluster_size < min_cluster_size:
            remaining[cluster if cluster_size > 0 else np.array([seed_index])] = False
            continue

        cluster_labels[lane_ys[cluster], lane_xs[cluster]] = cluster_id
        remaining[cluster] = False
        cluster_id += 1

    return _sort_clusters_left_to_right(cluster_labels)


def _sort_clusters_left_to_right(cluster_labels):
    sorted_labels = np.zeros_like(cluster_labels)
    lane_ids = [lid for lid in np.unique(cluster_labels) if lid != 0]
    lane_positions = []
    for lane_id in lane_ids:
        ys, xs = np.where(cluster_labels == lane_id)
        if len(xs) == 0:
            continue
        lower_half = ys >= np.percentile(ys, 50)
        x_pos = np.mean(xs[lower_half]) if np.any(lower_half) else np.mean(xs)
        lane_positions.append((x_pos, lane_id))
    for new_id, (_, old_id) in enumerate(sorted(lane_positions), start=1):
        sorted_labels[cluster_labels == old_id] = new_id
    return sorted_labels


# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

def fit_lane_polynomials(cluster_labels, H, lanenet_w, lanenet_h, poly_order=3):
    """Return (polys, y_ranges, curves_norm) for each lane cluster.

    polys      : {lane_id: coeff array (poly_order+1,)}, x = polyval(w, y)
    y_ranges   : {lane_id: (y_min, y_max)} in normalised [0,1] coords — the
                 range where the polynomial is valid. Drawing outside this
                 range causes wild 3rd-order extrapolation artefacts.
    curves_norm: {lane_id: (N,2) array} sampled points already clipped to the
                 valid y range, in normalised coords ready for reprojection.
    """
    polys, y_ranges, curves_norm = {}, {}, {}
    for lane_id in [lid for lid in np.unique(cluster_labels) if lid != 0]:
        ys, xs = np.where(cluster_labels == lane_id)
        if len(xs) < poly_order + 1:
            continue

        pts_norm = torch.tensor(
            np.stack([xs / lanenet_w, ys / lanenet_h], axis=1), dtype=torch.float32)

        try:
            if H is not None:
                curve_norm = fit_lane_bev(H, pts_norm, order=poly_order, n_samples=200)
                if curve_norm is None:
                    continue
                curve_np = curve_norm.cpu().numpy()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", _RANK_WARNING)
                    w = np.polyfit(curve_np[:, 1], curve_np[:, 0], poly_order)
                y_min, y_max = float(curve_np[:, 1].min()), float(curve_np[:, 1].max())
            else:
                xs_n, ys_n = xs / lanenet_w, ys / lanenet_h
                with warnings.catch_warnings():
                    warnings.simplefilter("error", _RANK_WARNING)
                    w = np.polyfit(ys_n, xs_n, poly_order)
                y_min, y_max = float(ys_n.min()), float(ys_n.max())
                y_samp = np.linspace(y_min, y_max, 200)
                curve_np = np.stack([np.polyval(w, y_samp), y_samp], axis=1)
        except _RANK_WARNING:
            continue  # skip this cluster — poorly conditioned fit, not a real lane

        polys[lane_id] = w
        y_ranges[lane_id] = (y_min, y_max)
        curves_norm[lane_id] = curve_np

    return polys, y_ranges, curves_norm


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_lane_clusters(frame_bgr, cluster_labels, dilation_iters=2):
    orig_h, orig_w = frame_bgr.shape[:2]
    labels_orig = cv2.resize(cluster_labels.astype(np.int32), (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)
    overlay = frame_bgr.copy()
    for i, lid in enumerate([l for l in np.unique(labels_orig) if l != 0][:len(LANE_COLORS_BGR)]):
        mask = labels_orig == lid
        if dilation_iters > 0:
            mask = binary_dilation(mask, iterations=dilation_iters)
        overlay[mask] = LANE_COLORS_BGR[i]
    return cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)


def draw_all_lane_curves(frame_bgr, curves_norm, thickness=4):
    orig_h, orig_w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    for i, (_, curve_np) in enumerate(curves_norm.items()):
        if i >= len(LANE_COLORS_BGR):
            break
        px = curve_np.copy()
        px[:, 0] = np.clip(px[:, 0] * orig_w, 0, orig_w - 1)
        px[:, 1] = np.clip(px[:, 1] * orig_h, 0, orig_h - 1)
        cv2.polylines(out, [px.astype(np.int32).reshape(-1, 1, 2)], False,
                      tuple(int(c) for c in LANE_COLORS_BGR[i]), thickness, cv2.LINE_AA)
    return out


def draw_ego_lanes(frame_bgr, left_w, right_w, left_conf, right_conf,
                   left_y_range=None, right_y_range=None,
                   thickness=6, min_conf=0.2):
    """Draw ego lane boundaries, clipped to the y range where the polynomial
    was actually fitted. Drawing outside that range causes wild extrapolation
    artefacts with a 3rd-order polynomial.
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    def _draw(w, conf, color, y_range):
        if w is None or conf < min_conf:
            return
        # Clamp to the detected range; always extend to the bottom of the image
        # (y=0.98) since lanes appear below their topmost detected point too,
        # but never extrapolate upward past the topmost fitted point.
        y_min = y_range[0] if y_range is not None else 0.3
        y_max = 0.98
        y = np.linspace(y_min, y_max, 300)
        px = np.stack([np.clip(np.polyval(w, y) * orig_w, 0, orig_w - 1),
                       y * orig_h], axis=1).astype(np.int32)
        faded = tuple(int(c * min(conf, 1.0)) for c in color)
        cv2.polylines(out, [px.reshape(-1, 1, 2)], False, faded, thickness, cv2.LINE_AA)

    _draw(left_w,  left_conf,  EGO_LEFT_COLOR_BGR,  left_y_range)
    _draw(right_w, right_conf, EGO_RIGHT_COLOR_BGR, right_y_range)

    labels = ["EGO-L"] * (left_w  is not None and left_conf  >= min_conf) + \
             ["EGO-R"] * (right_w is not None and right_conf >= min_conf)
    if labels:
        cv2.putText(out, "  ".join(labels), (10, orig_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Ego-lane tracker
# ---------------------------------------------------------------------------

class EgoLaneTracker:
    """Stable ego-lane identification via straddling + continuity guard + EMA.

    Each frame:
      1. Evaluate all polynomials at y=0.9 (near vehicle).
      2. Ego-left  = rightmost lane with x < 0.5 (left of image centre).
         Ego-right = leftmost  lane with x > 0.5 (right of image centre).
      3. Reject a new candidate if its coefficients differ from the previous
         frame by more than max_coeff_delta (L2 norm) — continuity guard.
      4. Blend accepted coefficients with EMA at rate alpha.
    """

    def __init__(self, alpha=0.25, max_coeff_delta=0.35, hold_decay=0.85):
        self.alpha = alpha
        self.max_coeff_delta = max_coeff_delta
        self.hold_decay = hold_decay
        self._left_w  = self._right_w  = None
        self._left_conf = self._right_conf = 0.0
        self._left_y_range  = None   # (y_min, y_max) of the fitted polynomial
        self._right_y_range = None

    def _accept(self, new_w, prev_w):
        if prev_w is None:
            return new_w.copy(), True
        if np.linalg.norm(new_w - prev_w) > self.max_coeff_delta:
            return prev_w.copy(), False
        return (1.0 - self.alpha) * prev_w + self.alpha * new_w, True

    def update(self, lane_polys, y_ranges=None, order=3):
        if y_ranges is None:
            y_ranges = {}
        bottom_xs = {lid: float(np.polyval(w, 0.9)) for lid, w in lane_polys.items()}
        left_cands  = {lid: x for lid, x in bottom_xs.items() if x < 0.5}
        right_cands = {lid: x for lid, x in bottom_xs.items() if x > 0.5}

        ego_left_id  = max(left_cands,  key=left_cands.get)  if left_cands  else None
        ego_right_id = min(right_cands, key=right_cands.get) if right_cands else None

        for ego_id, attr_w, attr_c, attr_y in [
            (ego_left_id,  "_left_w",  "_left_conf",  "_left_y_range"),
            (ego_right_id, "_right_w", "_right_conf", "_right_y_range"),
        ]:
            if ego_id is not None:
                smoothed, accepted = self._accept(lane_polys[ego_id], getattr(self, attr_w))
                setattr(self, attr_w, smoothed)
                setattr(self, attr_c, 1.0 if accepted else getattr(self, attr_c) * self.hold_decay)
                setattr(self, attr_y, y_ranges.get(ego_id))
            else:
                setattr(self, attr_c, getattr(self, attr_c) * self.hold_decay)

        return (self._left_w,  self._right_w,
                self._left_conf, self._right_conf,
                self._left_y_range, self._right_y_range)

    def reset(self):
        self._left_w = self._right_w = None
        self._left_conf = self._right_conf = 0.0
        self._left_y_range = self._right_y_range = None
