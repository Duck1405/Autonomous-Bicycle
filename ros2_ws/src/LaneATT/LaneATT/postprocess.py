"""Torch-free decode and drawing for the three Pathfinder models.

The repo's decode paths all run through torch: LaneATT's lane-NMS is a compiled
torch CUDA extension (lib/nms), YOLO drawing goes through ultralytics
Results.plot(), and the depth wrapper carries a PIL round-trip. None of that is
available on the Jetson, where torch is not installed. Everything here is numpy
+ cv2 and produces the same overlays.

Ports, with the original as the reference:
  lane_nms            <- lib/nms_pytorch.py:nms (itself a port of nms_kernel.cu)
  proposals_to_pred   <- lib/models/laneatt.py:proposals_to_pred
  laneatt_decode      <- lib/models/laneatt.py:nms + decode
  LaneHysteresis      <- lib/LaneATT.py:filter_lanes_hysteresis
  lanes_to_px         <- lib/LaneATT.py:lanes_to_px
  draw_lanes          <- lib/video.py:222-243
  yolo_decode/draw    <- Yolov11/jetson_infer_onnx.py:97-106
  depth_colorize      <- lib/depth.py:_postprocess + depthInference.py:79
"""
import cv2
import numpy as np

# ---------------------------------------------------------------- LaneATT ---

# Proposal layout (see lib/nms_pytorch.py:7): the engine emits 77 values per
# anchor as [logit_bg, logit_fg, start_y, start_x, length, x_0 ... x_71]. The
# anchors are added inside the exported graph (laneatt.py:106-110), so no anchor
# table is needed here; x_* are already in input-pixel space (0..640).


def _softmax2(logits):
    """Row-wise softmax over a (N, 2) block, max-subtracted for stability."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def lane_nms(proposals, scores, overlap=50.0, top_k=2):
    """Greedy lane NMS. Returns kept indices into `proposals`, best score first.

    "Overlap" is the mean absolute horizontal distance between two lanes over the
    vertical span where both exist; the lower-scored lane is suppressed when that
    mean is below `overlap` (in input pixels).
    """
    n = proposals.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    n_offsets = proposals.shape[1] - 5
    n_strips = n_offsets - 1
    order = np.argsort(-scores, kind="stable")

    # A non-positive threshold can never suppress (mean distance is >= 0 and the
    # test is strict), so short-circuit the O(N^2) loop. Matches the fast path in
    # lib/nms_pytorch.py.
    if overlap <= 0:
        return order[:min(n, int(top_k))]

    boxes = proposals[order]
    # Vertical extent in offset-index space, matching devIoU in the CUDA kernel.
    starts = (boxes[:, 2] * n_strips + 0.5).astype(np.int64)
    lengths = boxes[:, 4]
    ends = ((starts.astype(np.float64) + lengths - 1 + 0.5).astype(np.int64)
            - (lengths - 1 < 0).astype(np.int64))
    ends = np.minimum(ends, n_offsets - 1)
    xs = boxes[:, 5:5 + n_offsets]
    offset_idx = np.arange(n_offsets)

    removed = np.zeros(n, dtype=bool)
    keep = []
    for i in range(n):
        if removed[i]:
            continue
        keep.append(order[i])
        if len(keep) == int(top_k):
            break
        rest = slice(i + 1, n)
        s = np.maximum(starts[i], starts[rest])
        e = np.minimum(ends[i], ends[rest])
        valid = (offset_idx[None, :] >= s[:, None]) & (offset_idx[None, :] <= e[:, None])
        diff = np.abs(xs[i][None, :] - xs[rest]) * valid
        counts = valid.sum(axis=1)
        mean_dist = diff.sum(axis=1) / np.maximum(counts, 1)
        suppress = (counts > 0) & (mean_dist < overlap) & (~removed[rest])
        removed[i + 1:][suppress] = True

    return np.asarray(keep, dtype=np.int64)


def proposals_to_pred(proposals, img_w=640):
    """Post-NMS proposals -> [{'points': (N,2) normalized (x,y), 'conf': float}].

    Mirrors laneatt.py:proposals_to_pred, including its rule that a proposal not
    starting at the bottom of the image is extended upward only while x stays
    inside the frame. The original's `np.bool` (removed in numpy >= 1.24) is
    written as `bool` here.
    """
    if len(proposals) == 0:
        return []
    n_offsets = proposals.shape[1] - 5
    n_strips = n_offsets - 1
    anchor_ys = np.linspace(1.0, 0.0, n_offsets)

    lanes = []
    for lane in proposals:
        lane_xs = lane[5:].astype(np.float64) / img_w
        # Clamped: a degenerate proposal can produce start < 0, which would slice
        # from the end of the array instead of erroring. The torch original has
        # the same hazard; clamping only affects proposals that are already junk.
        start = int(round(float(lane[2]) * n_strips))
        start = max(0, min(start, n_offsets))
        length = int(round(float(lane[4])))
        end = max(min(start + length - 1, n_offsets - 1), -1)

        head = lane_xs[:start]
        inside = (head >= 0.0) & (head <= 1.0)
        # Reverse cumprod: keep the run of in-frame points adjacent to `start`.
        mask = ~(inside[::-1].cumprod()[::-1].astype(bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2

        valid = lane_xs >= 0
        if valid.sum() <= 1:
            continue
        pts_x = lane_xs[valid][::-1]
        pts_y = anchor_ys[valid][::-1]
        lanes.append({
            "points": np.stack([pts_x, pts_y], axis=1),
            "conf": float(lane[1]),
            "start_x": float(lane[3]),
            "start_y": float(lane[2]),
        })
    return lanes


def laneatt_decode(raw, conf_threshold=0.3, nms_thres=50.0, nms_topk=2, img_w=640):
    """Raw (1,1000,77) engine output -> list of lane dicts.

    Order matters: laneatt.py:nms scores with a softmax written to a *separate*
    tensor and slices the still-logit proposals, then decode() applies the softmax
    to the survivors. Softmaxing twice would squash every conf toward 0.5 and
    break the 0.5 acquire threshold downstream.
    """
    p = np.asarray(raw, dtype=np.float32).reshape(-1, np.shape(raw)[-1])
    scores = _softmax2(p[:, :2])[:, 1]

    above = scores > conf_threshold
    p, scores = p[above], scores[above]
    if p.shape[0] == 0:
        return []

    p = p[lane_nms(p, scores, overlap=nms_thres, top_k=nms_topk)].copy()
    p[:, :2] = _softmax2(p[:, :2])
    p[:, 4] = np.round(p[:, 4])
    return proposals_to_pred(p, img_w=img_w)


class LaneHysteresis:
    """Two-threshold acceptance, ported from LaneATT.py:filter_lanes_hysteresis.

    A new lane must score >= conf_threshold, but one whose bottom-row x matches a
    lane accepted in the previous frame survives down to keep_threshold. Stops
    real lanes blinking in and out when their score sits on a single threshold.
    Stateful: call reset() between videos.
    """

    def __init__(self, conf_threshold=0.5, keep_threshold=0.3, match_tolerance=0.05):
        self.conf_threshold = conf_threshold
        self.keep_threshold = keep_threshold
        self.match_tolerance = match_tolerance
        self.prev_lane_xs = []

    def reset(self):
        self.prev_lane_xs = []

    def __call__(self, lanes):
        accepted, accepted_xs = [], []
        for lane in lanes:
            pts = lane["points"]
            x_bottom = float(pts[np.argmax(pts[:, 1]), 0])  # largest y = nearest
            tracked = any(abs(x_bottom - px) < self.match_tolerance
                          for px in self.prev_lane_xs)
            if lane["conf"] >= self.conf_threshold or (
                    tracked and lane["conf"] >= self.keep_threshold):
                accepted.append(lane)
                accepted_xs.append(x_bottom)
        self.prev_lane_xs = accepted_xs
        return accepted


def lanes_to_px(lanes, w, h):
    """Normalized lane points -> integer pixel points at (w, h)."""
    out = []
    for lane in lanes:
        pts = lane["points"].astype(float).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append(pts.round().astype(int))
    return out


def draw_lanes(frame, lanes, color=(0, 255, 0), thickness=3):
    """Draw lane polylines in place (video.py uses green, width 3)."""
    for pts in lanes_to_px(lanes, frame.shape[1], frame.shape[0]):
        for p0, p1 in zip(pts[:-1], pts[1:]):
            cv2.line(frame, tuple(p0), tuple(p1), color, thickness)
    return frame


# ------------------------------------------------------------------- YOLO ---

# The engine is a 4-class custom model, not COCO-80 (jetson_infer_onnx.py:26).
NAMES = ["person", "vehicle", "traffic-light", "stop-sign"]
COLORS = [(0, 255, 0), (255, 160, 0), (0, 215, 255), (0, 0, 255)]  # BGR


def yolo_decode(raw, r, dx, dy, conf_threshold=0.35):
    """(1,300,6) engine output -> [(p1, p2, conf, cls)] in original frame coords.

    NMS is already inside the graph; rows are [x1,y1,x2,y2,conf,cls] in
    letterboxed 640-space and padding rows have conf == 0.
    """
    dets = np.asarray(raw, dtype=np.float32).reshape(-1, 6)
    out = []
    for x1, y1, x2, y2, conf, cls in dets[dets[:, 4] >= conf_threshold]:
        p1 = (int((x1 - dx) / r), int((y1 - dy) / r))
        p2 = (int((x2 - dx) / r), int((y2 - dy) / r))
        out.append((p1, p2, float(conf), int(cls)))
    return out


def draw_boxes(frame, dets):
    """Draw boxes + labels in place, matching jetson_infer_onnx.py:97-106."""
    for p1, p2, conf, cls in dets:
        color = COLORS[cls % len(COLORS)]
        name = NAMES[cls] if cls < len(NAMES) else str(cls)
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (p1[0], max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


# ------------------------------------------------------------------ Depth ---

def depth_colorize(raw, w, h, colormap=cv2.COLORMAP_INFERNO):
    """(1,518,518) relative depth -> (h, w, 3) BGR inferno image.

    Normalization is per frame (depth.py:105), so absolute brightness flickers
    between frames — only within-frame structure is meaningful.
    """
    depth = np.asarray(raw, dtype=np.float32).reshape(np.shape(raw)[-2:])
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    d_min, d_max = float(depth.min()), float(depth.max())
    norm = (depth - d_min) / (d_max - d_min + 1e-8)  # 0..1, high = closer
    return cv2.applyColorMap((norm * 255.0).astype(np.uint8), colormap)
