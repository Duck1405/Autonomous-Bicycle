"""Frame preprocessing for the three Pathfinder models.

Shared by trt_video_benchmark.py and trt_vs_onnx_parity.py so the parity check
validates exactly the tensors the benchmark feeds the engines — if these drifted
apart, parity would pass on inputs nobody actually runs.

Each mirrors the torch/ONNX path it replaces:
  laneatt -> LaneATT.frame_eval
  yolo    -> Yolov11/jetson_infer_onnx.py letterbox
  depth   -> lib/depth.py DPTImageProcessor
"""
import cv2
import numpy as np

# ImageNet stats used by Depth-Anything-V2's DPTImageProcessor (see lib/depth.py).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pre_laneatt(frame):
    """BGR frame -> (1,3,360,640) /255, BGR order kept (mirrors LaneATT.frame_eval)."""
    img = cv2.resize(frame, (640, 360))
    arr = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def pre_yolo(frame, size=640):
    """Letterbox to size x size, gray 114 pad, RGB, /255 (mirrors jetson_infer_onnx.py)."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, dtype=np.float32) / 255.0


def pre_depth(frame, size=518):
    """RGB, bicubic resize, /255, ImageNet-normalize (mirrors DepthInferenceONNX)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


# Keyed by the label used on the command line of the benchmark / parity scripts.
PREPROCESSORS = {"laneatt": pre_laneatt, "yolo": pre_yolo, "depth": pre_depth}
