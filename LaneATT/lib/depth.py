import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (pipeline, AutoConfig, AutoImageProcessor,
                          AutoModelForDepthEstimation)

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_ONNX = "depth_onnx/depth_anything_v2_small.onnx"
# ImageNet normalization + input size from depth_model/preprocessor_config.json.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_ONNX_SIZE = 518  # fixed square size the ONNX graph was exported at

# Print the current working directory
print(os.getcwd())

class DepthInference():
    """Monocular depth via Depth-Anything-V2 (transformers pipeline).

    Output is RELATIVE depth (larger value = closer), not meters — turning it
    into a distance needs a scale reference (e.g. lane width or a known box).
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        if device is None:
            # cuda when present, else cpu. Deliberately NOT mps: Depth-Anything's
            # bicubic upsample (aten::upsample_bicubic2d) isn't implemented on the
            # Apple MPS backend, so a pipeline auto-placed on mps crashes at infer.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.device = device
        # print(os.getcwd())
        # if Path()
        if Path("depth_model").exists():
            print("load model")
            self.pipe = self.load_model()
        else:
            print("save model")
            self.pipe = self.save_model()
        

    def infer(self, frame):
        """frame: BGR numpy array (as read by cv2). Returns the pipeline dict:
        'predicted_depth' (raw torch tensor at model resolution) and 'depth'
        (PIL image, 0-255 normalized, resized back to the input size)."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.pipe(image)
    
    def load_model(self):
        # Pass device explicitly, otherwise transformers auto-selects mps on
        # Apple Silicon (which Depth-Anything can't run on — see __init__).
        pipe = pipeline("depth-estimation", model="depth_model", device=self.device)
        return pipe
    
    def save_model(self):
        pipe = pipeline("depth-estimation", model=self.checkpoint, device=self.device)
        pipe.save_pretrained("depth_model")
        return pipe
    

    def convert_onnx(self, onnx):
        model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint)
        print(model)
        pass


class DepthInferenceONNX:
    """Depth-Anything-V2 inference via the exported ONNX graph (onnxruntime).

    Drop-in for DepthInference: give it a BGR numpy frame (as cv2 reads it) and
    infer() returns the same dict shape the transformers pipeline does —
    'predicted_depth' (raw float depth at the frame's resolution) and 'depth'
    (PIL image, 0-255 normalized, brighter = closer) — so depth2.py works
    unchanged.

    The ONNX graph is ONLY the network forward at a fixed 518x518. The resize +
    ImageNet normalization the pipeline's DPTImageProcessor does internally is
    replicated in _preprocess, and the model-resolution depth is resized back to
    the frame in _postprocess. See convert_depth_onnx.py for the export.
    """

    def __init__(self, onnx_path=DEFAULT_ONNX, size=_ONNX_SIZE, providers=None):
        import onnxruntime as ort  # lazy: pipeline-only users needn't have ORT
        if providers is None:
            # Use whatever this box offers (CUDA on the Jetson, CPU on a laptop).
            providers = ort.get_available_providers()
        self.size = size
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame):
        """BGR uint8 HxWx3 -> float32 (1, 3, size, size), ImageNet-normalized."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # INTER_CUBIC mirrors the processor's bicubic (resample=3) resize.
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = np.transpose(arr, (2, 0, 1))[None]  # HWC -> 1CHW
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _postprocess(self, depth_lowres, orig_h, orig_w):
        """(size, size) float depth -> (depth at frame res, PIL image 0-255)."""
        depth = cv2.resize(depth_lowres, (orig_w, orig_h),
                           interpolation=cv2.INTER_CUBIC)
        d_min, d_max = float(depth.min()), float(depth.max())
        norm = (depth - d_min) / (d_max - d_min + 1e-8)  # 0..1, high = closer
        depth_u8 = (norm * 255.0).astype(np.uint8)
        return depth, Image.fromarray(depth_u8)

    def infer(self, frame):
        """frame: BGR numpy array. Returns {'predicted_depth', 'depth'} like the
        transformers pipeline (see DepthInference.infer)."""
        h, w = frame.shape[:2]
        x = self._preprocess(frame)
        (pred,) = self.session.run(None, {self.input_name: x})
        predicted_depth, depth_img = self._postprocess(pred[0], h, w)
        return {"predicted_depth": predicted_depth, "depth": depth_img}
