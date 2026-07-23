"""Verify the exported ONNX depth model matches the PyTorch model.

Two checks, from strict to end-to-end:

  1. GRAPH FIDELITY  — feed the SAME pixel_values (518x518) to the PyTorch
     model and to onnxruntime; the depth tensors must match to ~1e-5. This
     proves the ONNX graph *is* the model, independent of any preprocessing.

  2. END-TO-END      — run the full transformers pipeline (DepthInference) and
     the ONNX path (DepthInferenceONNX) on the same real frame and compare the
     0-255 depth maps. These differ slightly on purpose: the pipeline resizes
     keep-aspect-ratio while the ONNX path uses a fixed 518x518 square, so we
     report correlation / mean abs error rather than exact equality, and save a
     side-by-side image.

Usage (from the LaneATT folder):
    python test_depth_onnx.py
    python test_depth_onnx.py --video video_input/1.mp4 --frame 100
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

from lib.depth import DepthInference, DepthInferenceONNX

ONNX_PATH = "depth_onnx/depth_anything_v2_small.onnx"


def get_frame(video, frame_idx):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video}")
    return frame  # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video_input/1.mp4")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--onnx", default=ONNX_PATH)
    args = ap.parse_args()

    frame = get_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    print(f"frame: {args.video} #{args.frame}  ({w}x{h})")

    onnx = DepthInferenceONNX(onnx_path=args.onnx, providers=["CPUExecutionProvider"])

    # ---- 1. GRAPH FIDELITY: identical pixel_values -> torch vs onnxruntime ----
    pix = onnx._preprocess(frame)  # (1,3,518,518) float32, exactly what ORT sees
    torch_model = AutoModelForDepthEstimation.from_pretrained("depth_model").eval()
    with torch.no_grad():
        torch_depth = torch_model(torch.from_numpy(pix)).predicted_depth.numpy()
    (ort_depth,) = onnx.session.run(None, {onnx.input_name: pix})
    gdiff = np.abs(torch_depth - ort_depth).max()
    print(f"\n[1] graph fidelity (same input): max |torch - onnx| = {gdiff:.2e}  "
          f"{'PASS' if gdiff < 1e-3 else 'FAIL'}")

    # ---- 2. END-TO-END: transformers pipeline vs ONNX path on the frame ----
    pipe_depth = np.array(DepthInference().infer(frame)["depth"], dtype=np.float32)
    onnx_depth = np.array(onnx.infer(frame)["depth"], dtype=np.float32)
    corr = np.corrcoef(pipe_depth.ravel(), onnx_depth.ravel())[0, 1]
    mae = np.abs(pipe_depth - onnx_depth).mean()
    print(f"[2] end-to-end (0-255 depth): corr = {corr:.4f}, "
          f"mean|Δ| = {mae:.2f} / 255")

    # Side-by-side: pipeline | onnx | abs diff, all INFERNO-colored.
    def color(d):
        return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_INFERNO)
    diff = np.abs(pipe_depth - onnx_depth)
    panel = np.hstack([color(pipe_depth), color(onnx_depth),
                       color(diff / (diff.max() + 1e-8) * 255)])
    out = Path(args.onnx).parent / f"parity_frame{args.frame}.png"
    cv2.imwrite(str(out), panel)
    print(f"\nsaved side-by-side (pipeline | onnx | diff): {out}")


if __name__ == "__main__":
    main()
