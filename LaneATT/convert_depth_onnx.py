"""Export Depth-Anything-V2-Small (HuggingFace transformers) to ONNX.

The HF checkpoint at depth_model/ IS the full model: config.json holds the
architecture (DepthAnythingForDepthEstimation, DINOv2 backbone) and
model.safetensors holds the weights. transformers reconstructs the runnable
nn.Module from those two, so there is nothing to hand-write — we just load it
and torch.onnx.export it.

    pixel_values (1, 3, 518, 518)  ->  predicted_depth (1, 518, 518)
                                        [relative depth, larger = closer]

Static 518x518 input (a multiple of 14, the DINOv2 patch size). The model's
forward returns a DepthEstimatorOutput dataclass, which the ONNX exporter can't
emit, so a thin wrapper returns just the predicted_depth tensor (same idea as
convertonnx.py stubbing out LaneATT's in-graph NMS).

Usage (from the LaneATT folder):
    python convert_depth_onnx.py
    python convert_depth_onnx.py --model depth-anything/Depth-Anything-V2-Small-hf
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForDepthEstimation

DEFAULT_MODEL = "depth_model"  # local dir saved by lib/depth.py (falls back to hub)
DEFAULT_OUT = "depth_onnx/depth_anything_v2_small.onnx"
INPUT_SIZE = 518  # multiple of 14 (DINOv2 patch size)


class DepthWrapper(nn.Module):
    """Unwrap the DepthEstimatorOutput dataclass to a single depth tensor so
    torch.onnx.export sees a plain tensor output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values).predicted_depth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model dir or hub id (default: local depth_model/)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .onnx path")
    ap.add_argument("--size", type=int, default=INPUT_SIZE,
                    help="square input size; must be a multiple of 14 (default 518)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    args = ap.parse_args()

    if args.size % 14 != 0:
        raise SystemExit(f"--size must be a multiple of 14 (got {args.size})")

    # Export on CPU — device-agnostic, and avoids the MPS bicubic issue noted in
    # lib/depth.py. Weights + architecture both come from from_pretrained.
    model = AutoModelForDepthEstimation.from_pretrained(args.model).eval()
    wrapper = DepthWrapper(model).eval()

    x = torch.randn(1, 3, args.size, args.size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (x,), str(out_path),
        input_names=["pixel_values"], output_names=["predicted_depth"],
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Structural validity.
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))
    print("onnx.checker: OK")

    # Parity: onnxruntime vs torch on the SAME input isolates export fidelity
    # from any preprocessing differences. Expect ~1e-4 in fp32.
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = wrapper(x).cpu().numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    (ort_out,) = sess.run(None, {"pixel_values": x.numpy()})
    diff = np.abs(torch_out - ort_out).max()
    print(f"output shape: {ort_out.shape}, max |torch - onnxruntime| = {diff:.2e}")
    if diff > 1e-3:
        print("WARNING: torch/onnx mismatch larger than expected (>1e-3)")


if __name__ == "__main__":
    main()
