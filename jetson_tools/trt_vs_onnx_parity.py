"""Compare a TensorRT engine's output against onnxruntime-CPU on the same frame.

    python jetson_tools/trt_vs_onnx_parity.py --model depth \
        --onnx LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.onnx \
        --engine LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine

onnxruntime on CPU runs in FP32 and is the reference. An FP16 engine will not
match bit-for-bit — the question is whether it is close enough to trust, and
whether anything overflowed.

This matters most for the depth model: NVIDIA's forum has an unresolved report
of Depth-Anything-V2 FP16 on Orin Nano / TensorRT 10.3 producing badly degraded
output, the usual cause being LayerNorm overflowing in FP16 in ViT backbones. A
fast engine that returns garbage is worse than no engine, so run this before
believing any FPS number.

Correlation is the headline metric for depth: the model is scale-ambiguous, so a
uniform scale shift matters far less than a change in the *shape* of the depth
map.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import PREPROCESSORS  # noqa: E402
from trt_runner import TrtEngine  # noqa: E402


def load_frame(video, index):
    cap = cv2.VideoCapture(str(video))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {index} of {video}")
    return frame


def compare(name, ref, got):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    got = np.asarray(got, dtype=np.float64).ravel()
    if ref.shape != got.shape:
        print(f"  {name}: SHAPE MISMATCH onnx {ref.shape} vs trt {got.shape}")
        return False

    bad_ref = ~np.isfinite(ref)
    bad_got = ~np.isfinite(got)
    if bad_got.any():
        print(f"  {name}: {bad_got.sum()} NaN/Inf in the ENGINE output "
              f"(onnx has {bad_ref.sum()}) -> FP16 overflow")
        return False

    finite = np.isfinite(ref) & np.isfinite(got)
    ref, got = ref[finite], got[finite]
    abs_err = np.abs(ref - got)
    scale = np.maximum(np.abs(ref), 1e-6)
    rel_err = abs_err / scale
    corr = np.corrcoef(ref, got)[0, 1] if ref.size > 1 and ref.std() > 0 else float("nan")

    print(f"  {name}: max_abs {abs_err.max():.4g}  mean_abs {abs_err.mean():.4g}  "
          f"p99_rel {np.percentile(rel_err, 99):.4g}  corr {corr:.6f}")
    print(f"      onnx range [{ref.min():.4g}, {ref.max():.4g}]   "
          f"trt range [{got.min():.4g}, {got.max():.4g}]")
    return corr > 0.999 or abs_err.max() < 1e-3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PREPROCESSORS),
                    help="which preprocessing to apply")
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--video", type=Path, default=Path("LaneATT/video_input/IMG_6540.MOV"))
    ap.add_argument("--frame", type=int, default=0, help="frame index to test")
    args = ap.parse_args()

    import onnxruntime as ort

    frame = load_frame(args.video, args.frame)
    blob = PREPROCESSORS[args.model](frame)
    print(f"{args.model}: frame {args.frame} of {args.video.name} -> input {blob.shape}")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    ref_names = [o.name for o in sess.get_outputs()]
    ref_outs = sess.run(None, {in_name: blob})
    print(f"onnxruntime CPU (FP32 reference): {len(ref_outs)} output(s)")

    eng = TrtEngine(args.engine)
    trt_outs = eng.infer(blob)
    print(f"tensorrt engine: {list(trt_outs)}")

    print("\nparity:")
    all_ok = True
    for i, ref_name in enumerate(ref_names):
        got = trt_outs.get(ref_name)
        if got is None:
            # Names can differ between exporter and engine; fall back to order.
            got = list(trt_outs.values())[i]
        all_ok &= compare(ref_name, ref_outs[i], got)
    eng.close()

    print(f"\n{'PASS' if all_ok else 'REVIEW NEEDED'} — "
          f"{'engine tracks the ONNX reference' if all_ok else 'see the deltas above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
