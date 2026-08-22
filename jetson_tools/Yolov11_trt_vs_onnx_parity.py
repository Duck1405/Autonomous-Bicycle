"""
python Yolov11_trt_vs_onnx_parity.py   --model yolo   --onnx /home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx   --engine /home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/YoloN/YoloN_fb16.engine   --video /home/mlc/aman/Autonomous-Bicycle/Videos2/IMG_6540.MOV   --max-frames 100
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import PREPROCESSORS  # noqa: E402
from trt_runner import TrtEngine  # noqa: E402

def frame_metrics(ref, got):
    """Return numerical parity metrics for one TensorRT/ONNX output pair."""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    got = np.asarray(got, dtype=np.float64).ravel()

    if ref.shape != got.shape:
        return {
            "valid": False,
            "reason": f"shape mismatch: ONNX {ref.shape}, TRT {got.shape}",
        }

    bad_ref = ~np.isfinite(ref)
    bad_got = ~np.isfinite(got)

    if bad_got.any():
        return {
            "valid": False,
            "reason": (
                f"TensorRT contains {bad_got.sum()} NaN/Inf values "
                f"(ONNX contains {bad_ref.sum()})"
            ),
        }

    finite = np.isfinite(ref) & np.isfinite(got)
    ref = ref[finite]
    got = got[finite]

    if ref.size == 0:
        return {"valid": False, "reason": "no finite values to compare"}

    abs_err = np.abs(ref - got)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-6)

    if ref.size > 1 and ref.std() > 0 and got.std() > 0:
        corr = float(np.corrcoef(ref, got)[0, 1])
    else:
        corr = float("nan")

    return {
        "valid": True,
        "count": ref.size,
        "corr": corr,
        "sum_abs_err": float(abs_err.sum()),
        "max_abs_err": float(abs_err.max()),
        "p99_rel_err": float(np.percentile(rel_err, 99)),
        "onnx_min": float(ref.min()),
        "onnx_max": float(ref.max()),
        "trt_min": float(got.min()),
        "trt_max": float(got.max()),
    }


def update_summary(summary, metrics):
    """Accumulate metrics without storing entire model outputs."""
    if not metrics["valid"]:
        summary["failed_frames"] += 1
        return

    summary["valid_frames"] += 1
    summary["total_values"] += metrics["count"]
    summary["total_abs_err"] += metrics["sum_abs_err"]
    summary["max_abs_err"] = max(summary["max_abs_err"], metrics["max_abs_err"])
    summary["p99_rel_errors"].append(metrics["p99_rel_err"])

    if np.isfinite(metrics["corr"]):
        summary["correlations"].append(metrics["corr"])


def print_summary(name, summary):
    print(f"\n{name} overall parity:")

    if summary["valid_frames"] == 0:
        print("  No valid frames.")
        return False

    correlations = np.asarray(summary["correlations"], dtype=np.float64)

    print(f"  valid frames:       {summary['valid_frames']}")
    print(f"  failed frames:      {summary['failed_frames']}")
    print(
        f"  global mean abs:    "
        f"{summary['total_abs_err'] / summary['total_values']:.6g}"
    )
    print(f"  worst max abs:      {summary['max_abs_err']:.6g}")

    if correlations.size:
        print(f"  mean correlation:   {correlations.mean():.8f}")
        print(f"  median correlation: {np.median(correlations):.8f}")
        print(f"  p5 correlation:     {np.percentile(correlations, 5):.8f}")
        print(f"  worst correlation:  {correlations.min():.8f}")

    print(
        f"  median frame p99 relative error: "
        f"{np.median(summary['p99_rel_errors']):.6g}"
    )

    # Do not make pass/fail depend only on mean correlation.
    return (
        summary["failed_frames"] == 0
        and correlations.size > 0
        and correlations.min() > 0.999
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model",
        required=True,
        choices=sorted(PREPROCESSORS),
        help="which preprocessing to apply",
    )
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument(
        "--video",
        type=Path,
        default=Path("/home/mlc/aman/Autonomous-Bicycle/Videos2/IMG_6540.MOV"),
    )
    ap.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="first video frame to compare",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="number of frames to compare; use 0 for entire video",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="compare every Nth frame",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="TensorRT warmup iterations before measuring",
    )
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX file not found: {args.onnx}")

    if not args.engine.exists():
        raise SystemExit(f"TensorRT engine not found: {args.engine}")

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")

    print("Loading ONNX Runtime CPU reference...")
    sess = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    ref_names = [output.name for output in sess.get_outputs()]

    print("Loading TensorRT engine...")
    eng = TrtEngine(args.engine)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        eng.close()
        raise SystemExit(f"Cannot open video: {args.video}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    print(
        f"Video: {args.video.name} | total frames: {total_video_frames} | "
        f"starting at: {args.start_frame}"
    )
    print(
        f"Comparing {'all remaining frames' if args.max_frames == 0 else args.max_frames} "
        f"frame(s), stride={args.stride}"
    )

    # Warm up TensorRT using actual video frames.
    print(f"Warming up TensorRT for {args.warmup} frame(s)...")
    warmup_done = 0

    while warmup_done < args.warmup:
        ok, frame = cap.read()

        if not ok:
            break

        blob = PREPROCESSORS[args.model](frame)
        print(f"blob shape: {blob.shape}, dtype: {blob.dtype}")
        eng.infer(blob)
        warmup_done += 1

    # Restart at requested frame so warmup frames are also evaluated.
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    summaries = {
        name: {
            "valid_frames": 0,
            "failed_frames": 0,
            "total_values": 0,
            "total_abs_err": 0.0,
            "max_abs_err": 0.0,
            "correlations": [],
            "p99_rel_errors": [],
        }
        for name in ref_names
    }

    compared_frames = 0
    actual_frame_index = args.start_frame

    try:
        while args.max_frames == 0 or compared_frames < args.max_frames:
            ok, frame = cap.read()

            if not ok:
                print("\nReached end of video.")
                break

            if (actual_frame_index - args.start_frame) % args.stride != 0:
                actual_frame_index += 1
                continue

            blob = PREPROCESSORS[args.model](frame)

            # ONNX FP32 CPU reference.
            ref_outs = sess.run(None, {in_name: blob})

            # TensorRT engine output. Compare immediately because its buffers are reused.
            trt_outs = eng.infer(blob)

            frame_ok = True

            for output_index, ref_name in enumerate(ref_names):
                got = trt_outs.get(ref_name)

                # Tensor names may differ between ONNX export and TensorRT engine.
                if got is None:
                    got = list(trt_outs.values())[output_index]

                metrics = frame_metrics(ref_outs[output_index], got)
                update_summary(summaries[ref_name], metrics)

                if not metrics["valid"]:
                    frame_ok = False
                    print(
                        f"Frame {actual_frame_index}, {ref_name}: "
                        f"FAIL - {metrics['reason']}"
                    )
                elif compared_frames % 25 == 0:
                    print(
                        f"Frame {actual_frame_index}, {ref_name}: "
                        f"corr={metrics['corr']:.8f}, "
                        f"mean_abs={metrics['sum_abs_err'] / metrics['count']:.6g}, "
                        f"max_abs={metrics['max_abs_err']:.6g}, "
                        f"p99_rel={metrics['p99_rel_err']:.6g}"
                    )

            compared_frames += 1
            actual_frame_index += 1

    finally:
        cap.release()
        eng.close()

    print(f"\n{'=' * 70}")
    print(f"PARITY RESULTS: {compared_frames} compared frame(s)")
    print(f"{'=' * 70}")

    all_ok = True

    for name, summary in summaries.items():
        output_ok = print_summary(name, summary)
        all_ok &= output_ok

    print(
        f"\n{'PASS' if all_ok else 'REVIEW NEEDED'} - "
        f"{'TensorRT consistently tracks the ONNX reference.' if all_ok else 'Check failed frames and worst-case metrics.'}"
    )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())