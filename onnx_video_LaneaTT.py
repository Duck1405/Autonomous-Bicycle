"""Per-frame ONNX Runtime video benchmark for the LaneATT model.

Mirrors trt_video_benchmark.py's sequential per-frame loop and prints the same
metrics, so the ONNX Runtime (CUDA) numbers can be put straight next to the
TensorRT ones. Preprocessing comes from jetson_tools/preprocess.py — the same
code the TensorRT benchmark calls — so the runtime is the only thing that
differs between the two measurements.

`engine_ms` is the session.run() call only; ORT's run() is synchronous, so it
already includes the H2D copy and the device sync, like the TensorRT path's
engine timing.

Needs: onnxruntime-gpu, cv2. Jetson env: LaneNet310Cuda.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import onnxruntime as ort

from jetson_tools.preprocess import pre_laneatt

LABEL = "laneatt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential per-frame ONNX Runtime benchmark of the LaneATT "
                    "model over a real video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=Path,
                        default=Path("Videos2/IMG_6893_30fps.mp4"))
    parser.add_argument("--frames", type=int, default=500,
                        help="frames to time (after warmup)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="untimed warmup iterations on the first frame")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="frame index to start timing from")
    parser.add_argument("--onnx", type=Path,
                        default=Path("LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx"))
    parser.add_argument("--json", type=Path, default="Benchmark",
                        help="also write the results here as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"{args.onnx} not found")

    session = ort.InferenceSession(str(args.onnx),
                                   providers=['CUDAExecutionProvider',
                                              'CPUExecutionProvider'])
    # get_providers() is what ORT actually bound. Asking for CUDA is not the same
    # as getting it: the CPU-only wheel warns once and falls back silently.
    active = session.get_providers()
    print(f"onnxruntime {ort.__version__}, providers {active}")
    if 'CUDAExecutionProvider' not in active:
        print("  -> running on CPU (needs the Jetson onnxruntime-gpu wheel for CUDA)")
        sys.exit(1)

    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    input_name, output_name = inp.name, out.name
    print(f"{LABEL}: {args.onnx}")
    print(f"    in  {input_name} {inp.shape} {inp.type}")
    print(f"    out {output_name} {out.shape} {out.type}")

    print(f"Opening video {args.video}")

    if not args.video.exists():
        raise SystemExit(f"{args.video} not found")

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    for _ in range(args.warmup):
        session.run([output_name], {input_name: pre_laneatt(first)})
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    t_read = 0.0
    t_pre = 0.0
    t_eng = 0.0
    done = 0
    t_wall = time.perf_counter()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {args.video} ({total_frames} frames), "
          f"timing {args.frames} frames from frame {args.start_frame} "
          f"after {args.warmup} warmup")

    available = total_frames - args.start_frame
    if args.frames > available:
        print(f"WARNING: requested {args.frames} frames, but video only has {available}. "
              f"Will process {available} frames instead.")
        args.frames = available

    while done < args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_read += time.perf_counter() - t0
        t0 = time.perf_counter()
        arr = pre_laneatt(frame)
        t_pre += time.perf_counter() - t0
        t0 = time.perf_counter()
        session.run([output_name], {input_name: arr})
        t_eng += time.perf_counter() - t0
        done += 1
        if done % 10 == 0 or done == args.frames:
            print(f"\r{done}/{args.frames} frames", end="", flush=True)

    t_wall = time.perf_counter() - t_wall
    cap.release()

    def ms(s):
        return 1000.0 * s / max(done, 1)

    print(f"\n=== ONNX Runtime sequential per-frame benchmark: {done} frames of {args.video} ===")
    print(f"video read: {ms(t_read):.1f} ms/frame")
    print(f"{LABEL:8s} preprocess {ms(t_pre):6.1f} ms/frame + "
          f"engine {ms(t_eng):6.1f} ms/frame")
    print(f"engines only:    {ms(t_eng):6.1f} ms/frame -> {done / t_eng:.1f} FPS")
    print(f"preprocess only: {ms(t_pre):6.1f} ms/frame")
    print(f"pipeline wall (read + preprocess + engines, sequential): "
          f"{ms(t_wall):.1f} ms/frame -> {done / t_wall:.2f} FPS")

    json_output = {
        "frames": done,
        "video": str(args.video),
        "runtime": "onnxruntime",
        "onnxruntime": ort.__version__,
        "providers": active,
        "model": str(args.onnx),
        "start_frame": args.start_frame,
        "warmup": args.warmup,
        "read_ms": ms(t_read),
        "models": {LABEL: {"preprocess_ms": ms(t_pre),
                           "engine_ms": ms(t_eng)}},
        "engines_only_ms": ms(t_eng),
        "engines_only_fps": done / t_eng,
        "pipeline_ms": ms(t_wall),
        "pipeline_fps": done / t_wall,
    }

    print(json.dumps(json_output, indent=2) + "\n")

    if args.json:
        print("Json is enabled")
        folder = Path(args.json)
        if folder.is_file():
            folder = folder.parent
        files = [f.name for f in folder.iterdir() if f.is_file()]
        total = len(files)
        name = f"Benchmark_{total + 1}.json"
        file_json = folder / name
        print(f"Writing to {args.json}")

        with open(file_json, "w") as f:
            json.dump(json_output, f, indent=2)


if __name__ == "__main__":
    main()
