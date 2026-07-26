"""Per-frame TensorRT video benchmark for the three Pathfinder engines.

Mirrors VideoInference's sequential per-frame loop — read a frame, preprocess,
run LaneATT + YOLO + depth engines one after the other — and reports ms/frame
per model plus the resulting pipeline FPS, comparable to run.log's numbers for
the torch (.pt) path. Engine times include the H2D input copy, like the torch
path's timings. LaneATT proposal decoding / drawing are NOT included: this
measures the network forward only.

Buffers go through jetson_tools/trt_runner.py (ctypes + libcudart), so torch is
not required. An earlier revision used torch CUDA tensors and produced the A100
figure of 71.19 FPS; both do the same cudaMemcpy underneath, so the numbers stay
comparable.

Needs: tensorrt (10.x), numpy, cv2. Run from the LaneATT directory so the
relative engine/video defaults resolve (cluster and Jetson share the onnxmodels/
layout). Cluster env: LaneNetCuda_12_6. Jetson env: LaneNet310.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jetson_tools"))
from preprocess import pre_depth, pre_laneatt, pre_yolo  # noqa: E402
from trt_runner import CudaRT, TrtEngine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential per-frame TensorRT benchmark of the LaneATT / "
                    "YOLO / depth engines over a real video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=Path,
                        default=Path("video_input/IMG_6540.MOV"))
    parser.add_argument("--frames", type=int, default=500,
                        help="frames to time (after warmup)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="untimed warmup iterations on the first frame")
    parser.add_argument("--laneatt-engine", type=Path,
                        default=Path("onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.engine"))
    parser.add_argument("--yolo-engine", type=Path,
                        default=Path("onnxmodels/YoloN/yolo11n_coco4_nms.engine"))
    parser.add_argument("--depth-engine", type=Path,
                        default=Path("onnxmodels/depth_onnx/depth_anything_v2_small.engine"))
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the results here as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    cuda = CudaRT()
    free, total = cuda.mem_info()
    cc = cuda.compute_capability()
    arch = f"sm{cc[0]}{cc[1]}" if cc else "sm?"
    print(f"tensorrt {trt.__version__}, {arch}, "
          f"{free / 2**30:.2f} GiB GPU free of {total / 2**30:.2f} GiB")

    specs = [("laneatt", args.laneatt_engine, pre_laneatt),
             ("yolo", args.yolo_engine, pre_yolo),
             ("depth", args.depth_engine, pre_depth)]
    models = []
    for label, path, pre in specs:
        if not path.exists():
            print(f"SKIPPING {label}: {path} not found")
            continue
        eng = TrtEngine(path, cuda=cuda)
        print(f"{label}: {path}")
        for line in eng.describe():
            print(f"    {line}")
        models.append((label, eng, pre))
    if not models:
        raise SystemExit("no engines found")

    cap = cv2.VideoCapture(str(args.video))
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    for _ in range(args.warmup):
        for _, eng, pre in models:
            eng.run(pre(first))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_read = 0.0
    t_pre = {label: 0.0 for label, _, _ in models}
    t_eng = {label: 0.0 for label, _, _ in models}
    done = 0
    t_wall = time.perf_counter()
    while done < args.frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_read += time.perf_counter() - t0
        for label, eng, pre in models:
            t0 = time.perf_counter()
            arr = pre(frame)
            t_pre[label] += time.perf_counter() - t0
            t0 = time.perf_counter()
            eng.run(arr)
            t_eng[label] += time.perf_counter() - t0
        done += 1
    t_wall = time.perf_counter() - t_wall
    cap.release()

    def ms(s):
        return 1000.0 * s / max(done, 1)

    total_eng = sum(t_eng.values())
    total_pre = sum(t_pre.values())
    print(f"\n=== TensorRT sequential per-frame benchmark: {done} frames of {args.video} ===")
    print(f"video read: {ms(t_read):.1f} ms/frame")
    for label, _, _ in models:
        print(f"{label:8s} preprocess {ms(t_pre[label]):6.1f} ms/frame + "
              f"engine {ms(t_eng[label]):6.1f} ms/frame")
    print(f"engines only:    {ms(total_eng):6.1f} ms/frame -> {done / total_eng:.1f} FPS")
    print(f"preprocess only: {ms(total_pre):6.1f} ms/frame")
    print(f"pipeline wall (read + preprocess + engines, sequential): "
          f"{ms(t_wall):.1f} ms/frame -> {done / t_wall:.2f} FPS")

    if args.json:
        args.json.write_text(json.dumps({
            "frames": done,
            "video": str(args.video),
            "tensorrt": trt.__version__,
            "compute_capability": f"sm{cc[0]}{cc[1]}" if cc else None,
            "read_ms": ms(t_read),
            "models": {label: {"preprocess_ms": ms(t_pre[label]),
                               "engine_ms": ms(t_eng[label])}
                       for label, _, _ in models},
            "engines_only_ms": ms(total_eng),
            "engines_only_fps": done / total_eng,
            "pipeline_ms": ms(t_wall),
            "pipeline_fps": done / t_wall,
        }, indent=2) + "\n")
        print(f"wrote {args.json}")

    for _, eng, _ in models:
        eng.close()


if __name__ == "__main__":
    main()
