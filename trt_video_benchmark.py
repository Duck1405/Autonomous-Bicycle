"""Per-frame TensorRT video benchmark for the three Pathfinder engines.

Mirrors VideoInference's sequential per-frame loop — read a frame, preprocess,
run LaneATT + YOLO + depth engines one after the other — and reports ms/frame
per model plus the resulting pipeline FPS, comparable to run.log's numbers for
the torch (.pt) path. Engine times include the H2D input copy, like the torch
path's timings, and nothing else: decoding and drawing land in `render_ms`, so
`engine_ms` means the same thing whether or not --render is on.

Buffers go through jetson_tools/trt_runner.py (ctypes + libcudart), so torch is
not required. An earlier revision used torch CUDA tensors and produced the A100
figure of 71.19 FPS; both do the same cudaMemcpy underneath, so the numbers stay
comparable.

With --render the engine outputs are also decoded and drawn (see
jetson_tools/postprocess.py) and written to an annotated video. The composed
frame is the depth colormap when depth is among --models, otherwise the native
frame; lanes and boxes are drawn on top of whichever it is.
VideoInference.video_eval owns reading and saving; get_frame owns ego-lane
selection and drawing. Output shows original and annotated frames side by side.

Needs: tensorrt (10.x), numpy, cv2. Run from the Autonomous-Bicycle repository root. Cluster env: LaneNetCuda_12_6. Jetson env: LaneNet310.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from types import SimpleNamespace
import tensorrt as trt

from jetson_tools.postprocess import (LaneHysteresis, depth_colorize, draw_boxes,  # noqa: E402
                         laneatt_decode, yolo_decode)
from jetson_tools.preprocess import pre_depth, pre_laneatt, pre_yolo_meta  # noqa: E402
from jetson_tools.trt_runner import CudaRT, TrtEngine  # noqa: E402

# Use the same VideoInference as LaneATT/inference.py, regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent / "LaneATT"))
from lib.video import VideoInference

LABELS = ("laneatt", "yolo", "depth")


# The preprocessors return (tensor, meta); only YOLO has meta — the letterbox
# (scale, dx, dy) its boxes have to be un-warped by. One shape keeps the timed
# loop free of per-model branching.
def _pre_laneatt(frame):
    return pre_laneatt(frame), None


def _pre_yolo(frame):
    arr, r, dx, dy = pre_yolo_meta(frame)
    return arr, (r, dx, dy)


def _pre_depth(frame):
    return pre_depth(frame), None


def str2bool(value):
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean string, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential per-frame TensorRT benchmark of the LaneATT / "
                    "YOLO / depth engines over a real video, optionally "
                    "rendering an annotated output video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=Path,
                        default=Path("video_input/IMG_6540.MOV"))
    parser.add_argument("--frames", type=int, default=500,
                        help="frames to time (after warmup)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="untimed warmup iterations on the first frame")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="frame index to start timing from")
    parser.add_argument("--models", default=",".join(LABELS),
                        help=f"comma-separated subset of {','.join(LABELS)}")
    parser.add_argument("--laneatt-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine"))
    parser.add_argument("--yolo-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/YoloN/YoloN_fb16.engine"))
    parser.add_argument("--laneatt-on", type=str2bool, default=True)
    parser.add_argument("--yolo-on", type=str2bool, default=True)
    parser.add_argument("--depth-on", type=str2bool, default=True)
    
    parser.add_argument("--depth-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine"))
    parser.add_argument("--render", type=Path, default="LaneATT/video_output_5/render.mp4",
                        help="write an annotated video here (decode + draw are "
                             "timed separately as render_ms)")
    parser.add_argument("--no-render", action="store_true",
                        help="skip writing the annotated video entirely (no D2H "
                             "copy, decode, draw, or encode) for an engines-only "
                             "benchmark; --render is ignored when this is set")
    parser.add_argument("--codec", default="mp4v",
                        help="fourcc for --render; avc1 has no encoder on the "
                             "Jetson, MJPG with a .avi path is the fallback")
    parser.add_argument("--no-hysteresis", action="store_true",
                        help="skip two-threshold hysteresis before selecting the ego lanes")
    parser.add_argument("--json", type=Path, default="Benchmark",
                        help="also write the results here as JSON")
    return parser.parse_args()


class TensorRTYoloDrawing:
    """Adapt already-decoded TensorRT boxes to VideoInference's drawing interface."""

    def draw(self, frame, results):
        draw_boxes(frame, results)
        return frame


def main():
    args = parse_args()
    if args.frames <= 0 or args.start_frame < 0 or args.warmup < 0:
        raise SystemExit("--frames must be positive; --start-frame and --warmup must be non-negative")
    if args.no_render:
        args.render = None
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(wanted) - set(LABELS)
    if unknown:
        raise SystemExit(f"unknown --models entries {sorted(unknown)}")
    cuda = CudaRT()
    free, total = cuda.mem_info()
    cc = cuda.compute_capability()
    print(f"tensorrt {trt.__version__}, {free / 2**30:.2f} GiB GPU free of {total / 2**30:.2f} GiB")
    specs = [("laneatt", args.laneatt_engine, _pre_laneatt, args.laneatt_on),
             ("yolo", args.yolo_engine, _pre_yolo, args.yolo_on),
             ("depth", args.depth_engine, _pre_depth, args.depth_on)]
    models = []
    try:
        for label, path, pre, enabled in specs:
            if not enabled or label not in wanted:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"{label} engine not found: {path}")
            eng = TrtEngine(path, cuda=cuda)
            models.append((label, eng, pre))
            print(f"{label}: {path}")
            for line in eng.describe():
                print(f"    {line}")
            if args.render and len(eng.outputs) != 1:
                raise ValueError(f"{label}: rendering expects one engine output")
        if not models:
            raise ValueError("No models enabled")
        pipeline = VideoInference(video_path=str(args.video), frame_limit=args.frames,
                                  device="TensorRT", model_path=args.laneatt_engine,
                                  initialize_models=False)
        if any(label == "yolo" for label, _, _ in models):
            pipeline.yolo = TensorRTYoloDrawing()
        hysteresis = None if args.no_hysteresis else LaneHysteresis()
        t_pre = {label: 0.0 for label, _, _ in models}
        t_eng = dict(t_pre)
        render_seconds = 0.0

        def warmup(first):
            for _ in range(args.warmup):
                for _, eng, pre in models:
                    eng.run(pre(first)[0])

        def process(frame):
            nonlocal render_seconds
            raw = {}
            for label, eng, pre in models:
                t0 = time.perf_counter()
                arr, meta = pre(frame)
                t_pre[label] += time.perf_counter() - t0
                t0 = time.perf_counter()
                eng.run(arr)
                t_eng[label] += time.perf_counter() - t0
                if args.render:
                    t0 = time.perf_counter()
                    # Shared runner buffers are consumed before the next inference.
                    raw[label] = (next(iter(eng.fetch().values())), meta)
                    render_seconds += time.perf_counter() - t0
            if not args.render:
                return None
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            canvas = depth_colorize(raw["depth"][0], w, h) if "depth" in raw else frame
            evaluation = []
            if "laneatt" in raw:
                lanes = laneatt_decode(raw["laneatt"][0], conf_threshold=pipeline.keep_threshold,
                                       nms_thres=pipeline.nms_thres, nms_topk=pipeline.nms_topk)
                if hysteresis is not None:
                    lanes = hysteresis(lanes)
                evaluation = [SimpleNamespace(points=lane["points"], metadata={"conf": lane["conf"]})
                              for lane in lanes]
            boxes = []
            if "yolo" in raw:
                out, (r, dx, dy) = raw["yolo"]
                boxes = yolo_decode(out, r, dx, dy)
            canvas = pipeline.get_frame(canvas, evaluation=evaluation, yolo_results=boxes)
            render_seconds += time.perf_counter() - t0
            return canvas

        stats = pipeline.video_eval(start_frame=args.start_frame, output_path=args.render,
                                    codec=args.codec, frame_processor=process, warmup=warmup,
                                    render=args.render is not None)
        done = stats["frames"]
        elapsed = stats["elapsed_seconds"]
        total_eng = sum(t_eng.values())
        def ms(seconds):
            return 1000.0 * seconds / done
        def fps(seconds):
            return done / seconds if seconds > 0 else 0.0
        render_seconds += stats["write_seconds"]
        result = {
            "frames": done, "start_frame": args.start_frame, "video": str(args.video),
            "tensorrt": trt.__version__,
            "compute_capability": f"sm{cc[0]}{cc[1]}" if cc else None,
            "render": stats["output"], "read_ms": ms(stats["read_seconds"]),
            "models": {label: {"preprocess_ms": ms(t_pre[label]), "engine_ms": ms(t_eng[label])}
                       for label, _, _ in models},
            "engines_only_ms": ms(total_eng), "engines_only_fps": fps(total_eng),
            "render_ms": ms(render_seconds), "pipeline_ms": ms(elapsed), "pipeline_fps": fps(elapsed),
        }
        print(json.dumps(result, indent=2))
        if args.json:
            folder = args.json.parent if args.json.is_file() else args.json
            folder.mkdir(parents=True, exist_ok=True)
            n = 1
            while (folder / f"Benchmark_{n}.json").exists():
                n += 1
            path = folder / f"Benchmark_{n}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Wrote {path}")
    finally:
        for _, eng, _ in models:
            eng.close()


if __name__ == "__main__":
    main()
