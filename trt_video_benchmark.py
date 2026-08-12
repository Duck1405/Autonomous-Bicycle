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

from jetson_tools.postprocess import (LaneHysteresis, depth_colorize, draw_boxes,  # noqa: E402
                         draw_lanes, laneatt_decode, yolo_decode)
from jetson_tools.preprocess import pre_depth, pre_laneatt, pre_yolo_meta  # noqa: E402
from jetson_tools.trt_runner import CudaRT, TrtEngine  # noqa: E402

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
    parser.add_argument("--models", default=",".join(LABELS),
                        help=f"comma-separated subset of {','.join(LABELS)}")
    parser.add_argument("--laneatt-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.engine"))
    parser.add_argument("--yolo-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.engine"))
    parser.add_argument("--depth-engine", type=Path,
                        default=Path("LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine"))
    parser.add_argument("--render", type=Path, default=None,
                        help="write an annotated video here (decode + draw are "
                             "timed separately as render_ms)")
    parser.add_argument("--codec", default="mp4v",
                        help="fourcc for --render; avc1 has no encoder on the "
                             "Jetson, MJPG with a .avi path is the fallback")
    parser.add_argument("--no-hysteresis", action="store_true",
                        help="draw every lane the model NMS returns instead of "
                             "the two-threshold filtered set")
    parser.add_argument("--json", type=Path, default="Benchmark",
                        help="also write the results here as JSON")
    return parser.parse_args()


def fourcc(code):
    """cv2.VideoWriter_fourcc is gone in OpenCV 5 (the Jetson runs 5.0.0)."""
    fn = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return fn(*code)


def compose(frame, raw, hysteresis):
    """Engine outputs for one frame -> the image to write.

    Depth replaces the canvas because it is a full-frame image; lanes and boxes
    are overlays and go on top of whatever the canvas ended up being.
    """
    h, w = frame.shape[:2]
    canvas = depth_colorize(raw["depth"][0], w, h) if "depth" in raw else frame
    if "laneatt" in raw:
        lanes = laneatt_decode(raw["laneatt"][0])
        draw_lanes(canvas, hysteresis(lanes) if hysteresis else lanes)
    if "yolo" in raw:
        out, (r, dx, dy) = raw["yolo"]
        draw_boxes(canvas, yolo_decode(out, r, dx, dy))
    return canvas


def main():
    args = parse_args()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in LABELS]
    if unknown:
        raise SystemExit(f"unknown --models entries {unknown}; pick from {list(LABELS)}")

    cuda = CudaRT()
    free, total = cuda.mem_info()
    cc = cuda.compute_capability()
    arch = f"sm{cc[0]}{cc[1]}" if cc else "sm?"
    print(f"tensorrt {trt.__version__}, {arch}, "
          f"{free / 2**30:.2f} GiB GPU free of {total / 2**30:.2f} GiB")

    specs = [("laneatt", args.laneatt_engine, _pre_laneatt),
             ("yolo", args.yolo_engine, _pre_yolo),]
            #  ("depth", args.depth_engine, _pre_depth)]
    models = []
    for label, path, pre in specs:
        if label not in wanted:
            continue
        if not path.exists():
            print(f"SKIPPING {label}: {path} not found")
            continue
        eng = TrtEngine(path, cuda=cuda)
        print(f"{label}: {path}")
        for line in eng.describe():
            print(f"    {line}")
        # compose() assumes one output per engine; all three export that way.
        if args.render and len(eng.outputs) != 1:
            raise SystemExit(f"{label}: --render expects a single output tensor, "
                             f"got {list(eng.outputs)}")
        models.append((label, eng, pre))
    if not models:
        raise SystemExit("no engines found")
    else:
        print(f"Models: {models}")

    cap = cv2.VideoCapture(str(args.video))
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    for _ in range(args.warmup):
        for _, eng, pre in models:
            eng.run(pre(first)[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer, hysteresis = None, None
    if args.render:
        h, w = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        args.render.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.render), fourcc(args.codec), fps, (w, h))
        # OpenCV reports a missing encoder by returning False here, not by raising.
        if not writer.isOpened():
            raise SystemExit(f"VideoWriter could not open {args.render} with codec "
                             f"'{args.codec}' — try --codec MJPG with a .avi path")
        drawing_lanes = any(label == "laneatt" for label, _, _ in models)
        if drawing_lanes and not args.no_hysteresis:
            hysteresis = LaneHysteresis()
        print(f"rendering {w}x{h} @ {fps:.2f} fps [{args.codec}] -> {args.render}"
              f"{' (no hysteresis)' if drawing_lanes and not hysteresis else ''}")

    t_read = 0.0
    t_render = 0.0
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
        raw = {}
        for label, eng, pre in models:
            t0 = time.perf_counter()
            arr, meta = pre(frame)
            t_pre[label] += time.perf_counter() - t0
            t0 = time.perf_counter()
            eng.run(arr)
            t_eng[label] += time.perf_counter() - t0
            if writer is not None:
                # The D2H belongs to rendering, not to the engine forward, so
                # engine_ms stays comparable to the non-render runs. Outputs are
                # reused host buffers; they are consumed before the next frame.
                t0 = time.perf_counter()
                raw[label] = (next(iter(eng.fetch().values())), meta)
                t_render += time.perf_counter() - t0
        if writer is not None:
            t0 = time.perf_counter()
            writer.write(compose(frame, raw, hysteresis))
            t_render += time.perf_counter() - t0
        done += 1
        if done % 10 == 0 or done == args.frames:
            print(f"\r{done}/{args.frames} frames", end="", flush=True)
        
    t_wall = time.perf_counter() - t_wall
    cap.release()
    if writer is not None:
        writer.release()

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
    if writer is not None:
        print(f"render (D2H + decode + draw + write): {ms(t_render):6.1f} ms/frame")
    print(f"pipeline wall (read + preprocess + engines"
          f"{' + render' if writer is not None else ''}, sequential): "
          f"{ms(t_wall):.1f} ms/frame -> {done / t_wall:.2f} FPS")


    json_output = {
        "frames": done,
        "video": str(args.video),
        "tensorrt": trt.__version__,
        "compute_capability": f"sm{cc[0]}{cc[1]}" if cc else None,
        "render": str(args.render) if args.render else None,
        "read_ms": ms(t_read),
        "models": {label: {"preprocess_ms": ms(t_pre[label]),
                           "engine_ms": ms(t_eng[label])}
                   for label, _, _ in models},
        "engines_only_ms": ms(total_eng),
        "engines_only_fps": done / total_eng,
        "render_ms": ms(t_render),
        "pipeline_ms": ms(t_wall),
        "pipeline_fps": done / t_wall,
    }
    
    print(json.dumps(json_output, indent=2) + "\n")

    if args.json:
        print("Json is enabled")
        folder = Path(args.json)
        files = [f.name for f in folder.iterdir() if f.is_file()]
        total = len(files)
        name = f"Benchmark_{total + 1}.json"
        file_json = folder / name
        print(f"Writing to {args.json}")
        
        with open(file_json, "w") as f:
            json.dump(json_output, f, indent=2)
    
    # if args.json:
        # args.json.write_text(json.dumps({
        #     "frames": done,
        #     "video": str(args.video),
        #     "tensorrt": trt.__version__,
        #     "compute_capability": f"sm{cc[0]}{cc[1]}" if cc else None,
        #     "render": str(args.render) if args.render else None,
        #     "read_ms": ms(t_read),
        #     "models": {label: {"preprocess_ms": ms(t_pre[label]),
        #                        "engine_ms": ms(t_eng[label])}
        #                for label, _, _ in models},
        #     "engines_only_ms": ms(total_eng),
        #     "engines_only_fps": done / total_eng,
        #     "render_ms": ms(t_render),
        #     "pipeline_ms": ms(t_wall),
        #     "pipeline_fps": done / t_wall,
        # }, indent=2) + "\n")
        # print(f"wrote {args.json}")

    for _, eng, _ in models:
        eng.close()


if __name__ == "__main__":
    main()
