"""Run models/yolo11*_coco4_nms.onnx with onnxruntime on a video or camera.

Jetson Orin Nano Super setup (JetPack 6): the PyPI `onnxruntime-gpu` wheel is
x86-only — install NVIDIA's Jetson build instead:
    pip3 install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
(For other JetPack versions grab the matching wheel from elinux.org/Jetson_Zoo.)

Usage:
    python3 jetson_infer_onnx.py --model models/yolo11n_coco4_nms.onnx \
        --source video.mp4 --out annotated.mp4
    python3 jetson_infer_onnx.py --model ... --source 0        # CSI/USB camera

The first run with the TensorRT provider is slow (it builds an FP16 engine,
can take minutes); the engine is cached in trt_cache/ so later runs start fast.
The *_nms.onnx model outputs (1, max_det, 6) = [x1, y1, x2, y2, conf, class]
in letterboxed-input pixels; rows with conf 0 are padding.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ["person", "vehicle", "traffic-light", "stop-sign"]
COLORS = [(0, 255, 0), (255, 160, 0), (0, 215, 255), (0, 0, 255)]  # BGR


def make_session(model_path):
    available = ort.get_available_providers()
    providers = []
    if "TensorrtExecutionProvider" in available:
        providers.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "trt_cache",
        }))
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    print("providers in use:", session.get_providers())
    return session


def letterbox(img, size):
    """Resize keeping aspect ratio, pad to size x size with gray. Returns
    (canvas, scale, x_offset, y_offset) so boxes can be mapped back."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo11n_coco4_nms.onnx")
    parser.add_argument("--source", default="0", help="video path or camera index")
    parser.add_argument("--out", default=None, help="write annotated video here")
    parser.add_argument("--conf", type=float, default=0.35, help="display threshold")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = all)")
    parser.add_argument("--show", action="store_true", help="cv2.imshow preview window")
    args = parser.parse_args()

    session = make_session(args.model)
    inp = session.get_inputs()[0]
    size = inp.shape[-1]  # static export: [1, 3, S, S]

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {args.source}")

    writer = None
    if args.out:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.frames and n >= args.frames):
            break
        canvas, r, dx, dy = letterbox(frame, size)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        dets = session.run(None, {inp.name: blob})[0][0]  # (max_det, 6)

        for x1, y1, x2, y2, conf, cls in dets:
            if conf < args.conf:
                continue
            cls = int(cls)
            # letterboxed 640-space -> original frame pixels
            p1 = (int((x1 - dx) / r), int((y1 - dy) / r))
            p2 = (int((x2 - dx) / r), int((y2 - dy) / r))
            cv2.rectangle(frame, p1, p2, COLORS[cls], 2)
            cv2.putText(frame, f"{NAMES[cls]} {conf:.2f}", (p1[0], max(p1[1] - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[cls], 2)
        n += 1
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("yolo11", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    dt = time.time() - t0
    print(f"{n} frames in {dt:.1f}s = {n / dt:.1f} FPS (incl. video I/O)")
    cap.release()
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
