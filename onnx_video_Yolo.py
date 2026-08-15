"""Save one frame with the YOLOv11 ONNX detections drawn on it.

Single-frame sanity check for the ONNX graph: grab a frame, run it through
onnxruntime, decode + draw with the same jetson_tools helpers the TensorRT
benchmark uses, and write the result into output_check/.
"""
import argparse
import time
from pathlib import Path

import cv2
import onnxruntime as ort
import sys

from jetson_tools.postprocess import draw_boxes, yolo_decode
from jetson_tools.preprocess import pre_yolo_meta


def get_frame(video_path, frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame < 0 or frame >= total_frames:
        cap.release()
        raise ValueError(f"frame {frame} out of range (video has {total_frames} frames)")

    # Seek directly to the requested frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"could not read frame {frame} from {video_path}")
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="LaneATT/video_input/1.mp4")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--onnx", default="LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=Path("output_check"))
    args = parser.parse_args()

    frame = get_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    print(f"frame: {args.video} #{args.frame}  ({w}x{h})")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)
    # get_providers() is what ORT actually bound. Asking for CUDA is not the same
    # as getting it: the CPU-only wheel warns once and falls back silently.
    active = session.get_providers()
    print(f"providers: {active}")
    if 'CUDAExecutionProvider' not in active:
        print("  -> running on CPU (needs the Jetson onnxruntime-gpu wheel for CUDA)")
        sys.exit(1)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # The letterbox (r, dx, dy) is needed to map boxes back to frame coords.
    input_data, r, dx, dy = pre_yolo_meta(frame)

    t0 = time.perf_counter()
    outputs = session.run([output_name], {input_name: input_data})
    t1 = time.perf_counter()
    print(f"ONNX Inference Time: {t1 - t0:.4f} seconds")

    # (1, 300, 6) -> [(p1, p2, conf, cls)] already in original frame coords.
    dets = yolo_decode(outputs[0], r, dx, dy, conf_threshold=args.conf)
    print(f"detections: {len(dets)}  confs: {[round(d[2], 3) for d in dets]}")

    draw_boxes(frame, dets)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{Path(args.video).stem}_frame{args.frame}_yolo.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"saved {out_path}")
