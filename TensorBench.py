



# from cuda.bindings import runtime as cudart
import sys
# import tensorrt as trt
from pathlib import Path

import re
import time
from pathlib import Path

import cv2
import numpy as np

import time 

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jetson_tools"))
from jetson_tools.postprocess import (LaneHysteresis, depth_colorize, draw_boxes,  # noqa: E402
                         draw_lanes, laneatt_decode, yolo_decode)
from jetson_tools.preprocess import pre_depth, pre_laneatt, pre_yolo_meta  # noqa: E402
from jetson_tools.trt_runner import CudaRT, TrtEngine  # noqa: E402



def check(err):
    # cuda-python returns (cudaError_t, *values); raise on any non-success status.
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

def load_engine(runtime, engine_path):
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())



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
    # cv2.imshow("title", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    cap.release()
    return img
    

def main(frame, model):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    #engine = load_engine(runtime, model)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example script using argparse.")

    # 2. Add arguments
    # Positional argument (Required by default)
    parser.add_argument("--frame", type=int, default=2100)
    parser.add_argument("--video", type=str, default="video_input/1.mp4", help="Path to the video file.")
    args = parser.parse_args()
    
    depth_engine= Path("LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine")
    laneNet = Path("LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine")
    YoloS_engine = Path("LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.engine")
    YoloM_onnx = Path("LaneATT/onnxmodels/YoloM/yolo11m_coco4_nms.onnx")
    YoloN_engine = Path("LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.engine")

    videos = "Videos2"
    video_main = [videos / Path("camera.mp4")]
    video_main = ["Videos2/camera.mp4"]

    
    frame = get_frame(video_main[0], args.frame)
    
    main(
        frame, laneNet
    )