import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment
import sys

import time
import os
import sys

import torch
# from model.lanenet.train_lanenet import train_model
# from dataloader.data_loaders import TusimpleSet
# from dataloader.transformers import Rescale
# from model.lanenet.LaneNet import LaneNet

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader, Dataset

from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

# from model.utils.cli_helper import parse_args
# from model.eval_function import Eval_Score
import sys
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from scipy.ndimage import binary_dilation, label
from matplotlib.patches import Patch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import json
from PIL import Image
import io

from lib.config import Config

import warnings
warnings.filterwarnings('ignore')

def lanes_to_px(lanes, w, h):
    out = []
    for lane in lanes:
        pts = lane.points.copy().astype(float)
        pts[:, 0] *= w          # Lane.points are normalized (x, y) in [0, 1]
        pts[:, 1] *= h
        out.append(pts.round().astype(int))
    return out
#check_file(model_path)

# config_path = "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/experiments/Testing_Pinnacle/config.yaml"
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
# cfg = Config(config_path)
# exp = Experiment(exp_name = "Testing_Pinnacle", args = None, mode="eval")
# device = DEVICE
# model = cfg.get_model()
# # print(model)
# runner = Runner(cfg, exp, device, view=True, resume=False, deterministic=False)
# print("Runner is intialized")
# model = runner.eval(epoch = 7)


# print(type(model))
from lib.video import VideoInference

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Default input video(s) for a zero-argument (MacBook) run; override with --videos.
filesed = [Path("video_input") / Path('IMG_6540.MOV'),]# Path("video_input") / Path('IMG_6892.MOV'), Path("video_input") / Path('IMG_6893.MOV')]

# (config.yaml, checkpoint) per model. Each experiment needs its OWN config
# because the backbone differs (resnet34 / resnet152 / resnet50).
# NEWEST_MODELS = [
#     ("experiments/LaneATTresnet18Aug2/config.yaml", "experiments/LaneATTresnet18Aug2/models/model_0020.pt"),
#     ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0020.pt"),
#     ("experiments/LaneATTresnet50Aug2/config.yaml", "experiments/LaneATTresnet50Aug2/models/model_0030.pt"),
#     ("experiments/LaneATTresnet101Aug2/config.yaml", "experiments/LaneATTresnet101Aug2/models/model_0024.pt"),
#     ("experiments/LaneATTresnet152Aug2/config.yaml", "experiments/LaneATTresnet152Aug2/models/model_0030.pt"),
# ]
# Default (MacBook) model set: (LaneATT config, LaneATT checkpoint, YOLO checkpoint).
# A zero-argument run uses exactly this list. On the server, don't edit this list —
# override the machine-specific paths from the CLI instead, e.g.:
#   python inference.py --yolo /home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11n_coco4/run7/yolo11n_coco4.pt
MODELSED = [
    # ("experiments/LaneATTresnet18Aug2/config.yaml", "experiments/LaneATTresnet18Aug2/models/model_0019.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11m_coco4_val3_new/run5/yolo11m_coco4.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11n_coco4_val3_new/run7/yolo11n_coco4.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11s_coco4_val3_new/run5/yolo11s_coco4.pt"),
    # server-side YOLO checkpoints (pass them via --yolo instead of editing this list):
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11m_coco4/run5/yolo11m_coco4.pt"),
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11n_coco4/run7/yolo11n_coco4.pt"),
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11s_coco4/run5/yolo11s_coco4.pt"),
    # ("experiments/LaneATTresnet50Aug2/config.yaml", "experiments/LaneATTresnet50Aug2/models/model_0015.pt"),
    # ("experiments/LaneATTresnet101Aug2/config.yaml", "experiments/LaneATTresnet101Aug2/models/model_0017.pt"),
    # ("experiments/LaneATTresnet152Aug2/config.yaml", "experiments/LaneATTresnet152Aug2/models/model_0015.pt" "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11n_coco45/weights/last.pt")
]

def video_inference(MODELS, files, frame_limit = 1000, output_root = Path("video_output_3"), yolo_conf = 0.2):
    video = None
    model_times = []   # (label, seconds) per model, printed at the end
    for config_path, path_model, path_yolo in MODELS:
        if not (Path(config_path).exists() and Path(path_model).exists()):
            print(f"SKIPPING {path_model}: config or checkpoint not found")
            continue
        if not Path(path_yolo).exists():
            print(f"SKIPPING {path_yolo}: YOLO checkpoint not found")
            continue

        cfg = Config(config_path)
        name = Path(path_model).stem
        model_name = Path(path_model).parent.parent.name
        output_folder = Path(output_root) / model_name / name
        print(f"=== {model_name}/{name} -> {output_folder} ===")

        if video is None:
            video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = frame_limit, video_path = str(files[0]), view = True, output_folder = output_folder, device = device, yolo_path = path_yolo, yolo_conf = yolo_conf)
        else:
            # Same pipeline object: swap the LaneATT model in place, keep YOLO loaded.
            # There is no set_yolo(): every entry after the first keeps the FIRST
            # entry's YOLO model, even if its tuple names a different checkpoint.
            video.set_model(cfg.get_model(), path_model)
            video.set_output_folder(output_folder)

        t_model = time.perf_counter()
        for i in files:
            print(i)
            video.set_video_path(str(i))
            video.video_eval()
        
            
        model_times.append((f"{model_name}/{name}", time.perf_counter() - t_model))

    print("\n=== time per model (all videos) ===")
    for label, seconds in model_times:
        print(f"{label}: {seconds:.1f} s ({seconds / 60:.1f} min)")

def image_inference(MODELS, files, frame):
   for config_path, path_model, path_yolo in MODELS:
        if not (Path(config_path).exists() and Path(path_model).exists()):
            print(f"SKIPPING {path_model}: config or checkpoint not found")
            continue

        cfg = Config(config_path)
        name = Path(path_model).stem
        model_name = Path(path_model).parent.parent.name
        output_folder = Path("image_output_1") / model_name / name
        print(f"=== {model_name}/{name} -> {output_folder} ===")
        video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = 1000, video_path = str(files[0]), view = True, output_folder = output_folder, device = device, yolo_path = path_yolo, yolo_conf = 0.1)
        video.set_model(cfg.get_model(), path_model)
        video.set_output_folder(output_folder)
        steering, ego_vehicle = video.image_eval(frame)
        print(f"steering: {steering}")
        print(f"ego_vehicle: {ego_vehicle}")
        
        
# image_inference(MODELSED, filesed, 200)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LaneATT + YOLO video inference. A zero-argument run uses the "
                    "MacBook defaults baked into MODELSED / filesed; on the server, "
                    "override the machine-specific paths from the CLI instead of "
                    "editing this file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--videos", nargs="+", type=Path, default=filesed,
                        help="input video file(s)")
    parser.add_argument("--config", default=None,
                        help="LaneATT config.yaml (default: taken from MODELSED)")
    parser.add_argument("--laneatt", default=None,
                        help="LaneATT checkpoint .pt (default: taken from MODELSED)")
    parser.add_argument("--yolo", nargs="+", default=None,
                        help="YOLO checkpoint(s); one full run per checkpoint "
                             "(default: taken from MODELSED)")
    parser.add_argument("--frame-limit", type=int, default=500,
                        help="max frames processed per video")
    parser.add_argument("--output-dir", type=Path, default=Path("video_output_3"),
                        help="root folder for annotated output videos")
    parser.add_argument("--yolo-conf", type=float, default=0.2,
                        help="YOLO confidence threshold")
    return parser.parse_args()


def build_models(args):
    """MODELSED is the default; --config/--laneatt/--yolo each override that field
    for every run (MODELSED entries share one LaneATT config+checkpoint, so the
    first entry fills in whatever isn't overridden)."""
    if args.config is None and args.laneatt is None and args.yolo is None:
        return MODELSED
    config_path = args.config or MODELSED[0][0]
    laneatt_path = args.laneatt or MODELSED[0][1]
    yolos = args.yolo if args.yolo is not None else [m[2] for m in MODELSED]
    return [(config_path, laneatt_path, str(y)) for y in yolos]


def main():
    args = parse_args()
    print(f"Using Device: {device}")

    missing = [v for v in args.videos if not Path(v).exists()]
    if missing:
        sys.exit("video(s) not found: " + ", ".join(str(v) for v in missing))

    video_inference(build_models(args), args.videos, args.frame_limit,
                    output_root=args.output_dir, yolo_conf=args.yolo_conf)


if __name__ == "__main__":
    main()
        


# video_inference(MODELSED, filesed)     

# DON"T REMOVE THIS SYS.exit() I am testing soemthing out with the code about
# sys.exit()



# ---- LaneNet + H-Net (model_type="laneNet"; everything else at the lanenet repo defaults) ----
# LANENET_MODEL = "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/lanenet-lane-detection-pytorch/trained_models/LaneNewTrained.pth"
# HNET_MODEL    = "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/lanenet-lane-detection-pytorch/trained_models/hnet_best.pth"
# YOLO_MODEL    = "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11n_coco45/weights/last.pt"

# model_times = []   # (label, seconds) per model, printed at the end

# if not Path(LANENET_MODEL).exists():
#     print(f"SKIPPING LaneNet: {LANENET_MODEL} not found")
# else:
#     lanenet_name = Path(LANENET_MODEL).stem
#     lanenet_out = Path("video_output_2") / "LaneNet" / lanenet_name
#     print(f"=== LaneNet/{lanenet_name} -> {lanenet_out} ===")
#     video = VideoInference(model_type = "laneNet", model_path = LANENET_MODEL, hnet_path = HNET_MODEL,
#                            frame_limit = 1000, video_path = str(files[0]), view = True,
#                            output_folder = lanenet_out, device = device,
#                            yolo_path = YOLO_MODEL, yolo_conf = 0.4)
#     t_model = time.perf_counter()
#     for i in files:
#         print(i)
#         video.set_video_path(str(i))
#         video.video_eval()
#     model_times.append((f"LaneNet/{lanenet_name}", time.perf_counter() - t_model))

# print("\n=== time per model (all videos) ===")
# for label, seconds in model_times:
#     print(f"{label}: {seconds:.1f} s ({seconds / 60:.1f} min)")

# ---- old per-model blocks (replaced by the MODELS loop above) ----

# config_path = "experiments/LaneATTresnet34Aug2/config.yaml"
# cfg = Config(config_path)
#
# path_model = "experiments/LaneATTresnet34Aug2/models/model_0013.pt"
# name = Path(path_model).stem
# model_name = Path(path_model).parent.parent.name
#
# print(f"name: {name}")
# print(f"model_name: {model_name}")
#
# output_folder = Path("video_output_2") / Path(model_name) / name
# print(f"output_folder: {output_folder}")
#
# s = "video_input/IMG_6892.MOV"
# d = "video_input/IMG_6893.MOV"
#
# video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = 9999, video_path = str(files[0]), view = True, output_folder = output_folder, device = device, yolo_path = "lib/yolo/models/yolo11s.pt", yolo_conf = 0.6)
#
# for i in files:
#     print(i)
#     video_test = str(i)
#     video.set_video_path(video_test)
#     video.video_eval()
#
#
# path_model = "experiments/LaneATTresnet152Aug2/models/model_0015.pt"
# name = Path(path_model).stem
# model_name = Path(path_model).parent.parent.name
#
# print(f"name: {name}")
# print(f"model_name: {model_name}")
#
# output_folder = Path("video_output_2") / Path(model_name) / name
# print(f"output_folder: {output_folder}")
#
# config_path = "experiments/LaneATTresnet152Aug2/config.yaml"
# cfg = Config(config_path)
#
# video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = 9999, video_path = str(files[0]), view = True, output_folder = output_folder, device = device, yolo_path = "lib/yolo/models/yolo11s.pt", yolo_conf = 0.6)
#
# for i in files:
#     print(i)
#     video_test = str(i)
#     video.set_video_path(video_test)
#     video.video_eval()
#
#
# path_model = "experiments/LaneATTresnet50Aug2/models/model_0015.pt"
#
# name = Path(path_model).stem
# model_name = Path(path_model).parent.parent.name
#
# print(f"name: {name}")
# print(f"model_name: {model_name}")
#
# output_folder = Path("video_output_2") / Path(model_name) / name
# print(f"output_folder: {output_folder}")
#
# config_path = "experiments/LaneATTresnet50Aug2/config.yaml"
# cfg = Config(config_path)
#
# video = VideoInference(model_archiecture = cfg.get_model(), model_path=path_model, frame_limit = 9999, video_path = str(files[0]), view = True, output_folder = output_folder, device = device, yolo_path = "lib/yolo/models/yolo11s.pt", yolo_conf = 0.6)
#
# for i in files:
#     print(i)
#     video_test = str(i)
#     video.set_video_path(video_test)
#     video.video_eval()

# video_test = "video_input/1.mp4"

# video = VideoInference(model_wieghts=buffer, frame_limit = 99999, video_path = video_test, view = True, output_folder = "video_output", device = device)

# video.video_eval()

# video_test = ""
# model = LaneATT(backbone = "resnet18", topk_anchors = 1000, anchors_freq_path = "data/culane_anchors_freq.pt" )
# state_dict = torch.load("experiments/Testing_Pinnacle/models/model_0007.pt", map_location='cpu')['model']
# model.load_state_dict(state_dict)
# model = model.to(device)
# model.eval()

# with torch.no_grad():
#     for idx in range(n_show):
#         img_t, label, _ = test_dataset[idx]
#         x = img_t.unsqueeze(0).to(device)
#         print(x.shape)

#         output = model(x, **infer_params)
#         pred_lanes = model.decode(output, as_lanes=True)[0]   # predicted Lane objects
#         gt_lanes   = test_dataset.label_to_lanes(label)        # ground-truth Lane objects

#         n_props = output[0][0].shape[0]                        # proposals surviving NMS
#         print(f"idx {idx}: {n_props} proposals after NMS -> {len(pred_lanes)} decoded lanes")

#         # build a displayable RGB image from the model's own input tensor (normalize:false in cfg)
#         img = (img_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()       # cv2.imread loads BGR

#         for pts in lanes_to_px(gt_lanes,   img.shape[1], img.shape[0]):   # GT = blue
#             for p0, p1 in zip(pts[:-1], pts[1:]):
#                 cv2.line(img, tuple(p0), tuple(p1), (0, 0, 255), 2)
#         for pts in lanes_to_px(pred_lanes, img.shape[1], img.shape[0]):   # pred = green
#             for p0, p1 in zip(pts[:-1], pts[1:]):
#                 cv2.line(img, tuple(p0), tuple(p1), (0, 255, 0), 3)

#         ax = plt.subplot(2, 3, idx + 1)
#         ax.imshow(img)
#         ax.set_title(f"idx {idx}: {len(pred_lanes)} pred (green) / {len(gt_lanes)} GT (blue)")
#         ax.axis('off')
# plt.tight_layout(); plt.show()