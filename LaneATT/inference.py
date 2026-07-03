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


infer_params = {'conf_threshold': 0, 'nms_thres': 50.0, 'nms_topk': 4}
# print(type(model))

import io

device = torch.device("cpu")

# VideoInference.load_model (lib/video.py) does `torch.load(wieghts)`, which requires
# either a path or a seekable file-like object -- it can't accept a live nn.Module
# (torch.load needs .seek()/.read()). So rebuild the model the same way
# Runner.get_model does (architecture from config + state_dict from the checkpoint's
# 'model' key), then re-serialize it into an in-memory buffer that torch.load() can
# actually consume, without touching video.py.
config_path = "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/experiments/Testing_Pinnacle/config.yaml"
cfg = Config(config_path)
model = cfg.get_model()
state_dict = torch.load(
    "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/LaneATT/experiments/Testing_Pinnacle/models/model_0007.pt",
    map_location='cpu'
)['model']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

buffer = io.BytesIO()
torch.save(model, buffer)
buffer.seek(0)

from lib.video import VideoInference

video_test = "video_input/1.mp4"

video = VideoInference(model_wieghts=buffer, frame_limit = 99999, video_path = video_test, view = True, output_folder = "video_output", device = device)

video.image_eval()
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