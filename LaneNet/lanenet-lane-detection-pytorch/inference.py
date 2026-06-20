import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score
import sys
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation, label
from matplotlib.patches import Patch

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help = "Model Wieghts", default = "/Users/amannindra/Projects/Autonomous-Bicycle/LaneNet/lanenet-lane-detection-pytorch/trained_models/best_model.pth")
    parser.add_argument("--video_file", default = "")
    parser.add_argument("--output_file")
    return parser.parse_args()
    
    

def frame(model, dummy_input, frame):
    # dummy_input = load_test_data(image_file, transform).to(DEVICE)
    # orig = np.array(Image.open(image_file).convert("RGB"))
    
    orig_h, orig_w = frame.shape[:2]

    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    with torch.no_grad():
        outputs = model(dummy_input)
        
    

    outputs.keys()

    instanc_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy().astype(np.uint8)

    pred_mask_orig = cv2.resize(binary_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


    lane_components, num_lanes = label(pred_mask_orig == 1)
    component_sizes = np.bincount(lane_components.ravel())
    component_ids = np.arange(1, len(component_sizes))
    component_ids = component_ids[component_sizes[component_ids] > 50]
    component_ids = sorted(component_ids, key=lambda lane_id: component_sizes[lane_id], reverse=True)

    lane_colors = np.array([
        [255, 0, 0],      # red
        [0, 255, 0],      # green
        [0, 128, 255],    # blue/orange-ish in RGB display
        [255, 255, 0],    # yellow
        [255, 0, 255],    # magenta
        [0, 255, 255],    # cyan
        [0, 255, 128],
        [0, 128, 128],
        [128, 255, 255],
        [128, 128, 128],
    ], dtype=np.uint8)

    overlay = frame.copy()
    legend_handles = []

    for lane_index, component_id in enumerate(component_ids[:len(lane_colors)]):
        lane_mask = binary_dilation(lane_components == component_id, iterations=2)
        color = lane_colors[lane_index]
        overlay[lane_mask] = color
        legend_handles.append(
            Patch(
                facecolor=color / 255.0,
                edgecolor='black',
                label=f'Lane {lane_index + 1}',
            )
        )



    print(type(overlay))
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    return blended

    # plt.figure(figsize=(14, 8))
    # plt.imshow(blended)
    # if legend_handles:
    #     plt.legend(handles=legend_handles, loc='lower right')
    # plt.axis("off")
    # plt.show()
    
def load_test_data_2(img,transform):
    img = transform(image=np.array(img))['image']
    return img


def main(args):
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    model = LaneNet(arch="ENet")
    weights = torch.load(args.model, map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights)

    
    transform = A.Compose([
      A.Resize(height=256, width=512),
      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ToTensorV2(),
    ])
    
    model.eval()
    model.to(DEVICE)
    
    video_cap = cv2.VideoCapture(args.video_file)
    count, success = 0, True
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print(f"FPB: {fps}")
    width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Width: {width}")
    height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"height: {height}")
    output_video = args.output_file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (int(width), int(height)))
    
    while success:
        success, image = video_cap.read() # Read frame
        if success:  
            img = transform(image=np.array(image))['image']
            orig = np.array(image)
            new_frame = frame(model, img, orig)
            count += 1
            video_writer.write(new_frame)
        count += 1
        if count > 200:
            break
        
    video_writer.release()
    print("Video successfully generated!")


if __name__ == "__main__":
    
    args = get_args()
    main(args)
    
    
