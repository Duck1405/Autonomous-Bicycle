"""Dataset for training H-Net on TuSimple ground-truth lane points.

H-Net trains on the raw TuSimple annotations (label_data_*.json), independently
of LaneNet and independently of tusimple_transform.py's binary/instance images.
Each json line has:
    raw_file  : image path relative to the TuSimple train_set root
    h_samples : list of y pixel coordinates (the same for every lane)
    lanes     : list of lanes; each lane is a list of x pixel coords (-2 = absent)

Each sample yields:
    image : (3, H, W) float tensor, resized to (resize_h, resize_w) and
            normalized with ImageNet statistics -> H-Net's conditioning input.
    lanes : list of (M_k, 2) float tensors with [x, y] coordinates normalized to
            [0, 1] by the ORIGINAL image size. Normalizing by the original size
            (not the resized size) makes the coordinate frame resolution
            independent, so it matches the frame used at inference time.
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HNetDataset(Dataset):
    def __init__(self, label_files, image_root, resize_w=128, resize_h=64,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if isinstance(label_files, str):
            label_files = [label_files]
        self.image_root = image_root
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        self.samples = []
        for lf in label_files:
            with open(lf, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        if not self.samples:
            raise RuntimeError("No samples parsed from: {}".format(label_files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        img_path = os.path.join(self.image_root, info["raw_file"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h0, w0 = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_w, self.resize_h),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        image = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()   # (3, H, W)

        h_samples = info["h_samples"]
        lanes = []
        for lane in info["lanes"]:
            pts = [(x / w0, y / h0) for x, y in zip(lane, h_samples) if x >= 0]
            if pts:
                lanes.append(torch.tensor(pts, dtype=torch.float32))     # (M, 2)

        return image, lanes


def hnet_collate(batch):
    """Lanes are ragged (variable count / length per image), so we keep them as
    a Python list and only stack the images."""
    images = torch.stack([b[0] for b in batch], dim=0)   # (N, 3, H, W)
    batch_lanes = [b[1] for b in batch]                  # list of lists of (M, 2)
    return images, batch_lanes
