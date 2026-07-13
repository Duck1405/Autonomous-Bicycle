"""LaneNet (+ optional H-Net) lane inference: model loading and per-frame
eval/draw, mirroring LaneATT.py (LaneATTInference). video.py owns the video
loop and calls into this class.

Model code copied from lanenet-lane-detection-pytorch lives in model/ and
lane_utils.py at the repo root. Only the two checkpoint paths vary between
runs; every algorithm parameter is pinned to that repo's defaults below.
"""
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from model.lanenet.LaneNet import LaneNet
from model.lanenet.backbone.H_Net import H_Net, build_H
from lane_utils import (
    cluster_lane_embeddings,
    fit_lane_polynomials,
    draw_lane_clusters,
    draw_all_lane_curves,
)


class LaneNetInference():
    """With an H-Net checkpoint lanes are drawn as fitted polynomial curves,
    without one as raw instance clusters."""

    # lanenet-lane-detection-pytorch/inference.py defaults
    WIDTH, HEIGHT = 512, 256            # LaneNet input (w, h)
    HNET_WIDTH, HNET_HEIGHT = 128, 64   # H-Net input (w, h)
    DELTA_V = 0.5
    MIN_CLUSTER_SIZE = 50
    MAX_LANES = 10
    POLY_ORDER = 3
    CURVE_THICKNESS = 4
    DILATION_ITERS = 2

    def __init__(self, model_path, hnet_path=None, device=torch.device("cpu")):
        self.device = device

        self.model_path = model_path
        self.lanenet = LaneNet(arch="ENet")
        self._load_weights(self.lanenet, model_path)

        self.hnet_path = hnet_path
        self.hnet = None
        if hnet_path:
            self.hnet = H_Net()
            self._load_weights(self.hnet, hnet_path)

        self.lane_tf = A.Compose([
            A.Resize(self.HEIGHT, self.WIDTH),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        # Same scaling H-Net was trained with (dataloader/hnet_dataset.py):
        # [0,1] floats, ImageNet-normalized.
        self.hnet_tf = A.Compose([
            A.Resize(self.HNET_HEIGHT, self.HNET_WIDTH),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) if self.hnet else None

    def _load_weights(self, model, path):
        try:
            weights = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            weights = torch.load(path, map_location=self.device)
        model.load_state_dict(weights)
        model.eval().to(self.device)

    def frame_eval(self, frame):
        """One BGR frame in -> (cluster_labels, curves_norm).
        curves_norm is None when no H-Net checkpoint was given."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lane_t = self.lane_tf(image=rgb)["image"]
        with torch.no_grad():
            outputs = self.lanenet(lane_t.unsqueeze(0).to(self.device))

        binary_pred = outputs["binary_seg_pred"][0, 0].detach().cpu().numpy().astype(np.uint8)
        # Cluster on the RAW instance embedding — that is the space the
        # discriminative loss was trained in; sigmoid squashes the embeddings
        # and collapses every lane into one cluster.
        emb = outputs.get("instance_embedding")
        if emb is None:
            emb = outputs["instance_seg_logits"]
        embedding = emb.detach().cpu()[0].numpy().astype(np.float32)

        cluster_labels = cluster_lane_embeddings(
            binary_pred, embedding, delta_v=self.DELTA_V,
            min_cluster_size=self.MIN_CLUSTER_SIZE, max_lanes=self.MAX_LANES)

        if self.hnet is None:
            return cluster_labels, None

        hnet_t = self.hnet_tf(image=rgb)["image"]
        with torch.no_grad():
            params = self.hnet(hnet_t.unsqueeze(0).to(self.device))
        H = build_H(params)[0]
        _, _, curves_norm = fit_lane_polynomials(
            cluster_labels, H, self.WIDTH, self.HEIGHT, self.POLY_ORDER)
        return cluster_labels, curves_norm

    def draw(self, frame, evaluation):
        """Draw a frame_eval() result onto a BGR frame; returns a new frame."""
        cluster_labels, curves_norm = evaluation
        if curves_norm is None:
            return draw_lane_clusters(frame, cluster_labels, self.DILATION_ITERS)
        return draw_all_lane_curves(frame, curves_norm, self.CURVE_THICKNESS)
