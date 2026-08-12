from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from backbone import HybridNetsBackbone
from utils.constants import MULTICLASS_MODE


ROAD_SEGMENTATION_CLASSES = ("background", "road", "lane")
ROAD_ONLY_NUM_DETECTION_CLASSES = 1
ROAD_ONLY_SEGMENTATION_CLASSES = 2
PAPER_ANCHOR_SCALES = (2**0, 2**0.7, 2**1.32)
PAPER_ANCHOR_RATIOS = ((0.62, 1.58), (1.0, 1.0), (1.58, 0.62))


class HybridNetsRoadSegmentationModel(nn.Module):
    """Run only the trained segmentation path of the road-only HybridNets checkpoint."""

    def __init__(self, compound_coef: int = 3, backbone_name: str | None = None):
        super().__init__()
        self.backbone = HybridNetsBackbone(
            num_classes=ROAD_ONLY_NUM_DETECTION_CLASSES,
            compound_coef=compound_coef,
            ratios=PAPER_ANCHOR_RATIOS,
            scales=PAPER_ANCHOR_SCALES,
            seg_classes=ROAD_ONLY_SEGMENTATION_CLASSES,
            backbone_name=backbone_name,
            seg_mode=MULTICLASS_MODE,
        )

    def load_weights(self, weights_path: Path) -> dict:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.backbone.load_state_dict(state_dict, strict=False)
        return checkpoint if isinstance(checkpoint, dict) else {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        p2, p3, p4, p5 = self.backbone.encoder(inputs)[-4:]
        p3, p4, p5, p6, p7 = self.backbone.bifpn((p3, p4, p5))
        decoder_outputs = self.backbone.bifpndecoder((p2, p3, p4, p5, p6, p7))
        return self.backbone.segmentation_head(decoder_outputs)


def build_road_segmentation_model(
    weights_path: Path,
    compound_coef: int,
    backbone_name: str | None,
    device: torch.device,
    gpu_ids: list[int] | None = None,
) -> nn.Module:
    model = HybridNetsRoadSegmentationModel(compound_coef=compound_coef, backbone_name=backbone_name)
    model.load_weights(weights_path)
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    gpu_ids = gpu_ids or []
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    return model
