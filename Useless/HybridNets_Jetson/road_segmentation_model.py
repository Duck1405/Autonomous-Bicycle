from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from backbone_runtime import HybridNetsBackboneRuntime
from utils.constants import MULTICLASS_MODE


ROAD_SEGMENTATION_CLASSES = ("background", "road", "lane")
ROAD_ONLY_SEGMENTATION_CLASSES = 2


class HybridNetsRoadSegmentationModel(nn.Module):
    def __init__(self, compound_coef: int = 3, backbone_name: str | None = None):
        super().__init__()
        if backbone_name is not None:
            raise RuntimeError(
                "Custom backbone selection is not packaged in this runtime bundle. "
                f"Received backbone={backbone_name!r}."
            )

        self.backbone = HybridNetsBackboneRuntime(
            compound_coef=compound_coef,
            seg_classes=ROAD_ONLY_SEGMENTATION_CLASSES,
            seg_mode=MULTICLASS_MODE,
        )

    def load_weights(self, weights_path: Path) -> dict:
        checkpoint = torch.load(weights_path, map_location="cuda")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.backbone.load_state_dict(state_dict, strict=False)
        return checkpoint if isinstance(checkpoint, dict) else {}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)

def build_road_segmentation_model(
    weights_path: Path,
    compound_coef: int,
    backbone_name: str | None,
    device: torch.device,
    device_id: int,
    use_ddp: bool = True,
) -> nn.Module:
    model = HybridNetsRoadSegmentationModel(
        compound_coef=compound_coef,
        backbone_name=backbone_name,
    )

    print(f"Loading model onto {device}")

    model.load_weights(weights_path)
    model = model.to(device)

    # if device.type == "cuda":
    #     model = model.to(memory_format=torch.channels_last)

    if use_ddp:

        model = DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
        )

    model.eval()
    model.requires_grad_(False)

    return model
