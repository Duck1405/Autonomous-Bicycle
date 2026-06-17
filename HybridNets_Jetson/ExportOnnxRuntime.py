from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

import onnx
import torch

from hybridnets.model_runtime import MemoryEfficientSwish, Swish
from road_segmentation_model import HybridNetsRoadSegmentationModel
from video_common import DEFAULT_WEIGHTS

'''
  env PYTHONNOUSERSITE=1 PYTHONPATH= LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu \
  conda run -n hybridnets \
  python ExportOnnxRuntime.py \
    --weights ./hybridnets-d3_19_43740_best.pth \
    --output ./onnx/hybridnets_road_segmentation.onnx \
    --image-height 384 \
    --image-width 640 \
    --device cpu
'''

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export the packaged HybridNets road-segmentation model to ONNX")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the HybridNets .pth weights")
    parser.add_argument("--output", default="hybridnets_road_segmentation.onnx", help="Output ONNX path")
    parser.add_argument("--image-width", type=int, default=640, help="Static ONNX input width")
    parser.add_argument("--image-height", type=int, default=384, help="Static ONNX input height")
    parser.add_argument("--compound-coef", type=int, default=3)
    parser.add_argument("--backbone", type=str, default=None, help="Unsupported in this runtime bundle; leave unset")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Device used for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise TypeError("The checkpoint does not contain a recognizable state dict.")


def _normalize_common_prefix(key: str) -> str:
    for prefix in ("module.backbone.", "backbone.", "module."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _build_timm_block_map(state_dict: Mapping[str, torch.Tensor]) -> dict[tuple[int, int], int]:
    block_ids: set[tuple[int, int]] = set()
    pattern = re.compile(r"^encoder\.blocks\.(\d+)\.(\d+)\.")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            block_ids.add((int(match.group(1)), int(match.group(2))))
    return {block_id: index for index, block_id in enumerate(sorted(block_ids))}


def _find_expanded_timm_blocks(state_dict: Mapping[str, torch.Tensor]) -> set[tuple[int, int]]:
    expanded_blocks: set[tuple[int, int]] = set()
    pattern = re.compile(r"^encoder\.blocks\.(\d+)\.(\d+)\.conv_pwl\.")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            expanded_blocks.add((int(match.group(1)), int(match.group(2))))
    return expanded_blocks


def _normalize_timm_efficientnet_key(
    key: str,
    block_map: dict[tuple[int, int], int],
    expanded_blocks: set[tuple[int, int]],
) -> str:
    if key.startswith("encoder.conv_stem."):
        return key.replace("encoder.conv_stem.", "encoder._conv_stem.", 1)
    if key.startswith("encoder.bn1."):
        return key.replace("encoder.bn1.", "encoder._bn0.", 1)

    match = re.match(r"^encoder\.blocks\.(\d+)\.(\d+)\.(.+)$", key)
    if not match:
        return key

    stage_id = int(match.group(1))
    local_id = int(match.group(2))
    suffix = match.group(3)
    flat_id = block_map.get((stage_id, local_id))
    if flat_id is None:
        return key

    if (stage_id, local_id) in expanded_blocks:
        replacements = (
            ("conv_pw.", "_expand_conv."),
            ("bn1.", "_bn0."),
            ("conv_dw.", "_depthwise_conv."),
            ("bn2.", "_bn1."),
            ("se.conv_reduce.", "_se_reduce."),
            ("se.conv_expand.", "_se_expand."),
            ("conv_pwl.", "_project_conv."),
            ("bn3.", "_bn2."),
        )
    else:
        replacements = (
            ("conv_dw.", "_depthwise_conv."),
            ("bn1.", "_bn1."),
            ("se.conv_reduce.", "_se_reduce."),
            ("se.conv_expand.", "_se_expand."),
            ("conv_pw.", "_project_conv."),
            ("bn2.", "_bn2."),
        )
    for old, new in replacements:
        if suffix.startswith(old):
            suffix = suffix.replace(old, new, 1)
            break

    return f"encoder._blocks.{flat_id}.{suffix}"


def _normalize_state_dict(raw_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = {_normalize_common_prefix(key): value for key, value in raw_state_dict.items()}
    block_map = _build_timm_block_map(state_dict)
    expanded_blocks = _find_expanded_timm_blocks(state_dict)
    return {
        _normalize_timm_efficientnet_key(key, block_map, expanded_blocks): value
        for key, value in state_dict.items()
    }


def _load_checkpoint(weights_path: Path, device: torch.device) -> object:
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def load_runtime_weights(model: HybridNetsRoadSegmentationModel, weights_path: Path, device: torch.device) -> None:
    checkpoint = _load_checkpoint(weights_path, device)
    raw_state_dict = _extract_state_dict(checkpoint)
    state_dict = _normalize_state_dict(raw_state_dict)

    model_state_dict = model.backbone.state_dict()
    compatible_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state_dict and tuple(value.shape) == tuple(model_state_dict[key].shape)
    }
    if not compatible_state_dict:
        sample_keys = ", ".join(list(state_dict.keys())[:5])
        raise RuntimeError(f"No checkpoint keys matched the runtime backbone. Sample checkpoint keys: {sample_keys}")

    incompatible = model.backbone.load_state_dict(compatible_state_dict, strict=False)
    shape_mismatches = [
        key
        for key, value in state_dict.items()
        if key in model_state_dict and tuple(value.shape) != tuple(model_state_dict[key].shape)
    ]
    print(f"Loaded {len(compatible_state_dict)} matching tensors from {weights_path}")
    if incompatible.missing_keys:
        print(f"Missing runtime keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Ignored checkpoint keys: {len(incompatible.unexpected_keys)}")
    if shape_mismatches:
        print(f"Skipped shape-mismatched tensors: {len(shape_mismatches)}")


def prepare_for_onnx(model: HybridNetsRoadSegmentationModel) -> None:
    encoder = model.backbone.encoder
    if hasattr(encoder, "set_swish"):
        encoder.set_swish(memory_efficient=False)

    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, MemoryEfficientSwish):
                setattr(module, name, Swish())


def export_onnx(args: argparse.Namespace) -> None:
    weights_path = Path(args.weights).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested, but torch.cuda.is_available() is false.")

    model = HybridNetsRoadSegmentationModel(
        compound_coef=args.compound_coef,
        backbone_name=args.backbone,
    )
    load_runtime_weights(model, weights_path, device=torch.device("cpu"))
    prepare_for_onnx(model)
    model = model.to(device).eval()
    model.requires_grad_(False)

    dummy_input = torch.zeros(1, 3, args.image_height, args.image_width, dtype=torch.float32, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX to {output_path}")
    print(f"Input shape: {tuple(dummy_input.shape)}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["segmentation_logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "segmentation_logits": {0: "batch"},
            },
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX checker passed: {output_path}")


def main() -> None:
    export_onnx(build_parser().parse_args())


if __name__ == "__main__":
    main()
