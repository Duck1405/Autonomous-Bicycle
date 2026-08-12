from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from road_segmentation_model import HybridNetsRoadSegmentationModel, ROAD_SEGMENTATION_CLASSES


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export the road-only HybridNets checkpoint as a segmentation-only ONNX model")
    parser.add_argument("--weights", required=True, help="Path to the trained road-only HybridNets weights (.pth)")
    parser.add_argument("--output-onnx", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--compound-coef", type=int, default=3)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--metadata-json", type=str, default=None, help="Optional metadata JSON path")
    return parser


def maybe_validate_onnx(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("onnx is not installed, skipping model validation.")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX validation passed.")


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_onnx = (
        Path(args.output_onnx).expanduser().resolve()
        if args.output_onnx
        else weights_path.with_name(
            f"{weights_path.stem}_road_segmentation_{args.image_height}x{args.image_width}.onnx"
        )
    )
    metadata_path = (
        Path(args.metadata_json).expanduser().resolve()
        if args.metadata_json
        else output_onnx.with_suffix(".json")
    )
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    print(f"Loading weights: {weights_path}")
    print(f"Export device: {device}")
    print(f"Exporting to: {output_onnx}")

    model = HybridNetsRoadSegmentationModel(compound_coef=args.compound_coef, backbone_name=args.backbone)
    model.load_weights(weights_path)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    dummy_input = torch.randn(1, 3, args.image_height, args.image_width, dtype=torch.float32, device=device)
    if device.type == "cuda":
        dummy_input = dummy_input.to(memory_format=torch.channels_last)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_onnx),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["segmentation_logits"],
            dynamic_axes=None,
            dynamo=False,
        )

    maybe_validate_onnx(output_onnx)

    metadata = {
        "format": "hybridnets_road_segmentation_onnx",
        "weights": str(weights_path),
        "onnx": str(output_onnx),
        "input_name": "input",
        "output_name": "segmentation_logits",
        "input_shape": [1, 3, args.image_height, args.image_width],
        "classes": list(ROAD_SEGMENTATION_CLASSES),
        "preprocessing": {
            "color_order": "RGB",
            "input_scale": "float32 in [0, 1]",
            "normalize_mean": list(MEAN),
            "normalize_std": list(STD),
        },
        "export": {
            "device": str(device),
            "compound_coef": args.compound_coef,
            "backbone": args.backbone,
            "opset": args.opset,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
