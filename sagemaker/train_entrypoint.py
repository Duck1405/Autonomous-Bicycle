from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SageMaker entrypoint for the Autonomous-Bicycle train.py script.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--image-channel",
        default=os.environ.get("SM_CHANNEL_IMAGES", "/opt/ml/input/data/images"),
        help="SageMaker channel directory containing image splits.",
    )
    parser.add_argument(
        "--annotation-channel",
        default=os.environ.get("SM_CHANNEL_ANNOTATIONS", "/opt/ml/input/data/annotations"),
        help="SageMaker channel directory containing JSON annotation splits.",
    )
    parser.add_argument(
        "--train-script",
        default="/opt/program/train.py",
        help="Path to the training script bundled in the container.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded directly to train.py.",
    )
    return parser


def load_sagemaker_hyperparameters() -> dict[str, object]:
    hyperparameters_path = Path("/opt/ml/input/config/hyperparameters.json")
    if not hyperparameters_path.exists():
        return {}
    with hyperparameters_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hyperparameters_to_cli_args(hyperparameters: dict[str, object]) -> list[str]:
    cli_args: list[str] = []
    for key, value in hyperparameters.items():
        if value is None:
            continue
        cli_args.extend([f"--{key}", str(value)])
    return cli_args


def main() -> None:
    args, unknown_args = build_parser().parse_known_args()

    image_root = Path(args.image_channel)
    json_root = Path(args.annotation_channel)
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    hyperparameters = load_sagemaker_hyperparameters()

    command = [
        sys.executable,
        args.train_script,
        "--image-root",
        str(image_root),
        "--json-root",
        str(json_root),
        "--model-dir",
        model_dir,
        "--output-dir",
        output_dir,
    ]
    command.extend(hyperparameters_to_cli_args(hyperparameters))

    if unknown_args:
        command.extend(unknown_args)
    if args.extra_args:
        command.extend(args.extra_args)

    print("Launching training command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
