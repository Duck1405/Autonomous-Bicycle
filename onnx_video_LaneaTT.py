"""Run laneatt ONNX inference with lib.video.VideoInference visualization.

Run from the Autonomous-Bicycle repository root. CUDA is the default provider;
use --provider cpu for explicit CPU inference. --no-render keeps timing only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "LaneATT"))
from lib.onnx_video import parse_args as _parse_args, run

LABEL = "laneatt"


def parse_args():
    return _parse_args(LABEL)


def main():
    return run(LABEL, parse_args())


if __name__ == "__main__":
    main()
