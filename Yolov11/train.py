"""Fine-tune YOLOv11 on the 4-class bicycle dataset (person/vehicle/traffic-light/stop-sign).

Usage:
    python train.py --size n                    # full training run
    python train.py --size n --epochs 1 --batch 8 --device cpu --fraction 0.01   # smoke test
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["n", "s", "m"], default="n", help="YOLOv11 model size")
    parser.add_argument("--data", default=str(Path(__file__).parent / "dataset/coco4/data.yaml"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="0", help='"0" for first GPU, "cpu", or "mps"')
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use (smoke tests)")
    args = parser.parse_args()

    model = YOLO(f"yolo11{args.size}.pt")  # pretrained COCO weights; this is a fine-tune
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project="runs",
        name=f"yolo11{args.size}_coco4",
    )
    # Final report on the held-out test split (train/val used valid/).
    metrics = model.val(data=args.data, split="test", device=args.device)
    print(f"test split: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
