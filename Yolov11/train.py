"""Fine-tune YOLOv11 on 4 classes for the bicycle: person / vehicle / traffic-light / stop-sign.

Usage:
    python train.py --size n                                                     # full run (cluster)
    python train.py --size n --epochs 1 --batch 8 --device mps --fraction 0.002  # smoke test
    python train.py --size n --resume runs/yolo11n_coco4/weights/last.pt         # continue after walltime

Artifacts land in models/yolo11<size>_coco4/run<K>/ (a fresh runK per training,
same numbering convention as LaneATT's video output — nothing gets overwritten):
    yolo11<size>_coco4.pt        best checkpoint (ultralytics format)
    yolo11<size>_coco4.onnx      raw head (1, 8, 8400) — for trtexec/TensorRT engine builds
    yolo11<size>_coco4_nms.onnx  end-to-end with NMS (1, 300, 6) — for onnxruntime-gpu on the Jetson
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def next_run_dir(base: Path) -> Path:
    """base/run1, base/run2, ... — first number after the highest already taken."""
    base.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[3:]) for p in base.glob("run*") if p.name[3:].isdigit()]
    run_dir = base / f"run{max(taken, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["n", "s", "m"], default="n", help="YOLOv11 model size")
    parser.add_argument("--data", default="dataset/coco4/data.yaml", help="data.yaml from prepare_dataset.py")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--device", default="0", help='"0" for first GPU, "cpu", or "mps"')
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (smoke tests)")
    parser.add_argument("--save-period", type=int, default=10, help="also save a checkpoint every N epochs")
    parser.add_argument("--resume", default=None, help="runs/<run>/weights/last.pt to continue an interrupted run")
    args = parser.parse_args()

    stem = f"yolo11{args.size}_coco4"
    models_dir = next_run_dir(Path(__file__).resolve().parent / "models" / stem)
    print(f"artifacts will be saved to {models_dir}")

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(f"yolo11{args.size}.pt")  # pretrained COCO weights; this is a fine-tune
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            fraction=args.fraction,
            save_period=args.save_period,
            project="runs",
            name=stem,
        )

    # Keep deliverables out of runs/: copy the best checkpoint to models/.
    final_pt = models_dir / f"{stem}.pt"
    shutil.copy2(model.trainer.best, final_pt)

    metrics = YOLO(final_pt).val(data=args.data, split="val", device=args.device,
                                 project="runs", name=f"{stem}_val")
    print(f"val: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")

    # Jetson deployment exports. Both are FP32 static 1x3x<imgsz>x<imgsz> ONNX;
    # FP16 happens on the Jetson (TensorRT engine build / provider flag), because
    # TensorRT engines are device-specific and must be built there.
    # 1) End-to-end with NMS baked in (conf/iou fixed at export) -> onnxruntime-gpu.
    onnx_nms = YOLO(final_pt).export(format="onnx", imgsz=args.imgsz, opset=13,
                                     simplify=True, nms=True, conf=0.25, iou=0.45,
                                     device="cpu")
    shutil.move(onnx_nms, models_dir / f"{stem}_nms.onnx")
    # 2) Raw head output -> trtexec engine building or custom decode+NMS.
    #    Lands at models/<stem>.onnx (export writes next to the .pt).
    YOLO(final_pt).export(format="onnx", imgsz=args.imgsz, opset=13, simplify=True,
                          device="cpu")

    print("artifacts:", final_pt, models_dir / f"{stem}.onnx", models_dir / f"{stem}_nms.onnx")


if __name__ == "__main__":
    main()
