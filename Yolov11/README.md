# YOLOv11 — 4-class detector for the autonomous bicycle

Detects `person`, `vehicle` (car+motorcycle+bus+truck), `traffic-light`, `stop-sign`.
Fine-tuned from pretrained YOLOv11 weights on official COCO 2017 remapped to 4 classes.

## Pipeline

| file | purpose |
|---|---|
| `prepare_dataset.py` | COCO 2017 JSONs → 4-class YOLO dataset (symlinked images, box labels) |
| `train.py` | train + val, then saves deliverables into `models/` |
| `yolo11n.sh` / `yolo11s.sh` / `yolo11m.sh` | SLURM jobs (1× L40S each) |
| `jetson_infer_onnx.py` | run the exported ONNX with onnxruntime (Jetson-ready) |

`train.py` leaves three artifacts in `models/yolo11<size>_coco4/run<K>/` — a fresh
`run<K>` folder per training (LaneATT-style numbering), so nothing gets overwritten:

- `yolo11<size>_coco4.pt` — best checkpoint (ultralytics format)
- `yolo11<size>_coco4.onnx` — raw head `(1, 8, 8400)`, for `trtexec` engine builds or custom decode
- `yolo11<size>_coco4_nms.onnx` — end-to-end with NMS baked in `(1, 300, 6)` = `[x1,y1,x2,y2,conf,cls]`, for onnxruntime-gpu (conf 0.25 / IoU 0.45 fixed at export)

## Cluster (Pinnacles)

```bash
# one-time env (login node — compute nodes may have no internet)
conda create -n yolo python=3.10 -y
conda activate yolo
pip install ultralytics onnx onnxslim onnxruntime

# one-time: pre-download pretrained weights + fonts while online
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11s.pt')"

# one-time: build the 4-class dataset (expects annotations/, train2017/, val2017/ under --coco-root)
python prepare_dataset.py \
    --coco-root /home/anindra/data/ObjectDetection/yolo11Dataset \
    --dst /home/anindra/data/ObjectDetection/yolo11Dataset/coco4
# sanity: train counts must be ~ person 257k / vehicle 68k / traffic-light 12.8k / stop-sign 2.0k

mkdir -p logs
sbatch yolo11n.sh   # ~13-16 h for 150 epochs (single L40S; less with 2)
sbatch yolo11s.sh   # ~25-30 h
```

Watch progress: `tail -f logs/yolo11n-<jobid>.out` (per-epoch mAP table) and
`runs/yolo11n_coco4/results.csv`. If a job dies at walltime, resubmit with
`--resume runs/yolo11<size>_coco4/weights/last.pt`.

## Jetson Orin Nano Super deployment

TensorRT engines are device-specific — always build them **on the Jetson**, from the ONNX.

```bash
# JetPack 6: NVIDIA's Jetson wheel (the PyPI onnxruntime-gpu is x86-only)
pip3 install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# easiest path: onnxruntime with the TensorRT provider (FP16 + engine cache)
python3 jetson_infer_onnx.py --model models/yolo11n_coco4/run1/yolo11n_coco4_nms.onnx \
    --source 0 --show
# first run builds the engine (minutes); it's cached in trt_cache/ afterwards
```

Max-performance alternative (pure TensorRT, no python overhead in the engine):

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo11n_coco4/run1/yolo11n_coco4.onnx \
    --fp16 --saveEngine=yolo11n_coco4_fp16.engine
# then decode (1,8,8400) + NMS yourself, or use ultralytics on the Jetson:
# yolo export model=models/yolo11n_coco4/run1/yolo11n_coco4.pt format=engine half=True
```

Expected on the Orin Nano Super at 640×640 FP16: yolo11n ~80-100+ FPS, yolo11s ~50-60 FPS
(leaves headroom to run LaneATT in parallel; consider running YOLO every 2nd-3rd frame).

## Local smoke test (Mac, conda env `yolo`)

```bash
python prepare_dataset.py --coco-root dataset --dst dataset/coco4
python train.py --size n --epochs 1 --batch 8 --device mps --fraction 0.002 --workers 4
python jetson_infer_onnx.py --model models/yolo11n_coco4/run1/yolo11n_coco4_nms.onnx \
    --source ../LaneATT/video_input/IMG_5106.MOV --out /tmp/yolo_check.mp4 --frames 60
```
