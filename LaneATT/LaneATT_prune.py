import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from lib.config import Config
from pathlib import Path


MODELSED = [
    # ("experiments/LaneATTresnet18Aug2/config.yaml", "experiments/LaneATTresnet18Aug2/models/model_0019.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt"),
    ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt"),
    # server-side YOLO checkpoints (pass them via --yolo instead of editing this list):
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11m_coco4/run5/yolo11m_coco4.pt"),
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11n_coco4/run7/yolo11n_coco4.pt"),
    # ("experiments/LaneATTresnet34Aug2/config.yaml", "experiments/LaneATTresnet34Aug2/models/model_0013.pt", "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11s_coco4/run5/yolo11s_coco4.pt"),
    # ("experiments/LaneATTresnet50Aug2/config.yaml", "experiments/LaneATTresnet50Aug2/models/model_0015.pt"),
    # ("experiments/LaneATTresnet101Aug2/config.yaml", "experiments/LaneATTresnet101Aug2/models/model_0017.pt"),
    # ("experiments/LaneATTresnet152Aug2/config.yaml", "experiments/LaneATTresnet152Aug2/models/model_0015.pt" "/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Yolov11/runs/yolo11n_coco45/weights/last.pt")
]

def prune_model(config_path, model_path):
    cfg = Config(config_path)
    name = Path(model_path).stem
    model_arch= cfg.get_model()
    print(model_arch)

for yaml, model in MODELSED:
    prune_model(yaml, model)
    
    
    
    
    