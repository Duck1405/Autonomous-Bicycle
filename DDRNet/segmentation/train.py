import math
import re
import torch
import cv2
import time as time
import logging
from pathlib import Path

from lib.LaneATT import LaneATTInference
from lib.lanenet_infer import LaneNetInference
from lib.yolo import YoloInference
from lib.depth import DepthInference
from lib.angle import Angle
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from lib.config import Config
import cv2
import torch
from torchvision import transforms


import warnings
from DDRNet_23_slim import DualResNet, BasicBlock

class bdd100kSegmentation:
    def __init__(self, images, json):
        self.images = images
        self.json = json
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
    def __getitem__(self, frame):
        return


class DDRNetTrain:
    def __init__(self,  images, json):
        self.model = model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64, augment=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        if self.device == "cpu":
            print("We are using CPU")

        self.train_loader = bdd100kSegmentation(images = "/Users/amannindra/Projects/Auto/100k_images", json = "/Users/amannindra/Projects/Auto/100k_json")
            
        
           
    def train(self):



