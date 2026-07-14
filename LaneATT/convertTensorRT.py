import argparse
import torch
import onnxruntime as ort
from lib.config import Config
from pathlib import Path

import numpy as np
PRECISION = np.float16
PRECESION_8 = np.int8




