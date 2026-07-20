import cv2
import torch
from PIL import Image
from transformers import pipeline

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthInference():
    """Monocular depth via Depth-Anything-V2 (transformers pipeline).

    Output is RELATIVE depth (larger value = closer), not meters — turning it
    into a distance needs a scale reference (e.g. lane width or a known box).
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.device = device
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def infer(self, frame):
        """frame: BGR numpy array (as read by cv2). Returns the pipeline dict:
        'predicted_depth' (raw torch tensor at model resolution) and 'depth'
        (PIL image, 0-255 normalized, resized back to the input size)."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.pipe(image)
