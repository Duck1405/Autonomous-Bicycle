from pathlib import Path

from ultralytics import YOLO

# Weights live inside lib/ so loading never depends on the working directory.
# (A bad relative path would make ultralytics silently download a fresh copy.)
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "yolo" / "models" / "yolo11n.pt"

class YoloInference():
    def __init__(self, model_path = None, conf_threshold = 0.5):
        self.model = None
        self.conf_threshold = conf_threshold
        self.load_model(model_path if model_path is not None else DEFAULT_WEIGHTS)

    def load_model(self, model_path):
        self.model = YOLO(model_path)

    def infer(self, image):
        # verbose=False: ultralytics otherwise prints a summary line per frame,
        # flooding the console and the run.log.
        if self.model == None:
            print("Model is None")
            return None
        else:
            return self.model(image, verbose=True, conf=self.conf_threshold)

    def draw(self, frame, results):
        """Draw labeled detection boxes onto frame (BGR) and return it."""
        if results is None:
            return frame
        return results[0].plot(img=frame)
