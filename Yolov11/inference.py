
from ultralytics import YOLO

class YoloInference():
    def __init__(self, model_path = "models/yolo11n.pt"):
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        self.model = YOLO(model_path)
    
    def evaluate_image(self, image):
        if self.model == None:
            print("Model is None")
            return None
        else:
            return self.model(image)
    
