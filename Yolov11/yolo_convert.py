from ultralytics import YOLO

m = "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11m_coco4/run5/yolo11m_coco4.pt"
n = "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11n_coco4/run7/yolo11n_coco4.pt"
s = "/home/anindra/data/Autonomous-Bicycle/Yolov11/models/yolo11s_coco4/run5/yolo11s_coco4.pt" 

l = [m,n,s]
for i in l:
    model = YOLO(i)
    model.export(format='engine', dynamic=True,half=True,)
    