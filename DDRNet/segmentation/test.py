import cv2
import time
import torch
from DDRNet_23_slim_eval_speed import DualResNet_imagenet, DualResNet, BasicBlock
from pathlib import Path

device = torch.device('cuda')
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True


model = DualResNet_imagenet(pretrained=True)

model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64)
model.eval()
model.to(device)
iterations = None
# sys.exit()
video_path = "/home/anindra/data/Autonomous-Bicycle/LaneATT/video_input/IMG_5105.mp4"
output_folder = Path("output")
output_name = Path("output.mp4")
final_video_path = output_folder / output_name
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_stream = cv2.VideoWriter(str(final_video_path), fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

input = torch.randn(1, 3, 1024, 2048).cuda()
with torch.no_grad():
    i = 0 
    
    print("Start")
    while i < 10:
        ret, frame = cap.read()
        resized_image = cv2.resize(frame, (1024, 2048), interpolation=cv2.INTER_LINEAR)

        if not ret:
            print("ret failed")    
            break
        
        output = model(frame)
        
        print(type(output), output) 
        i +=1
    

#     if iterations is None:
#         elapsed_time = 0
#         iterations = 100
#         while elapsed_time < 1:
#             torch.cuda.synchronize()
#             torch.cuda.synchronize()
#             t_start = time.time()
#             for _ in range(iterations):
#                 model(input)
#             torch.cuda.synchronize()
#             torch.cuda.synchronize()
#             elapsed_time = time.time() - t_start
#             iterations *= 2
#         FPS = iterations / elapsed_time
#         iterations = int(FPS * 6)

#     print('=========Speed Testing=========')
#     torch.cuda.synchronize()
#     torch.cuda.synchronize()
#     t_start = time.time()
#     for _ in range(iterations):
#         model(input)
#     torch.cuda.synchronize()
#     torch.cuda.synchronize()
#     elapsed_time = time.time() - t_start
#     latency = elapsed_time / iterations * 1000
# torch.cuda.empty_cache()
# FPS = 1000 / latency
# print(FPS)
