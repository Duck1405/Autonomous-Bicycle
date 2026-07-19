from transformers import pipeline
from accelerate import Accelerator
import torch
from PIL import Image
import requests
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

print(f"Device: {device}, and type{type(device)}")
checkpoint = "depth-anything/Depth-Anything-V2-Small-hf"
pipe = pipeline("depth-estimation", model=checkpoint, device=device)


url = "/Users/amannindra/Projects/Auto/100k_images/train/0a0a0b1a-27d9fc44.jpg"
image = Image.open(url)

start_time = time.perf_counter()
predictions = pipe(image)
end_time = time.perf_counter()

time_change = end_time - start_time
print(f"Time elapsed: {time_change} seconds")
print(f"Frames per second: {1/time_change}")
print(type(predictions["depth"]))
predictions["depth"].show()
image.show()