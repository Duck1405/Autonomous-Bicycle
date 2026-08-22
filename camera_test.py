from jetson_tools.trt_runner import CudaRT, TrtEngine
from jetson_tools.postprocess import laneatt_decode
import cv2, numpy as np

cuda = CudaRT()
eng = TrtEngine("/home/mlc/aman/Autonomous-Bicycle/LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine", cuda)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to exit the video stream.")

while True:
    # 2. Capture frame-by-frame
    # ret is a boolean (True if frame reading succeeded), frame is the image array
    ret, frame = cap.read()

    # If the frame was not grabbed properly, break the loop
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # 3. Display the resulting frame in a window named 'Camera Input'
    img = cv2.resize(frame, (640, 360)).astype(np.float32) / 255.0
    arr = np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    out = eng.infer(arr)
    raw = next(iter(out.values()))
    print("raw engine output shape:", raw.shape)      # (1, 1000, 77)

    lanes = laneatt_decode(raw)
    print("num lanes:", len(lanes))
    for l in lanes:
        print(l["points"].shape, l["conf"])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. When everything is done, release the capture handle and destroy windows
cap.release()
cv2.destroyAllWindows()
