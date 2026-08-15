import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort




one = "LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx"
two = "LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx"


def get_frame(video_path, frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame < 0 or frame >= total_frames:
        cap.release()
        raise ValueError(f"frame {frame} out of range (video has {total_frames} frames)")

    # Seek directly to the requested frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    # cv2.imshow("title", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    cap.release()
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="video_input/1.mp4")
    parser.add_argument("--frame", type=int, default=100)
    
    parser.add_argument("--onnx", default="LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx")
    
    args = parser.parse_args()

    frame = get_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    print(f"frame: {args.video} #{args.frame}  ({w}x{h})")
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

    session = ort.InferenceSession(args.onnx, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Model Input Node Name: {input_name}")
    print(f"Model Output Node Name: {output_name}")
    
    
    
    



    # onnx = DepthInferenceONNX(onnx_path=args.onnx, providers=["CPUExecutionProvider"])

    # # ---- 1. GRAPH FIDELITY: identical pixel_values -> torch vs onnxruntime ----
    # pix = onnx._preprocess(frame)  # (1,3,518,518) float32, exactly what ORT sees
    # torch_model = AutoModelForDepthEstimation.from_pretrained("depth_model").eval()
    # with torch.no_grad():
    #     torch_depth = torch_model(torch.from_numpy(pix)).predicted_depth.numpy()
    # (ort_depth,) = onnx.session.run(None, {onnx.input_name: pix})
    # gdiff = np.abs(torch_depth - ort_depth).max()
    # print(f"\n[1] graph fidelity (same input): max |torch - onnx| = {gdiff:.2e}  "
    #       f"{'PASS' if gdiff < 1e-3 else 'FAIL'}")

    # # ---- 2. END-TO-END: transformers pipeline vs ONNX path on the frame ----
    # pipe_depth = np.array(DepthInference().infer(frame)["depth"], dtype=np.float32)
    # onnx_depth = np.array(onnx.infer(frame)["depth"], dtype=np.float32)
    # corr = np.corrcoef(pipe_depth.ravel(), onnx_depth.ravel())[0, 1]
    # mae = np.abs(pipe_depth - onnx_depth).mean()
    # print(f"[2] end-to-end (0-255 depth): corr = {corr:.4f}, "
    #       f"mean|Δ| = {mae:.2f} / 255")