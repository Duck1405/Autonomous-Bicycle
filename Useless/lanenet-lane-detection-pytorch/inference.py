import argparse
import json
import math
import os
from datetime import datetime

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from model.lanenet.LaneNet import LaneNet

from pathlib import Path

def save_run_log(args, log_path="inference_runs.json"):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": args.output_file,
        "lanenet_model": args.model,
        "lanenet_arch": args.model_type,
        "hnet_model": args.hnet_model or "none",
        "video_file": args.video_file,
        "width": args.width,
        "height": args.height,
        "embedding_activation": args.embedding_activation,
        "hnet_poly_order": args.hnet_poly_order,
        "hnet_curve_thickness": args.hnet_curve_thickness,
        "delta_v": args.delta_v,
        "min_cluster_size": args.min_cluster_size,
        "max_lanes": args.max_lanes,
        "max_frames": args.max_frames,
    }

    runs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                runs = json.load(f)
            except json.JSONDecodeError:
                runs = []

    runs.append(entry)
    with open(log_path, "w") as f:
        json.dump(runs, f, indent=2)
    print("Run logged to: {}".format(log_path))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="trained_models/best_model.pth")
    p.add_argument("--model_type", default="ENet")
    p.add_argument("--hnet_model", default=None,
                   help="Legacy option; ignored for semantic-mask video output.")
    p.add_argument("--hnet_poly_order", type=int, default=3)
    p.add_argument("--hnet_width",  type=int, default=128)
    p.add_argument("--hnet_height", type=int, default=64)
    p.add_argument("--hnet_curve_thickness", type=int, default=4)
    p.add_argument("--video_file", default=None)
    p.add_argument("--output_file")
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--delta_v", type=float, default=0.5)
    p.add_argument("--cluster_radius", type=float, default=None)
    p.add_argument("--mean_shift_bandwidth", type=float, default=None)
    p.add_argument("--mean_shift_iters", type=int, default=10)
    p.add_argument("--min_cluster_size", type=int, default=50)
    p.add_argument("--max_lanes", type=int, default=4)
    p.add_argument("--dilation_iters", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--debug_every", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="Legacy option; output is always the red semantic-mask overlay.")
    p.add_argument("--embedding_activation", choices=["raw", "sigmoid"], default="raw")
    p.add_argument("--binary_threshold", type=float, default=None,
                   help="If set, build the lane mask from softmax(lane_prob) > threshold "
                        "instead of argmax (0.5). Lower values (e.g. 0.3) capture more "
                        "lane pixels — no retraining needed.")
    return p.parse_args()


def _load_weights(model, path, device):
    print("_load_weights:")
    print(model)
    print(path)
    print(device)
    print("-------")
    
    try:
        w = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        w = torch.load(path, map_location=device)
    model.load_state_dict(w)


def process_frame(lanenet, hnet, input_tensor, hnet_tensor, frame_bgr, device, args):
    """Color lane pixels red; legacy H-Net arguments are unused."""
    with torch.no_grad():
        outputs = lanenet(torch.unsqueeze(input_tensor, 0).to(device))

    if args.binary_threshold is not None:
        lane_prob = torch.softmax(outputs["binary_seg_logits"], dim=1)[0, 1]
        binary_pred = (lane_prob > args.binary_threshold).detach().cpu().numpy().astype(np.uint8)
    else:
        binary_pred = outputs["binary_seg_pred"][0, 0].detach().cpu().numpy().astype(np.uint8)

    # Nearest-neighbor resizing preserves the predicted class labels.
    mask = cv2.resize(binary_pred, (frame_bgr.shape[1], frame_bgr.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    out = frame_bgr.copy()
    out[mask == 1] = (0, 0, 255)  # OpenCV frames use BGR.
    return out, binary_pred


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lanenet = LaneNet(arch=args.model_type)
    _load_weights(lanenet, args.model, device)
    lanenet.eval().to(device)

    lane_tf = A.Compose([
        A.Resize(args.height, args.width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    cap    = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.video_file}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS: {}  Size: {}x{}  Frames: {}".format(fps, w, h, total))

    file_output = Path(args.output_file)
    num = 1
    while os.path.isfile(file_output):
        file_name = file_output.stem
        parent = file_output.parent
        print(f"While Loop: file_output: {file_name}")
        file_name = file_name + "_" + str(num) + file_output.suffix
        file_output = parent / file_name
        num += 1
        
    
    print(f"Confirmed Output File: {file_output}")

    writer = cv2.VideoWriter(str(file_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video: {file_output}")
    count  = 0
    prog   = max(math.floor(total / 25), 1)

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lt  = lane_tf(image=rgb)["image"]
        out, mask = process_frame(lanenet, None, lt, None, bgr, device, args)
        writer.write(np.hstack((bgr, out)))
        count += 1

        if args.debug_every > 0 and count % args.debug_every == 0:
            print("Frame {}: {} lane pixels (model resolution)".format(
                count, np.count_nonzero(mask)))
        if args.max_frames > 0 and count >= args.max_frames:
            break
        if count % prog == 0:
            print("{}/{}".format(count, total))

    cap.release()
    writer.release()
    print("Done ->", file_output)
    args.output_file = str(file_output)
    save_run_log(args)


if __name__ == "__main__":
    main(get_args())
