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
from model.lanenet.backbone.H_Net import H_Net, build_H
from lane_utils import (
    cluster_lane_embeddings,
    fit_lane_polynomials,
    draw_lane_clusters,
    draw_all_lane_curves,
)

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
    p.add_argument("--model", default=None)
    p.add_argument("--model_type", default="ENet")
    p.add_argument("--hnet_model", default=None,
                   help="H-Net weights. When supplied, polynomial curves are drawn.")
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
    p.add_argument("--max_lanes", type=int, default=10)
    p.add_argument("--dilation_iters", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--debug_every", type=int, default=0)
    p.add_argument("--debug", action="store_true",
                   help="Write a 2x2 debug video: ENet binary, ENet clusters, "
                        "H-Net curves, and a per-frame info panel.")
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


def _extract_embedding(outputs, activation):
    # Cluster on the RAW instance embedding — that is the space the
    # discriminative loss was trained in. 'instance_seg_logits' is already
    # sigmoid(instance); clustering on it (or applying sigmoid here) squashes
    # every embedding into a tiny range and collapses all lanes into 1 cluster.
    emb = outputs.get("instance_embedding")
    if emb is None:
        emb = outputs["instance_seg_logits"]
    if activation == "sigmoid":
        emb = torch.sigmoid(emb)
    return emb.detach().cpu()[0].numpy().astype(np.float32)


def _panel(img, label, size):
    """Resize an image to `size` (w, h), ensure 3 channels, draw a label."""
    p = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    if p.ndim == 2:
        p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(p, (0, 0), (size[0] - 1, 26), (0, 0, 0), -1)
    cv2.putText(p, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return p


def warp_to_bev(frame_bgr, H_norm):
    """Warp the frame with the normalized-space homography H_norm (3x3) so you
    can SEE the transform H-Net learned. Near-identity H -> looks almost
    unchanged; a degenerate H -> heavily distorted or mostly black."""
    h, w = frame_bgr.shape[:2]
    D     = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]], dtype=np.float64)
    D_inv = np.array([[w, 0, 0], [0, h, 0], [0, 0, 1]], dtype=np.float64)
    H_px = D_inv @ H_norm.astype(np.float64) @ D
    try:
        return cv2.warpPerspective(frame_bgr, H_px, (w, h))
    except cv2.error:
        return np.zeros_like(frame_bgr)


def make_debug_composite(frame_bgr, binary_pred, cluster_labels, curves_frame,
                         info_lines, bev_frame=None):
    """2x2 grid. Returns a frame the same size as the input so the VideoWriter
    dimensions don't change.

        ENet instance clusters | ENet binary seg
        H-Net curves           | H-Net BEV warp (+ info text overlaid)

    When H-Net is off, the bottom-right is a plain info panel.
    """
    h, w = frame_bgr.shape[:2]
    size = (w // 2, h // 2)

    clusters = draw_lane_clusters(frame_bgr, cluster_labels, dilation_iters=2)
    binary_vis = (binary_pred.astype(np.uint8) * 255)

    if bev_frame is not None:
        br = _panel(bev_frame, "H-Net BEV warp", size)
    else:
        br = _panel(np.zeros((h, w, 3), np.uint8), "info", size)

    # Overlay the info text on the bottom-right panel (green for readability).
    yy = 48
    for line in info_lines:
        cv2.putText(br, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        yy += 22

    top = np.hstack([_panel(clusters, "ENet instance clusters", size),
                     _panel(binary_vis, "ENet binary seg", size)])
    bot = np.hstack([_panel(curves_frame, "H-Net curves", size), br])
    composite = np.vstack([top, bot])
    return cv2.resize(composite, (w, h))   # guard against odd-dimension rounding


def process_frame(lanenet, hnet, input_tensor, hnet_tensor, frame_bgr, device, args):
    # LaneNet forward
    with torch.no_grad():
        outputs = lanenet(torch.unsqueeze(input_tensor, 0).to(device))

    if args.binary_threshold is not None:
        lane_prob = torch.softmax(outputs["binary_seg_logits"], dim=1)[0, 1]
        binary_pred = (lane_prob > args.binary_threshold).detach().cpu().numpy().astype(np.uint8)
    else:
        binary_pred = outputs["binary_seg_pred"][0, 0].detach().cpu().numpy().astype(np.uint8)
    embedding   = _extract_embedding(outputs, args.embedding_activation)

    cluster_labels = cluster_lane_embeddings(
        binary_pred, embedding,
        delta_v=args.delta_v, cluster_radius=args.cluster_radius,
        mean_shift_bandwidth=args.mean_shift_bandwidth,
        mean_shift_iters=args.mean_shift_iters,
        min_cluster_size=args.min_cluster_size, max_lanes=args.max_lanes,
    )

    n_clusters = len([l for l in np.unique(cluster_labels) if l != 0])

    H = None
    if hnet is not None:
        with torch.no_grad():
            params = hnet(torch.unsqueeze(hnet_tensor, 0).to(device))
        H = build_H(params)[0]

    polys, curves_norm = {}, {}
    if hnet is not None or args.debug:
        polys, _, curves_norm = fit_lane_polynomials(
            cluster_labels, H, args.width, args.height, args.hnet_poly_order)

    if hnet is None:
        out = draw_lane_clusters(frame_bgr, cluster_labels, args.dilation_iters)
    else:
        out = draw_all_lane_curves(frame_bgr, curves_norm, args.hnet_curve_thickness)

    if args.debug:
        info_lines = [
            "lane px:        {}".format(int(np.count_nonzero(binary_pred))),
            "clusters found: {}".format(n_clusters),
            "polys fit:      {}".format(len(polys)),
            "dropped:        {}".format(n_clusters - len(polys)),
        ]
        bev = None
        if H is not None:
            Hn = H.detach().cpu().numpy()
            bev = warp_to_bev(frame_bgr, Hn)
            info_lines += ["H = [{:+.2f} {:+.2f} {:+.2f}]".format(*Hn[0]),
                           "    [{:+.2f} {:+.2f} {:+.2f}]".format(*Hn[1]),
                           "    [{:+.2f} {:+.2f} {:+.2f}]".format(*Hn[2]),
                           "(identity = 1,0,0 / 0,1,0 / 0,0,1)"]
        else:
            info_lines.append("H-Net: off")
        out = make_debug_composite(frame_bgr, binary_pred, cluster_labels,
                                   out, info_lines, bev_frame=bev)

    return out, cluster_labels


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lanenet = LaneNet(arch=args.model_type)
    _load_weights(lanenet, args.model, device)
    lanenet.eval().to(device)

    hnet = None
    if args.hnet_model:
        hnet = H_Net()
        _load_weights(hnet, args.hnet_model, device)
        hnet.eval().to(device)
        print("H-Net:", args.hnet_model)

    lane_tf = A.Compose([
        A.Resize(args.height, args.width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    hnet_tf = A.Compose([
        A.Resize(args.hnet_height, args.hnet_width),
        ToTensorV2(),
    ]) if hnet else None

    cap    = cv2.VideoCapture(args.video_file)
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

    writer = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    count  = 0
    prog   = max(math.floor(total / 25), 1)

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lt  = lane_tf(image=rgb)["image"]
        ht  = hnet_tf(image=rgb)["image"] if hnet_tf else None

        out, labels = process_frame(lanenet, hnet, lt, ht, rgb, device, args)
        writer.write(out)
        count += 1

        if args.debug_every > 0 and count % args.debug_every == 0:
            n = len([l for l in np.unique(labels) if l != 0])
            print("Frame {}: {} lanes, {} px".format(count, n, np.count_nonzero(labels)))
        if args.max_frames > 0 and count >= args.max_frames:
            break
        if count % prog == 0:
            print("{}/{}".format(count, total))

    cap.release()
    writer.release()
    print("Done ->", file_output)
    save_run_log(args)


if __name__ == "__main__":
    main(get_args())
