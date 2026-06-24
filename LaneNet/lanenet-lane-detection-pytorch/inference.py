import argparse
import math

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation

from model.lanenet.LaneNet import LaneNet


LANE_COLORS_BGR = np.array([
    [0, 0, 255],      # red
    [0, 255, 0],      # green
    [255, 128, 0],    # blue/orange-ish
    [0, 255, 255],    # yellow
    [255, 0, 255],    # magenta
    [255, 255, 0],    # cyan
    [128, 255, 0],
    [128, 128, 0],
    [255, 255, 128],
    [128, 128, 128],
], dtype=np.uint8)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Model weights",
        default="/Users/amannindra/Projects/Autonomous-Bicycle/LaneNet/lanenet-lane-detection-pytorch/trained_models/best_model.pth",
    )
    parser.add_argument("--model_type", default="ENet")
    parser.add_argument("--video_file", default="")
    parser.add_argument("--output_file")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--cluster_radius", type=float, default=None)
    parser.add_argument("--mean_shift_bandwidth", type=float, default=None)
    parser.add_argument("--mean_shift_iters", type=int, default=10)
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--max_lanes", type=int, default=10)
    parser.add_argument("--dilation_iters", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--debug_every", type=int, default=0)
    parser.add_argument(
        "--embedding_activation",
        choices=["raw", "sigmoid"],
        default="raw",
        help="Use sigmoid for old checkpoints trained with sigmoid(instance); use raw for paper-style embeddings.",
    )
    return parser.parse_args()


def load_weights(model, model_path, device):
    try:
        weights = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)


def extract_instance_embedding(outputs, embedding_activation="raw"):
    embedding = outputs.get("instance_embedding")
    if embedding is None:
        embedding = outputs["instance_seg_logits"]

    if embedding_activation == "sigmoid":
        embedding = torch.sigmoid(embedding)

    return embedding.detach().to("cpu")[0].numpy().astype(np.float32)


def mean_shift_center(embeddings, center, bandwidth, max_iters):
    center = center.astype(np.float32, copy=True)

    for _ in range(max_iters):
        distances = np.linalg.norm(embeddings - center, axis=1)
        neighbors = embeddings[distances <= bandwidth]
        if len(neighbors) == 0:
            break

        next_center = np.mean(neighbors, axis=0)
        if np.linalg.norm(next_center - center) < 1e-3:
            center = next_center
            break
        center = next_center

    return center


def cluster_lane_embeddings(binary_pred, instance_embedding, delta_v=0.5,
                            cluster_radius=None, mean_shift_bandwidth=None,
                            mean_shift_iters=10, min_cluster_size=50,
                            max_lanes=10):
    lane_mask = binary_pred == 1
    cluster_labels = np.zeros(binary_pred.shape, dtype=np.int32)
    if not np.any(lane_mask):
        return cluster_labels

    if instance_embedding.shape[1:] != binary_pred.shape:
        raise ValueError(
            "Instance embedding shape {} does not match binary mask shape {}".format(
                instance_embedding.shape,
                binary_pred.shape,
            )
        )

    radius = 2.0 * delta_v if cluster_radius is None else cluster_radius
    bandwidth = radius if mean_shift_bandwidth is None else mean_shift_bandwidth

    lane_ys, lane_xs = np.where(lane_mask)
    lane_embeddings = instance_embedding[:, lane_ys, lane_xs].T
    remaining = np.ones(lane_embeddings.shape[0], dtype=bool)

    rng = np.random.default_rng(0)
    cluster_id = 1

    while np.any(remaining) and cluster_id <= max_lanes:
        remaining_indices = np.flatnonzero(remaining)
        seed_index = rng.choice(remaining_indices)

        center = mean_shift_center(
            lane_embeddings[remaining],
            lane_embeddings[seed_index],
            bandwidth,
            mean_shift_iters,
        )

        distances = np.linalg.norm(lane_embeddings - center, axis=1)
        cluster = (distances <= radius) & remaining
        cluster_size = np.count_nonzero(cluster)

        if cluster_size < min_cluster_size:
            if cluster_size > 0:
                remaining[cluster] = False
            else:
                remaining[seed_index] = False
            continue

        cluster_labels[lane_ys[cluster], lane_xs[cluster]] = cluster_id
        remaining[cluster] = False
        cluster_id += 1

    return sort_lane_clusters_left_to_right(cluster_labels)


def sort_lane_clusters_left_to_right(cluster_labels):
    sorted_labels = np.zeros_like(cluster_labels)
    lane_ids = [lane_id for lane_id in np.unique(cluster_labels) if lane_id != 0]

    lane_positions = []
    for lane_id in lane_ids:
        ys, xs = np.where(cluster_labels == lane_id)
        if len(xs) == 0:
            continue
        lower_half = ys >= np.percentile(ys, 50)
        x_position = np.mean(xs[lower_half]) if np.any(lower_half) else np.mean(xs)
        lane_positions.append((x_position, lane_id))

    for new_id, (_, old_id) in enumerate(sorted(lane_positions), start=1):
        sorted_labels[cluster_labels == old_id] = new_id

    return sorted_labels


def draw_lane_clusters(frame_bgr, cluster_labels, dilation_iters=2):
    orig_h, orig_w = frame_bgr.shape[:2]
    cluster_labels_orig = cv2.resize(
        cluster_labels.astype(np.int32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = frame_bgr.copy()
    lane_ids = [lane_id for lane_id in np.unique(cluster_labels_orig) if lane_id != 0]

    for lane_index, lane_id in enumerate(lane_ids[:len(LANE_COLORS_BGR)]):
        lane_mask = cluster_labels_orig == lane_id
        if dilation_iters > 0:
            lane_mask = binary_dilation(lane_mask, iterations=dilation_iters)
        overlay[lane_mask] = LANE_COLORS_BGR[lane_index]

    return cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)


def frame(model, input_tensor, frame_bgr, device, args):
    input_tensor = torch.unsqueeze(input_tensor, dim=0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)

    binary_pred = outputs["binary_seg_pred"][0, 0].detach().to("cpu").numpy().astype(np.uint8)
    instance_embedding = extract_instance_embedding(
        outputs,
        embedding_activation=args.embedding_activation,
    )
    cluster_labels = cluster_lane_embeddings(
        binary_pred,
        instance_embedding,
        delta_v=args.delta_v,
        cluster_radius=args.cluster_radius,
        mean_shift_bandwidth=args.mean_shift_bandwidth,
        mean_shift_iters=args.mean_shift_iters,
        min_cluster_size=args.min_cluster_size,
        max_lanes=args.max_lanes,
    )

    return draw_lane_clusters(
        frame_bgr,
        cluster_labels,
        dilation_iters=args.dilation_iters,
    ), cluster_labels


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    model = LaneNet(arch=args.model_type)
    load_weights(model, args.model, device)
    model.eval()
    model.to(device)

    transform = A.Compose([
        A.Resize(height=args.height, width=args.width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    video_cap = cv2.VideoCapture(args.video_file)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS: {}".format(fps))
    print("Width: {}".format(width))
    print("Height: {}".format(height))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))

    count = 0
    progress_interval = max(math.floor(total_frames / 25), 1)

    while True:
        success, image_bgr = video_cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image=image_rgb)["image"]
        output_frame, cluster_labels = frame(model, input_tensor, image_bgr, device, args)

        count += 1
        video_writer.write(output_frame)

        if args.debug_every > 0 and count % args.debug_every == 0:
            lane_pixels = int(np.count_nonzero(cluster_labels))
            lane_count = len([lane_id for lane_id in np.unique(cluster_labels) if lane_id != 0])
            print("Frame {}: {} clustered lane pixels, {} lanes".format(count, lane_pixels, lane_count))

        if args.max_frames > 0 and count >= args.max_frames:
            break
        if count % progress_interval == 0:
            print("Count: {}/{}".format(count, total_frames))

    video_cap.release()
    video_writer.release()
    print("Video successfully generated!")


if __name__ == "__main__":
    main(get_args())
