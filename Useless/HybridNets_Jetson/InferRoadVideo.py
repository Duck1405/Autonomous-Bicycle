from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from video_common import (
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO,
    DEFAULT_WEIGHTS,
    parse_bool,
    preprocess_frame,
)
from road_segmentation_model import build_road_segmentation_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run the Jetson runtime HybridNets road-guidance model on a video")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the packaged HybridNets weights (.pth)")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output video path")
    parser.add_argument(
        "--prediction-output",
        type=str,
        default=None,
        help="Output .npy file that stores one uint8 segmentation mask per video frame",
    )
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--output-width", type=int, default=640, help="Output video width")
    parser.add_argument("--output-height", type=int, default=384, help="Output video height")
    parser.add_argument("--compound-coef", type=int, default=3)
    parser.add_argument("--backbone", type=str, default=None, help="Unsupported in the runtime bundle; leave unset")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--gpu-ids", type=str, default="all", help="Comma-separated CUDA device ids for inference, or 'all'")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per inference batch")
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--ui-scale", type=float, default=1.0, help="Scale factor for the on-video HUD and angle gauge")
    parser.add_argument("--smooth-alpha", type=float, default=0.18, help="EMA smoothing factor for path-heading signals")
    parser.add_argument("--steering-smooth-alpha", type=float, default=0.22, help="EMA smoothing factor for the final Stanley steering command")
    parser.add_argument("--lookahead-ratio", type=float, default=0.62, help="Row ratio used for the center-path lookahead point")
    parser.add_argument("--roi-top-ratio", type=float, default=0.52, help="Top row ratio of the guidance ROI")
    parser.add_argument("--sample-step", type=int, default=6, help="Vertical sampling stride for road-center extraction")
    parser.add_argument("--stanley-gain", type=float, default=1.2, help="Cross-track gain used by the Stanley steering controller")
    parser.add_argument("--stanley-softening", type=float, default=1.0, help="Softening term added to the Stanley speed denominator")
    parser.add_argument("--vehicle-speed-mps", type=float, default=3.0, help="Nominal vehicle speed used by Stanley when live speed is unavailable")
    parser.add_argument("--max-angle", type=float, default=45.0, help="Maximum steering angle shown in the guidance gauge")
    parser.add_argument("--write-mask-video", type=parse_bool, default=False)
    parser.add_argument("--mask-output", type=str, default=None, help="Deprecated alias for --prediction-output")
    return parser


def parse_gpu_ids(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return []
    gpu_ids = []
    for item in value.split(","):
        item = item.strip()
        if item:
            gpu_ids.append(int(item))
    return gpu_ids


def resolve_prediction_output(args: argparse.Namespace, output_path: Path) -> Path:
    prediction_output = args.prediction_output or args.mask_output
    if prediction_output:
        path = Path(prediction_output).expanduser().resolve()
        if path.suffix != ".npy":
            path = path.with_suffix(".npy")
        return path
    return output_path.with_name(output_path.stem + "_predictions.npy")


def resolve_gpu_ids(gpu_ids_arg: str) -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP video inference requires CUDA, but CUDA is not available.")

    available_gpu_count = torch.cuda.device_count()
    requested_gpu_ids = parse_gpu_ids(gpu_ids_arg)
    gpu_ids = [gpu_id for gpu_id in requested_gpu_ids if 0 <= gpu_id < available_gpu_count]
    if not gpu_ids:
        gpu_ids = list(range(available_gpu_count))
    if not gpu_ids:
        raise RuntimeError("No CUDA GPUs are available for inference.")
    return gpu_ids


def load_model(
    weights_path: Path,
    compound_coef: int,
    backbone_name: str | None,
    device: torch.device,
    device_id: int,
    use_ddp: bool,
) -> torch.nn.Module:
    return build_road_segmentation_model(
        weights_path=weights_path,
        compound_coef=compound_coef,
        backbone_name=backbone_name,
        device=device,
        device_id=device_id,
        use_ddp=use_ddp,
    )


def run_segmentation_batch(model: torch.nn.Module, batch_tensor: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.no_grad():
        if batch_tensor.device.type == "cuda" and use_amp:
            with torch.cuda.amp.autocast():
                return model(batch_tensor)
        return model(batch_tensor)


def create_prediction_file(prediction_output_path: Path, total_frames: int, image_height: int, image_width: int) -> None:
    prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_file = np.lib.format.open_memmap(
        prediction_output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, image_height, image_width),
    )
    prediction_file.flush()
    del prediction_file


def frame_range_for_rank(total_frames: int, rank: int, world_size: int) -> tuple[int, int]:
    frames_per_rank = math.ceil(total_frames / world_size)
    start_frame = rank * frames_per_rank
    end_frame = min(start_frame + frames_per_rank, total_frames)
    return start_frame, end_frame


def main(rank: int, world_size: int, gpu_ids: list[int]) -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    prediction_output_path = resolve_prediction_output(args, output_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Parsed arguments: {args}")
        print(f"Using GPU ids: {gpu_ids}")
        print(f"Loading weights: {weights_path}")
        print(f"Input video: {video_path}")
        print(f"Prediction output: {prediction_output_path}")
        print(f"Batch size per process: {args.batch_size}")

    print(f"Rank {rank}: selected device {device}")
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Rank {rank}: AMP enabled: {use_amp}")
    print(f"weights_path: {weights_path}")
    print(f"compound: {args.compound_coef}")
    print(f"Backbone: {args.backbone}")
    print(device)
    print(device_id)
    print(use_ddp)
    model = load_model(weights_path, args.compound_coef, args.backbone, device, device_id, use_ddp)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {video_path}")
    
    if rank == 0:
        create_prediction_file(prediction_output_path, total_frames, args.image_height, args.image_width)
    
    if use_ddp:
        dist.barrier()

    prediction_file = np.lib.format.open_memmap(
        prediction_output_path,
        mode="r+",
        dtype=np.uint8,
        shape=(total_frames, args.image_height, args.image_width),
    )

    start_frame, end_frame = frame_range_for_rank(total_frames, rank, world_size)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Rank {rank}: processing frames {start_frame} through {end_frame - 1}")

    processed_frames = 0
    start_time = time.perf_counter()
    pending_tensors: list[torch.Tensor] = []
    pending_frame_indices: list[int] = []


    def flush_batch():
        nonlocal processed_frames, pending_tensors, pending_frame_indices
        if not pending_tensors:
            return
        batch_tensor = torch.stack(pending_tensors, dim=0).to(device, non_blocking=True)
        if device.type == "cuda":
            batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
        segmentation_logits = run_segmentation_batch(model, batch_tensor, use_amp)
        segmentation_masks = segmentation_logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)

        for original_frame_index, segmentation_mask in zip(pending_frame_indices, segmentation_masks):
            prediction_file[original_frame_index] = segmentation_mask
            processed_frames += 1

        if processed_frames % max(args.batch_size * 10, 1) == 0 or processed_frames >= end_frame - start_frame:
            elapsed = max(time.perf_counter() - start_time, 1e-8)
            print(f"Rank {rank}: processed {processed_frames}/{end_frame - start_frame} frames ({processed_frames / elapsed:.2f} fps)")
                
        pending_tensors = []
        pending_frame_indices = []

    source_frame_index = start_frame

    while source_frame_index < end_frame:
        ok, frame_bgr = capture.read()
        if not ok:
            print(f"Rank {rank}: stopped early at frame {source_frame_index}")
            break

        input_array, _ = preprocess_frame(
            frame_bgr,
            args.image_width,
            args.image_height,
            args.output_width,
            args.output_height,
        )
        
        pending_tensors.append(torch.from_numpy(input_array).float())
        pending_frame_indices.append(source_frame_index)
        if len(pending_tensors) >= args.batch_size:
            flush_batch()
        source_frame_index += 1

    flush_batch()

    elapsed = max(time.perf_counter() - start_time, 1e-8)
    prediction_file.flush()
    del prediction_file
    capture.release()
    if use_ddp:
        dist.barrier()
    if rank == 0:
        print(f"Saved per-frame segmentation masks to {prediction_output_path}")
    print(f"Rank {rank}: finished {processed_frames} frames in {elapsed:.2f}s ({processed_frames / elapsed:.2f} fps)")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser_args = build_parser().parse_args()
    selected_gpu_ids = resolve_gpu_ids(parser_args.gpu_ids)
    world_size = len(selected_gpu_ids)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    print(f"GPUs available: {torch.cuda.device_count()}")
    if world_size == 1:
        print(f"Starting single-GPU inference with GPU id: {selected_gpu_ids[0]}")
        main(0, world_size, selected_gpu_ids)
    else:
        print(f"Starting DDP inference with GPU ids: {selected_gpu_ids}")
        mp.spawn(
            main,
            args=(world_size, selected_gpu_ids),
            nprocs=world_size,
            join=True,
        )
