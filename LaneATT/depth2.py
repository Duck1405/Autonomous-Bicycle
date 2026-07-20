"""Time ONLY the Depth-Anything-V2 model (lib/depth.py) over a video.

Runs depth on frames 0..FRAME_LIMIT of VIDEO, prints per-frame and average
inference time in ms, and saves the colorized depth maps as a video at
depth_output/<video_stem>/run<K>/depth.mp4 (bright = close, dark = far).
No lanes, no YOLO — just the depth model.

Usage (from the LaneATT folder):
    python depth2.py
"""
import re
import time
from pathlib import Path

import cv2
import numpy as np

from lib.depth import DepthInference



def run_depth_on_video(video_path, frame_limit, out):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = min(frame_limit, total_frames)
    print(f"video: {video_path} ({total_frames} frames), running depth on {n_frames}")

    # depth_output/<video_stem>/run<K>/ — same numbering convention as video.py.
    video_folder = Path(out) / Path(video_path).stem
    video_folder.mkdir(parents=True, exist_ok=True)
    existing_runs = [int(m.group(1)) for d in video_folder.iterdir()
                     if d.is_dir() and (m := re.fullmatch(r'run(\d+)', d.name))]
    folder_path = video_folder / f"run{max(existing_runs, default=0) + 1}"
    folder_path.mkdir()
    out_path = folder_path / "depth.mp4"
    print(f"Output Located: {out_path}")

    depth = DepthInference()
    print(f"model: {depth.checkpoint}, device: {depth.device}")

    out_stream = None
    times_ms = []
    t_start = time.time()
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = depth.infer(frame)
        ms = 1000 * (time.perf_counter() - t0)
        times_ms.append(ms)

        # result["depth"] is per-frame normalized 0-255 (255 = closest thing in
        # THIS frame, so absolute brightness flickers between frames).
        depth_u8 = np.array(result["depth"])
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        if out_stream is None:
            out_stream = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (colored.shape[1], colored.shape[0]))
        out_stream.write(colored)

        # Live throughput: model + colorize + write, wall clock.
        print(f"\rFrame {i + 1}/{n_frames}: {ms:.0f} ms, "
              f"{(i + 1) / (time.time() - t_start):.2f} FPS", end="", flush=True)
    cap.release()
    if out_stream is not None:
        out_stream.release()
    print()

    # The first frame includes one-time warmup, so it gets reported on its
    # own and left out of the average.
    avg = sum(times_ms[1:]) / len(times_ms[1:]) if len(times_ms) > 1 else times_ms[0]
    print(f"frames run: {len(times_ms)}")
    print(f"first frame (warmup): {times_ms[0]:.0f} ms")
    print(f"average: {avg:.1f} ms/frame  ({1000 / avg:.2f} FPS)")
    print(f"saved: {out_path}")
    return avg


if __name__ == "__main__":
    video = "video_input/1.mp4"
    frame = 1000
    output = Path("depth_output2")
    filesed = [Path("video_input") / Path('IMG_6540.MOV'), Path("video_input") / Path('IMG_6892.MOV'), Path("video_input") / Path('IMG_6893.MOV')]

    for i in filesed:
        print(f"file_name: {i.stem}")
        output2 = output / Path(i.stem)
        print(f"output: {output2}")
        run_depth_on_video(i, frame, output2)
