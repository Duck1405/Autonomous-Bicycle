"""Time ONLY the Depth-Anything-V2 model (lib/depth.py) over a video.

Runs depth on frames [START_FRAME, END_FRAME) of VIDEO, prints per-frame and
average inference time in ms, and saves the colorized depth maps as a video at
depth_output/<video_stem>/run<K>/depth_<start>-<end>.mp4 (bright = close, dark = far).
No lanes, no YOLO — just the depth model.

Usage (from the LaneATT folder):
    python depth2.py
"""
import re
import time
from pathlib import Path

import cv2
import numpy as np

import time 
from lib.depth import DepthInference



def run_depth_on_video(video_path, start_frame, end_frame, out):
    """Run depth on frames [start_frame, end_frame) of a video and save the
    colorized depth maps as <out>/<video_stem>/run<K>/depth_<start>-<end>.mp4.
    start_frame is inclusive, end_frame is exclusive; end_frame=None runs to the
    end of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Clamp the requested window to what the video actually has.
    end = total_frames if end_frame is None else min(end_frame, total_frames)
    if not (0 <= start_frame < total_frames):
        cap.release()
        raise ValueError(f"start_frame {start_frame} out of range (video has {total_frames} frames)")
    if end <= start_frame:
        cap.release()
        raise ValueError(f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})")
    n_frames = end - start_frame
    print(f"video: {video_path} ({total_frames} frames), running depth on "
          f"frames {start_frame}..{end - 1} ({n_frames} frames)")

    # depth_output/<video_stem>/run<K>/ — same numbering convention as video.py.
    video_folder = Path(out) / Path(video_path).stem
    video_folder.mkdir(parents=True, exist_ok=True)
    existing_runs = [int(m.group(1)) for d in video_folder.iterdir()
                     if d.is_dir() and (m := re.fullmatch(r'run(\d+)', d.name))]
    folder_path = video_folder / f"run{max(existing_runs, default=0) + 1}"
    folder_path.mkdir()
    out_path = folder_path / f"depth_{start_frame}-{end}.mp4"
    print(f"Output Located: {out_path}")

    depth = DepthInference()
    print(f"model: {depth.checkpoint}, device: {depth.device}")

    # Seek directly to the first frame we care about instead of decoding 0..start.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_stream = None
    times_ms = []
    t_start = time.time()
    for offset in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        i = start_frame + offset   # absolute frame index in the video

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
        print(f"\rFrame {i} ({offset + 1}/{n_frames}): {ms:.0f} ms, "
              f"{(offset + 1) / (time.time() - t_start):.2f} FPS", end="", flush=True)
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

def depth_run_on_image(video_path, frame, out):
    """Run depth on a SINGLE frame of a video and save the colorized depth map
    as a PNG at <out>/<video_stem>/run<K>/depth_frame<frame>.png
    (bright = close, dark = far)."""
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
    cap.release()
    if not ret:
        raise RuntimeError(f"could not read frame {frame} from {video_path}")

    # <out>/<video_stem>/run<K>/ — same numbering convention as run_depth_on_video.
    video_folder = Path(out) / Path(video_path).stem
    video_folder.mkdir(parents=True, exist_ok=True)
    existing_runs = [int(m.group(1)) for d in video_folder.iterdir()
                     if d.is_dir() and (m := re.fullmatch(r'run(\d+)', d.name))]
    folder_path = video_folder / f"run{max(existing_runs, default=0) + 1}"
    folder_path.mkdir()
    out_path = folder_path / f"depth_frame{frame}.png"

    depth = DepthInference()
    print(f"model: {depth.checkpoint}, device: {depth.device}")

    print(f"type: {type(img)}")
    print(f"shape: {img.shape}")
    t0 = time.perf_counter()
    result = depth.infer(img)
    ms = 1000 * (time.perf_counter() - t0)
 
    # print(f"frame {frame}: {ms:.0f} ms")

    # result["depth"] is per-frame normalized 0-255 (255 = closest thing).
    depth_u8 = np.array(result["depth"])
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    # cv2.imwrite(str(out_path), colored)

    print(f"frame {frame}: {ms:.0f} ms")
    print(f"saved: {out_path}")
    return out_path


def convert_video_fps(video_path, target_fps, out):
    """Downsample a video to target_fps by DROPPING frames (keeps every
    src_fps/target_fps-th frame, spread evenly). Resolution is unchanged — only
    the temporal rate drops, so a 60 fps clip at target_fps=30 becomes half the
    frames. Writes <out>/<video_stem>_<target>fps.mp4 and returns its path.

    Note: this only downsamples. It cannot add frames, so target_fps must be
    <= the source fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if target_fps > src_fps:
        cap.release()
        raise ValueError(f"target_fps {target_fps} > source fps {src_fps:.2f}; "
                         "this only drops frames, it can't create new ones")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video_path).stem}_{int(round(target_fps))}fps.mp4"
    print(f"video: {video_path} ({total_frames} frames @ {src_fps:.2f} fps)")
    print(f"downsampling to {target_fps} fps -> {out_path}")

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(target_fps), (width, height))

    # Accumulator: write a frame each time the running source-index passes the
    # next capture point (step = how many source frames per output frame).
    step = src_fps / target_fps
    next_capture = 0.0
    i = 0        # source frames read
    kept = 0     # frames written
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i >= next_capture:
            writer.write(frame)
            kept += 1
            next_capture += step
        i += 1
        print(f"\rread {i}/{total_frames}, kept {kept}", end="", flush=True)
    cap.release()
    writer.release()
    print()
    print(f"done: {i} frames read, {kept} written @ {target_fps} fps "
          f"({time.time() - t_start:.1f}s)")
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    # ---- fps downsample mode (run this first to make a lighter video) ----
    src = Path("video_input") / Path("IMG_6893_30fps.mp4")   # 60 fps
    # start_time = time.perf_counter()
    # out_path = convert_video_fps(src, 30, Path("video_input"))     # -> video_input/IMG_6893_30fps.mp4
    # end_time = time.perf_counter()
    # execution_time = end_time - start_time
    # print(f"Executed in: {execution_time:.6f} seconds")

    # ---- single-frame image mode ----
    # # Pick the video and the frame index to run depth on.
    # image_video = Path("video_input") / Path("IMG_6893.MOV")
    # image_frame = 500          # which frame of the video to inference
    # image_output = Path("depth_output2")
    # depth_run_on_image(image_video, image_frame, image_output)

    # ---- full-vide mode ----
    start_frame = 4400           # inclusive
    end_frame = None           # exclusive; use None to run to the end of the video
    output = Path("depth_output2")
    # filesed = [Path("video_input") / Path('IMG_6893.MOV'), Path("video_input") / Path('IMG_6892.MOV'), Path("video_input") / Path('IMG_6893.MOV')]
    filesed = [src]
    
    for i in filesed:
        print(f"file_name: {i.stem}")
        output2 = output / Path(i.stem)
        print(f"output: {output2}")
        run_depth_on_video(i, start_frame, end_frame, output2)
