"""Proof of concept: PHYSICAL (metric) depth in meters with Depth-Anything-V2.

The base project (LaneATT/lib/depth.py) uses the RELATIVE model — its output has
no physical scale (larger = closer, but not meters). The metric_depth/ folder
provides encoders fine-tuned to regress depth in METERS directly: the DPT head
ends in Sigmoid and forward() returns  head(x) * max_depth  (see
metric_depth/depth_anything_v2/dpt.py:183). For an OUTDOOR bike we use the
Virtual-KITTI (vkitti) model with max_depth = 80 m.

What this script does:
  1. downloads the metric vkitti checkpoint (once) into ../checkpoints/,
  2. runs it on one real road frame from LaneATT/video_input/1.mp4,
  3. prints per-pixel depth IN METERS at sampled points,
  4. saves a raw .npy (meters) + an annotated side-by-side image.

Run with the Lannet310 env:
    /Users/amannindra/miniconda3/envs/Lannet310/bin/python Try/metric_depth_test.py
"""
import sys
from pathlib import Path

import torch
# Force CPU: no CUDA on this Mac, and DINOv2's bicubic position-embedding interp
# is not implemented on Apple MPS (the same reason lib/depth.py refuses mps).
torch.backends.mps.is_available = lambda: False

import cv2
import numpy as np

REPO = Path("/Users/amannindra/Projects/Auto/Autonomous-Bicycle/Depth-Anything-V2")
METRIC = REPO / "metric_depth"
sys.path.insert(0, str(METRIC))  # import the METRIC depth_anything_v2, not the base one
from depth_anything_v2.dpt import DepthAnythingV2

# ---- configuration ----
ENCODER = "vits"        # small = fast; 'vitb'/'vitl' are more accurate, larger
DATASET = "vkitti"      # 'vkitti' = outdoor (bike); 'hypersim' = indoor
MAX_DEPTH = 80.0        # meters; 80 for outdoor vkitti, 20 for indoor hypersim
INPUT_SIZE = 518        # multiple of 14

HF_REPO = "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small"
CKPT_NAME = f"depth_anything_v2_metric_{DATASET}_{ENCODER}.pth"
CKPT_PATH = REPO / "checkpoints" / CKPT_NAME

VIDEO = REPO.parent / "LaneATT" / "video_input" / "1.mp4"
FRAME_IDX = 100
OUT = REPO / "Try" / "outputs"

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def ensure_checkpoint():
    """Download the metric checkpoint once (the local .pth is the RELATIVE model,
    which would give meaningless meters here)."""
    if CKPT_PATH.exists():
        print(f"[ckpt] present: {CKPT_PATH}")
        return
    print(f"[ckpt] downloading {CKPT_NAME} from HF '{HF_REPO}' ...")
    from huggingface_hub import hf_hub_download
    import shutil
    cached = hf_hub_download(repo_id=HF_REPO, filename=CKPT_NAME)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, CKPT_PATH)
    print(f"[ckpt] saved -> {CKPT_PATH}")


def load_model():
    model = DepthAnythingV2(**{**MODEL_CONFIGS[ENCODER], "max_depth": MAX_DEPTH})
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    return model.to("cpu").eval()


def get_frame():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {VIDEO}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {FRAME_IDX} from {VIDEO}")
    return frame


def colorize(depth_m):
    """Meters -> INFERNO image, inverted so CLOSE = bright (matches depth2.py)."""
    norm = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-8)
    u8 = ((1.0 - norm) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_checkpoint()
    model = load_model()

    frame = get_frame()
    h, w = frame.shape[:2]
    print(f"[frame] {VIDEO.name}#{FRAME_IDX}  {w}x{h}")

    depth_m = model.infer_image(frame, INPUT_SIZE)  # HxW numpy, METERS
    print(f"[depth] shape={depth_m.shape}  range=[{depth_m.min():.2f}, "
          f"{depth_m.max():.2f}] m  median={np.median(depth_m):.2f} m")

    # Sample the center column top->bottom: sky/horizon (far) down to road (near).
    cx = w // 2
    print("\n[samples] metric depth down the center column:")
    picks = [0.15, 0.35, 0.50, 0.65, 0.80, 0.95]
    for frac in picks:
        y = int(h * frac)
        print(f"   y={frac:0.2f}·H (row {y:4d}):  {depth_m[y, cx]:6.2f} m")

    # Save raw meters (full precision) and an annotated visualization.
    np.save(OUT / f"depth_meters_frame{FRAME_IDX}.npy", depth_m)
    vis = colorize(depth_m)
    for frac in [0.35, 0.50, 0.65, 0.80]:
        y = int(h * frac)
        cv2.circle(vis, (cx, y), 6, (255, 255, 255), -1)
        cv2.putText(vis, f"{depth_m[y, cx]:.1f} m", (cx + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    panel = np.hstack([frame, np.ones((h, 20, 3), np.uint8) * 255, vis])
    out_png = OUT / f"metric_frame{FRAME_IDX}.png"
    cv2.imwrite(str(out_png), panel)
    print(f"\n[saved] {out_png}")
    print(f"[saved] {OUT / f'depth_meters_frame{FRAME_IDX}.npy'}")


if __name__ == "__main__":
    main()
