# Autonomous-Bicycle (Pathfinder) — session bootstrap

**Read `skills.md` first.** It is the authoritative project reference: full
architecture, repo map, cluster/Jetson infrastructure, TensorRT workflow,
measured benchmarks, and known issues. Do not re-derive repo structure.

## Quick state (updated 2026-07-26)

- **Jetson engines are BUILT and VALIDATED** (see `Jetson.txt`, the authoritative
  Jetson reference). All three FP16 engines built on-device in `LaneNet310`,
  IO contracts identical to the A100 engines, and all three pass numeric parity
  against onnxruntime-CPU (depth corr 0.999999 — the documented
  Depth-Anything-V2 FP16 degradation did *not* reproduce).
- **Jetson access**: prefer the LAN IP `mlc@10.0.0.226` (~8 ms) over
  `mlc@ubuntu` via Tailscale (~240 ms, and it dropped out mid-session while the
  box was perfectly healthy). Same host key on both.
- **Jetson benchmark (500 frames, sequential)**: **18.95 FPS at MAXN_SUPER**,
  9.88 FPS at the 15W default (1.92x), vs 71.19 FPS on the A100. No thermal
  throttling (60.7 C peak). Engines are now 73% of the frame budget — the
  bottleneck *inverted* vs the A100, where preprocessing dominated.
- **Jetson is currently at 15W (`nvpmodel -m 0`)**. nvpmodel persists across
  reboots; `jetson_clocks` does not.
- **torch is not installed on the Jetson and is not needed** —
  `jetson_tools/trt_runner.py` does all GPU IO through ctypes + libcudart.

## Quick state (updated 2026-07-24)

- **Cluster access**: `ssh ucmerced` (Mac alias → `anindra@login.rc.ucmerced.edu`,
  key `~/.ssh/id_ed25519_ucmerced`). Compute nodes (e.g. `gnode010`) need an
  active Slurm job (`pam_slurm_adopt`); if blocked, grab a node with the salloc
  fallback line in skills.md (test partition, A100, 1 h). Repo on cluster:
  `/home/anindra/data/Autonomous-Bicycle`.
- **Conda envs (cluster)**: `LaneNet310` = training/inference;
  `LaneNetCuda_12_6` = TensorRT conversion + TRT benchmarking (TRT 10.3.0,
  torch 2.5.0+cu124, opencv-headless). Never build engines in `LaneNet310`.
  Mac env: `Lannet310`.
- **Engines built** (FP16, TRT 10.3, cluster-only — Jetson builds on-device via
  its own `trtexec`): yolo11n_nms, depth_anything_v2_small, LaneATT r34
  model_0013 — paths + IO in skills.md.
- **Latest benchmarks (A100, 500 frames IMG_6540.MOV)**: torch pipeline
  17.87 FPS (no depth); TensorRT sequential all-three-models **71.2 FPS**
  (engines only 184 FPS) via `LaneATT/trt_video_benchmark.py`. CPU preprocess
  is now the bottleneck. ROS2 model-parallelism planned (later).
- **`LaneATT/inference.py`** is argparse-driven: zero-arg = MacBook defaults,
  `--yolo/--videos/...` on the server. One YOLO per invocation (no `set_yolo`).

## Working conventions

- When anything goes wrong while debugging: first failure — explain and keep
  diagnosing; second — stop, report, and wait for Aman's approval before fixing.
- Keep `skills.md` updated with anything new learned about the project.
- Never quote the Jetson SSH password (it sits in `Jetson.txt`); refer to it
  descriptively only.
