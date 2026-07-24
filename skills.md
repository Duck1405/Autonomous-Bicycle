---
name: AV
description: Context and working conventions for Aman's real-time autonomous e-bike ("Pathfinder") ADAS perception-to-control pipeline. Target deployment is a Jetson Orin Nano Super, but current dev/validation happens on a Mac (local) and UC Merced's HPC cluster (L40S/A100) — no physical-Jetson benchmarks exist in this repo yet. Covers lane detection (LaneATT), object detection (YOLO11), monocular depth (Depth-Anything-V2, relative and metric variants), Stanley steering + lead-vehicle (CIPV) selection, planned Kalman tracking + IDM control, ONNX/TensorRT export, ESP32-S3 firmware (steering encoder, brake relay, PWM speed — on the unmerged origin/S3-code branch, not on main), and the UC Merced HPC/Vista Compute Lab infrastructure. Use this skill whenever working in this repo, discussing architecture/compute-budget decisions, or reviewing what's actually implemented vs. only planned — even if the user doesn't name a component directly.
---

# Pathfinder — AV Perception-to-Control Pipeline

## Project Goal

End-to-end, real-time ADAS pipeline for "Pathfinder," Aman's 250W autonomous
e-bike (UC Merced), capable of lane keeping and front-collision avoidance.
Target deployment is a **Jetson Orin Nano Super**. Success = every pipeline
stage hitting real-time throughput on-device, with safe behavior in degraded
conditions (night, glare, etc.).

**Where work actually happens today:** local Mac (CPU/MPS, small-scale
testing) + UC Merced's Slurm HPC cluster (L40S for multi-day training jobs,
A100 via the `test` partition for quick interactive throughput checks). No
physical Jetson Orin Nano Super work exists in this repo yet — the only
Jetson-named folder (`HybridNets_Jetson/`) is a dormant, currently-broken
experiment (see Repository Map). Treat every FPS number attributed to "the
Jetson" elsewhere in this doc as a *projection*, not a measurement, unless
explicitly marked otherwise.

Owner: Aman, UC Merced. Also does infra work at UC Merced's Vista Compute Lab
(Kubernetes ML training platform) and is starting LLNL's Data Science
Challenge program (agentic AI pipelines for materials discovery) — that work
is a separate track, not part of this pipeline.

## Hardware Constraint (read this before suggesting any model)

**67 TOPS on the Orin Nano Super is a ceiling, not a floor.** That number is
INT8-sparse only, requires TensorRT quantization + 2:4 structured sparsity to
approach, and real measured throughput is roughly half that in practice
(forum benchmarks show even AGX Orin hitting ~45% of advertised TOPS). FP32/FP16
workloads get no tensor-core benefit. Always reason about compute budget in
these realistic terms, not the spec-sheet number.

**Architecture-class lesson (directionally solid, but the specific number is
unverified):** HybridNets (multi-task, multi-branch, irregular ops) was tried
in `HybridNets_Jetson/` for lane+drivable-area+object detection in one net.
The commonly-cited "~12 FPS after TensorRT conversion" figure does **not**
appear anywhere in the repo — no benchmark log, notebook output, README, or
commit message documents it (checked directly, including parsed notebook
outputs and full git log). What *is* verifiable: the folder's ONNX export
script (`ExportOnnxRuntime.py`) has had a syntax error (an unclosed paren) in
its `torch.onnx.export(...)` call since its last commit on 2026-07-14 and
hasn't been touched since, so no TensorRT benchmark could even run right now.
Treat the architectural conclusion (single-task, standard-op CNNs are the
safer bet for this hardware class) as sound reasoning worth keeping, but
don't repeat "~12 FPS" as a measured fact — say "unverified, and the export
path is currently broken" if asked. Be skeptical of any multi-task/
multi-branch network proposed for this project regardless.

## Current Pipeline State

### Completed and validated

**Lane detection — LaneATT** (`LaneATT/`)
- Active model: `experiments/LaneATTresnet34Aug2/models/model_0013.pt` +
  `experiments/LaneATTresnet34Aug2/config.yaml` (ResNet34 backbone). This is
  what every live entrypoint (`inference.py`, `convertonnx.py`) actually
  points at.
- All 5 backbones (ResNet18/34/50/101/152) were trained and compared at some
  point — `video_output_2/` has inference output for all five — but only
  ResNet18 and ResNet34 have live `experiments/<name>Aug2/` configs +
  checkpoints in this checkout right now; the 50/101/152 experiment folders
  aren't currently present locally (likely cleaned up or never synced from
  the cluster).
- Ego-lane assignment is solved **spatially, per frame**: confidence-
  hysteresis filtering (`LaneATT.py:filter_lanes_hysteresis` — two
  thresholds; a lane needs the high one to be newly acquired but only the low
  one to survive if it matches last frame's position) feeds `get_ego_lanes()`,
  which classifies lanes left/right by bottom-row x vs. image midline and, if
  only one edge is visible, **synthesizes** the missing edge from a learned
  per-row lane-width EMA. This — not tracking IDs — is what avoids the
  clustering-ID instability that plagued LaneNet. Don't suggest
  ID-tracking-based lane assignment.
- CLRNet and UFLD are **not implemented anywhere in this repo** — zero code,
  zero checkpoints. (Earlier notes described them as "comparison/fallback
  models"; that was aspirational. The only CLRNet mention anywhere is inside
  a vendored third-party comparison script in `LaneTCA/`, unrelated to this
  project's own code.)
- NMS: `lib/models/laneatt.py` tries a compiled CUDA extension (`lib/nms/`,
  source present, **not built** in this checkout) and falls back to a
  pure-PyTorch reimplementation (`lib/nms_pytorch.py`) with identical
  similarity-metric semantics. The Python fallback is what's actually running
  locally; verify the CUDA extension is built before assuming it's used on
  the cluster.
- Exported to ONNX, NMS deliberately excluded (data-dependent shape, can't be
  traced): `convertonnx.py` → `onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx`
  (104 MB, output `(1, 1000, 77)`, verified against PyTorch at build time —
  the parity check itself is now commented out in the script, so re-enable it
  if you re-export).

**Object detection — YOLO11** (`Yolov11/`)
- **Class scheme correction: there are two, not "three classes (pedestrians,
  cars)."**
  1. **2-class COCO scheme** (`Preprocess.py` → `dataset/coco4/`): person,
     vehicle (car+motorcycle+bus+truck). **This is
     what's actually been trained** — all three sizes (n/s/m), fully trained
     (150–250 epochs) and exported (`.pt` + raw `.onnx` + NMS-baked
     `_nms.onnx`) — see `Yolov11/runs/yolo11{n,s,m}_coco4*/`. Copies also
     live in `LaneATT/onnxmodels/{YoloN,YolloS,YoloM}/` (the `YolloS` folder
     name has a typo). `jetson_infer_onnx.py` is the Jetson-side ONNX runtime
     counterpart; `onnx_tensorRT.py` is the working cluster-side TensorRT
     conversion tool (see Infrastructure → ONNX → TensorRT conversion).
  2. **2-class BDD100K scheme** (`prepare_bdd100k.py` → `Bdd100Final/`,
     outside the repo one level up): person, vehicle only — traffic-light and
     stop-sign deliberately dropped. Dataset is prepared and confirmed
     ultralytics-loadable (`.cache` files dated Jul 13), **but no model has
     been trained on it yet** — no "bdd" run exists anywhere in `runs/`. This
     is the newer, intended-successor pipeline, not yet acted on.
  - Both schemes keep `vehicle` at class id `1` on purpose —
    `prepare_bdd100k.py`'s own docstring says this is so
    `LaneATT/lib/angle.py`'s `Angle(vehicle_class_id=1)` keeps working
    unchanged regardless of which scheme is loaded.
- TensorRT FP16 is the deployment target. `Yolov11/README.md` states
  *expected* Orin Nano Super throughput (yolo11n ~80–100+ FPS, yolo11s
  ~50–60 FPS at 640×640 FP16) — a documented projection, not a logged
  benchmark; don't cite it as measured.

**Steering + lead-vehicle selection — Stanley + CIPV** (`LaneATT/lib/angle.py`)
- This is **done and wired into the live pipeline**, not "selected but not
  yet integrated" — correct that if it comes up. `Angle.compute_steering()`
  implements classic Stanley (heading term via linear fit through
  near-vehicle midpoints, plus `atan2(gain × cross-track, speed)`) — **no
  curvature/feedforward term exists**; that was an earlier design idea, never
  implemented. `Angle.select_ego_vehicle()` picks the CIPV by testing whether
  a vehicle box's bottom-center falls inside the interpolated left/right
  ego-lane edges, nearest wins.
- Real measured coverage from an A100 cluster run (`IMG_6540.MOV`, 500
  frames, logged in `training_pinnacle.txt`): **500/500 frames** got a
  steering value, **500/500** got a lead-vehicle selection, ego-lane coverage
  500/500 with 0 frames needing synthesis. Production params from that run:
  `hfov=90.0, stanley_gain=1.0, nominal_speed=3.0, hold_decay=0.85,
  max_extrapolation_px=60`.
- **Bug**: `ASSUMED_HFOV_DEG = 90.0` (`angle.py:13`) contradicts its own
  in-code comment, which argues for 70° as the typical wide-webcam/action-cam
  HFOV. Either the constant or the comment is wrong — resolve before trusting
  cross-track-in-degrees output. No camera has ever been calibrated (no
  checkerboard run exists anywhere in the repo), so both numbers are guesses
  regardless.
- EMA smoothing on the steering output is implemented but **explicitly
  disabled** (commented out, with a note to keep the controller simple;
  re-enable if the raw command jitters on real footage).
- `distance_to_ego_vehicle()` is a real stub — returns `None`, never called.
  `VideoInference.speed_eval()` (`video.py`) is also a bare stub, never
  called — presumably where an ego wheel-speed reading would eventually be
  consumed.

### Active focus: distance & speed estimation for front-collision avoidance

**This section was stale in earlier notes — dense monocular depth is not
ruled out, it's the thing actually being built right now.**

- `LaneATT/lib/depth.py` wraps **Depth-Anything-V2-Small-hf** (relative
  depth) via the `transformers` pipeline, with local caching (`depth_model/`)
  and deliberate MPS avoidance (bicubic upsample isn't implemented on Apple's
  MPS backend — forces cuda/cpu only). It's wired into `VideoInference`
  (`video.py`) and exported to ONNX with a self-checking parity test
  (`convert_depth_onnx.py` → `onnxmodels/depth_onnx/depth_anything_v2_small.onnx`,
  95 MB, `test_depth_onnx.py` → `parity_frame100.png`).
- **Separately, and more actively**, the vendored `Depth-Anything-V2/` folder
  (metric variants, not the small relative model above) is the single
  most-recently-touched folder in the entire repo — a metric-outdoor
  checkpoint trained on **Virtual KITTI 2**
  (`checkpoints/depth_anything_v2_metric_vkitti_vits.pth`) was downloaded and
  tested (`Try/metric_depth_test.py`, `Try/outputs/{metric_frame100.png,
  depth_meters_frame100.npy}`) as recently as **yesterday**. This is the live
  thread: getting *metric* (not just relative) monocular depth working.
- **Real, measured cost of turning depth on** — same A100, same clip,
  `training_pinnacle.txt`:

  | Config | Pipeline throughput | Per-frame cost |
  |---|---|---|
  | LaneATT + YOLO (cuda) | 18.63 FPS (54 ms/frame) | LaneATT 22ms, YOLO 12ms |
  | LaneATT + YOLO + Depth-Anything-V2-Small (cuda) | 8.77 FPS (114 ms/frame) | LaneATT 26ms, YOLO 12ms, **Depth 56ms** |

  Depth roughly **doubles** wall-clock cost, and dominates it — on an A100.
  This is almost certainly *why* `video.py`'s main loop currently has the
  `depth_results = self.depth.infer(frame)` call **commented out**
  (`video_eval()`; depth still runs in `image_eval()`'s single-frame path,
  where the result is computed but not yet consumed downstream). If you
  re-enable it, expect this cost, and expect it to matter a lot more on a
  Jetson than an A100.
- **The concrete downstream plan** (from `depthandspeedestimationresearch.md`,
  a ~135-citation lit review compiled this month specifically for this
  problem — read it directly for citations):
  ```
  Z_t   = per-box metric depth (median inside the YOLO box, or a future
          per-object regressor)
  v_rel = Kalman-filtered dZ/dt   (state: [Z, Ż], constant-velocity model)
  v_car = v_ego (wheel/hall sensor) + v_rel
  TTC   = Z / max(-Ż, ε)
  ```
  A 10-frame Kalman window is expected to bring per-frame depth noise
  (σ_Z ≈ 0.3 m → raw σ_v ≈ 9 m/s, useless) down to ≈0.5–1 m/s — matching
  literature systems' ±1–3 km/h. **Ego-speed is settled as a hardware
  problem, not a vision one** — the research doc's own verdict is to use a
  wheel/hall sensor and treat vision-based ego-speed as a research toy; this
  lines up with the ESP32 firmware's `PIN_PWM_SPEED` (see Firmware, below).
  TTC is flagged as possibly the more useful safety quantity for a bike
  specifically because it needs no absolute depth scale at all (Z and Ż
  ratios cancel).
- **Stereo was considered and set aside.** The research doc's own "ship this
  month" recommendation (Tier A) is actually stereo-camera-based (OAK-D-class
  depth + YOLO + wheel sensor, no monocular network at all, metric by
  construction) — but Aman has ruled out the stereo camera for this build.
  Current code direction has moved to the doc's Tier B (dense monocular
  *metric* depth) instead, which is what `Depth-Anything-V2/Try/` is
  actively testing.
- **RT-MonoDepth is a literature candidate, not "the current strongest
  candidate"** — zero implementation anywhere in the repo. It appears once,
  in the research doc's edge-deployment survey (§9), alongside FastDepth,
  ZipDepth, AsyncMDE, and a DA3 ROS2/TensorRT wrapper — one option among
  several, not a decision that's been made.
- **A cheaper, not-yet-tried alternative worth knowing about**: Dist-YOLO
  (add a distance channel directly to YOLO's detection head, near-zero extra
  latency since it shares the backbone) is called out in the research doc as
  "the single most direct solution for your current stack." Nobody has built
  this yet, but it's a real, low-cost alternative to dense depth if the
  Depth-Anything-V2 latency cost above doesn't fit the Jetson budget.
- **Rule of thumb, still valid**: most lightweight depth literature outputs
  *relative*, not *metric*, depth. Always verify metric output explicitly
  before proposing a model here.

### Selected but not yet integrated
- **IDM (Intelligent Driver Model)** for longitudinal control, consuming the
  `(d, Δv)` pipeline above (`a_IDM = f(v, Δv, d)`; background paper:
  `2506.05909v1.pdf`, "Twenty-Five Years of the Intelligent Driver Model," at
  the repo top level). No Kalman tracker exists yet in this codebase (the
  only Kalman code anywhere in the repo is generic Ultralytics
  multi-object-tracker code vendored inside `yolov12/`, unrelated to this
  project's own tracking) — it's a hard prerequisite for IDM's Δv term, not
  optional. See the math above.
- Longer-term upgrade path: MPC over Stanley, optimizing over the full
  polynomial horizon with kinematic bicycle model forward simulation. Not
  current priority — don't push this over finishing the base pipeline.

### Known gaps — not yet built (the "connective tissue")
1. ~~Object-lane association~~ — **done**, see `select_ego_vehicle()` above.
2. **Kalman-based object tracking** — produces the velocity estimate IDM
   needs; see the `[Z, Ż]` state-space above.
3. **Behavioral branching layer** — distinct handling for cars vs.
   pedestrians vs. traffic lights.
4. **Actuation layer** — converting a desired acceleration into
   throttle/brake commands. The **low-level plumbing already exists**: the
   ESP32-S3 firmware (unmerged `origin/S3-code` branch) has a dedicated brake
   relay pin, a PWM speed pin, and a servo/ESC steering output already wired
   to real pins (see Firmware, below). What's missing is the Python/Jetson
   side — nothing in the working tree currently talks to the ESP32 at all.
5. **Cross-model latency synchronization** — aligning independently-timed
   model outputs (lane, detection, depth) into one control loop. Now backed
   by real numbers: LaneATT ~22–26 ms, YOLO ~12 ms, Depth ~56 ms per frame on
   an A100 (see table above) — depth is the long pole.

### Known bugs / rough edges (current, as of this pass)
- `video.py:video_eval()` will **crash** (`AttributeError`) if run with
  `model_type="laneNet"` and `view=True` — the correctly-branched drawing
  code is commented out, and the live code unconditionally calls
  `self.laneatt.lanes_to_px(...)`, which is `None` in LaneNet mode.
- `lib/yolo.py:infer()` — comment claims `verbose=False` to avoid flooding
  logs; the actual call passes `verbose=True`.
- `depthInference.py` (renamed from `depth2.py` — docstring and
  `__pycache__` still say the old name) — `depth_run_on_image()`'s
  `cv2.imwrite(...)` line is commented out, but the function still logs
  `"saved: ..."` right after. No file is actually written.
- The depth-ONNX default path is inconsistent across three files
  (`lib/depth.py`'s `DEFAULT_ONNX`, `convert_depth_onnx.py`'s `DEFAULT_OUT`,
  `test_depth_onnx.py`'s `ONNX_PATH` all say bare `depth_onnx/...`) — the
  real file lives at `onnxmodels/depth_onnx/...`. Running any of the three
  with no explicit override will fail to find it.
- `LaneATT/inference.py` has no `if __name__ == "__main__":` guard (runs at
  import time) and its active `MODELSED` list hardcodes YOLO checkpoint paths
  to a different (Linux/cluster) machine (`/home/anindra/data/...`); its
  skip-check verifies the LaneATT checkpoint exists but never checks the YOLO
  path, so on a machine without that path it fails instead of skipping.
- `LaneATT/requirements.txt` is essentially the untouched upstream LaneATT
  file — missing `torch`, `ultralytics`, `transformers`, `onnxruntime`,
  `albumentations`, `pandas`: every dependency the live pipeline actually
  needs. There's no `environment.yml` anywhere in the repo either; the real
  environment is an undocumented conda env (see Infrastructure).

## Build Order Discipline

Aman validates each stage on hardware before integrating it with the next.
Don't suggest skipping ahead to integration/control work before a stage is
confirmed working at real-time throughput. When picking up a task, check
which stage is actually active before assuming the next gap is ready to be
built. Right now that's squarely the distance/speed-estimation thread above.

## Repository Map

Top level of `Autonomous-Bicycle/` (13 folders). Everything not in the
"active" rows below has a specific, checkable reason it isn't used — not
just neglect.

| Folder | Status | What it is / why (not) used |
|---|---|---|
| `LaneATT/` | **Active** — most recently touched folder in the repo | Lane detection, the Angle/Stanley/CIPV controller, YOLO + depth wiring, all ONNX export. The core of the live pipeline. |
| `Yolov11/` | **Active** | YOLO11 training (both class schemes, see above), TensorRT export in progress. |
| `Depth-Anything-V2/` | **Active** — 2nd-most recently touched, 20 files in the last 3 days | Vendored official repo; where *metric* depth (vKITTI checkpoint) is being tested right now, separate from `LaneATT/lib/depth.py`'s relative-depth pipeline. |
| `HybridNets_Jetson/` | Dormant, currently broken | Standalone deployment bundle for the abandoned HybridNets multi-task net. `road_guidance.py`'s Stanley-formula + HUD-overlay pattern is confirmed to be the structural template `LaneATT/lib/angle.py` was independently written against (no import, just pattern reuse) — that's the one thing still "used" from this folder. ONNX export script has had a syntax error since 2026-07-14. |
| `HybridNets/` | Dormant | The original, full HybridNets training repo `HybridNets_Jetson/` was extracted from. Superseded by the LaneATT+YOLO11 split-model approach. |
| `LaneTCA/` | Dormant | Vendored comparison-paper repo (temporal-context lane detection). Its `evaluate` binary is a Linux ELF — won't even run on the Mac. Reference-only. |
| `lanenet-lane-detection-pytorch/` | Dormant, deliberately rejected | Vendored LaneNet. Rejected because instance-clustering + ID tracking is less stable than LaneATT's spatial approach — a **confirmed, working decision**, not neglect. (Note: `LaneATT/lib/lanenet_infer.py` is a *different*, in-pipeline LaneNet wrapper — see below.) |
| `yolov12/` | Dormant | Vendored Ultralytics/YOLOv12 clone, superseded by `Yolov11/`. |
| `deeplab/` | Dormant | DeepLabV3-ResNet50 drivable-area segmentation, AWS-SageMaker-oriented. A different approach (segmentation vs. detection) that wasn't pursued past initial setup. |
| `configs/` | Stable, low-traffic | Shared category-map JSONs for segmentation-style labels. Not actively changing. |
| `ros2_ws/` | Dormant | Gazebo/Webots **simulation only** — includes its own Stanley-controller ROS2 package. Distinct from, and unrelated to, the real Stanley implementation in `LaneATT/lib/angle.py`. No ESP32/firmware code here. |
| `Send/` | Dormant, not source | 6.2 GB of leftover output video/cache from running `HybridNets_Jetson/` scripts (`.pyc` files confirm it, no `.py` of its own). Safe to ignore or clean up. |
| `new_data/` | Frozen since April, gitignored | 44 GB of raw TuSimple-style driving-clip frames cached locally, outside git. |

**Inside `LaneATT/lib/`, a second lane-detection path also exists but isn't
active**: `lanenet_infer.py` + top-level `lane_utils.py` implement a full
LaneNet inference path (clustering, optional H-Net bird's-eye polynomial fit)
with its **own, more sophisticated ego-lane tracker** (`EgoLaneTracker` in
`lane_utils.py` — continuity-guarded, EMA-blended) that is fully built but
**never imported anywhere**. `inference.py`'s LaneNet-invoking block is
commented out. Historical output exists at `video_output_2/LaneNet/LaneNewTrained/`.
If this path is ever revisited, fix the `video_eval()` crash bug above first.

**Firmware lives outside the working tree entirely** — see Infrastructure →
Firmware below.

**One level up, outside the repo** (`/Users/amannindra/Projects/Auto/`): the
real dataset stores — `CuLaneDataset/` (43 G), `TUSimple/` (26 G),
`OpenLane/` (7.3 G), `100k_images/` + `100k_json/` (BDD100K raw, 5.6 G +
1.9 G), `Bdd100Final/` (614 M, the prepared 2-class derivative), plus a full
`openpilot/` clone (reference material, not integrated) and a `papers/`
folder.

## Infrastructure

### Training (UC Merced HPC)
- **Getting in: `ssh ucmerced`** — SSH alias in the Mac's `~/.ssh/config` for
  `anindra@login.rc.ucmerced.edu` (key `~/.ssh/id_ed25519_ucmerced`). Lands on
  a login node. Compute nodes (e.g. `gnode010`) additionally require an active
  Slurm job of yours on that node before SSH lets you in (`pam_slurm_adopt`) —
  and when that job ends, every process you left on the node is killed. The
  Vista Compute Lab hosts have their own aliases (`vistacompute1`–`4`) in the
  same file.
- Slurm cluster, login nodes `rclogin01/02/03`, partitions `cenvalarc.gpu`
  (3-day max, up to 4 concurrent jobs — used for the actual multi-day L40S
  training runs, `--gres=gpu:l40s:1`) and `test` (1-hour max, but can target
  any node type including A100 via `--gres=gpu:a100:1` — used for quick
  interactive throughput checks, e.g. the depth-cost comparison table above).
- Conda envs on the cluster (`/home/anindra/data/conda/envs/`): **`LaneNet310`**
  is the training/inference env (Python 3.10, torch 2.5.1+cu124 — matches the
  `.cpython-310.pyc` files under every `__pycache__/`; not documented in any
  repo file, only recoverable from the `.sh` job scripts). **`LaneNetCuda_12_6`**
  is the TensorRT conversion env (see the conversion section below). A third,
  older `LaneNet` env exists but is unused.
- **Never build TensorRT engines in `LaneNet310`** — its `tensorrt-cu12*`
  packages are at 11.1.0.106, whose Python bindings lack `BuilderFlag.FP16`:
  every `--fp16` build dies with `AttributeError: ... BuilderFlag has no
  attribute 'FP16'` (reproduced on both YOLO11n and Depth-Anything-V2,
  2026-07-23). Non-FP16 builds did work there, which is how the pre-existing
  LaneATT engine got built before this was understood.
- `salloc`/`srun` for interactive GPU sessions. PyTorch with AMP/mixed
  precision. Checkpoints/engines move around via SFTP or `rsync -e ssh`.
- **If a GPU node is unreachable or busy** (e.g. can't get onto `gnode010`),
  grab a fresh one on the `test` partition instead of waiting:
  ```
  salloc --partition=test --nodes=1 --ntasks-per-node=2 --cpus-per-task=28 \
    --mem=96G --gres=gpu:a100:1 --time=01:00:00
  ```
  Then SSH into whatever node it hands back and run whatever you need there.

### ONNX → TensorRT conversion (cluster side)
- Env: **`LaneNetCuda_12_6`** — deliberately version-matched to the Jetson
  (2026-07-23): TensorRT **10.3.0** (pip `tensorrt` + `tensorrt-cu12*`, same
  10.3.0 line as the Jetson's 10.3.0.30), torch **2.5.0+cu124** (Jetson
  JetPack-6.x torch is the 2.5.0 line), nvidia-cuda-runtime-cu12 **12.6.77**
  (JetPack 6.2 = CUDA 12.6), onnxruntime-gpu 1.20.1, Python 3.10.20. Known
  cosmetic wart: pip warns that torch pins `nvidia-cuda-runtime-cu12==12.4.127`
  — harmless (CUDA minors are ABI-compatible), deliberate, don't "fix" it back.
- Tool: `Yolov11/onnx_tensorRT.py` (TensorRT Python API, tqdm progress bars,
  `--hardware-compat` flag added 2026-07-23). **`trtexec` does not exist
  anywhere on the cluster** — pip TensorRT wheels never ship the binary, no
  module provides it, and the full SDK isn't installed. Don't go looking.
- The working command (GPU node, e.g. gnode010):
  ```
  cd /home/anindra/data/Autonomous-Bicycle/Yolov11
  conda run -n LaneNetCuda_12_6 python onnx_tensorRT.py \
    --onnx /home/anindra/data/Autonomous-Bicycle/LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx \
    --fp16
  ```
  The engine lands next to the ONNX. A YOLO11n FP16 build takes ~7–8 min on
  an A100, nearly all of it silent kernel autotuning — and tqdm bars don't
  survive log redirection, so an empty log does NOT mean it's hung; check
  `ps`/CPU% instead.
- `--hardware-compat` builds Ampere+-portable engines (one engine usable on
  both A100 and L40S cluster nodes). It does **not** make engines
  Jetson-loadable.
- **Engines built 2026-07-23** (FP16, TRT 10.3.0, A100, all deserialize-verified
  with correct IO shapes; per Aman's call only yolo11n among the YOLOs —
  s/m not built yet):

  | Engine (next to its ONNX under `LaneATT/onnxmodels/`) | Size | IO |
  |---|---|---|
  | `YoloN/yolo11n_coco4_nms.engine` | 7.8 MB | `images (1,3,640,640)` → `output0 (1,300,6)` |
  | `depth_onnx/depth_anything_v2_small.engine` | 52 MB | `pixel_values (1,3,518,518)` → `predicted_depth (1,518,518)` |
  | `LaneATTresnet34Aug2/models/model_0013_raw.engine` | 52 MB | `image (1,3,360,640)` → `proposals (1,1000,77)` |

  FP16 roughly halves size vs. the ONNX (LaneATT 105 MB → 52 MB; the old
  non-FP16 TRT-11 LaneATT engine, 98 MB, is preserved as
  `model_0013_raw.engine.trt11.bak`). Build times on one shared A100: ~8–15
  min each under 3-way contention (~7.5 min solo). Remember these engines are
  **cluster-only artifacts** — Jetson engines get built on-device (above).
- **Cluster-built engines can never deploy to the Jetson** — TensorRT engines
  don't cross platforms (x86_64 → aarch64) under any flag or version
  combination. Cluster builds are for TRT-10.3 parity validation (same
  builder/parser version the Jetson runs) and A100 benchmarking. For the
  Jetson: `scp` the **ONNX** file over and build on-device with NVIDIA's CLI,
  which the Jetson already has (`libnvinfer-bin`):
  ```
  /usr/src/tensorrt/bin/trtexec --onnx=<model>.onnx \
    --saveEngine=<model>.engine --fp16
  ```
  Expect on-device builds to be much slower than the A100 (tens of minutes
  for the 95–105 MB models) and RAM-tight on the 8 GB Orin — close other
  processes first.

### Firmware — ESP32-S3 (not on `main`)
- Real, functioning firmware exists only on the unmerged `origin/S3-code`
  branch (`S3/platformio.ini`, `[env:esp32-s3-devkitc-1]`, Arduino
  framework) — 4 commits, most recent tagged `fw v1`. Not checked out or
  referenced by anything in the working tree.
- **Steering**: AS5600 magnetic rotary encoder over I2C, proportional control
  (gain 10) driving a `Servo`-class ESC signal (pin 18).
- **Brake**: dedicated relay pin (`PIN_RELAY_BRAKE`, pin 6, comment says
  "Linear Actuator Brake") — checked every loop, kills speed and centers
  steering when triggered.
- **Speed/throttle**: LEDC PWM (`PIN_PWM_SPEED`, pin 7) into an op-amp.
- **L/R relay pins** (4/5) exist but their purpose isn't spelled out in
  comments — likely turn signals, not independently confirmed.
- **Control interface**: I2C slave at address `0x04` handling `BTN:`/`SPD:`/
  `ANG:` commands (comment references an external "NodeMCU"), plus a serial
  `angle_remote` command — presumably the intended Jetson↔ESP32 link, though
  nothing in the Python codebase talks to it yet.
- **No IMU code exists yet** despite the "ESP32-S3 as safety/IMU hub" framing
  — as of `fw v1`, IMU integration is still aspirational; only the steering
  encoder is wired up.
- A sibling `origin/math` branch (same base history) holds an earlier
  iteration that used **YOLOPv2** before the project moved to the current
  LaneATT+YOLO11 split — see local branch
  `backup/pre-yolopv2-cleanup-20260417` for the cleanup commit if that
  history is ever relevant.

### Vista Compute Lab (separate infra project, same person)
- Kubernetes cluster via kubeadm, 4 nodes. Coder deployed via Helm. PostgreSQL
  (Bitnami) backend. Terraform workspace templates (up to 2× A30 GPUs/user).
  firewalld hardening. Active debugging: pod lifecycle issues, RBAC/namespace
  permissions. Collaborator: Emery Silberman.
- Only pull this context in if the task is explicitly about Vista Compute Lab
  infra, not the AV pipeline itself.

### Jetson deployment environment (target — not yet exercised on real hardware)
- Known device state (from `Jetson.txt`, verified on-device 2026-07): JetPack
  **6.2.1+b38**, L4T **R36.4.7**, TensorRT **10.3.0.30**+cuda12.5 (aarch64),
  `libnvinfer-bin` installed — so NVIDIA's official CLI
  `/usr/src/tensorrt/bin/trtexec` **is available on the Jetson** even though
  it doesn't exist on the cluster. Engines for the Jetson must be built on the
  Jetson itself (see the conversion section above for why and how).
- **Use `venv` over system Python, not conda.** conda-forge can silently
  shadow JetPack's TensorRT/CUDA/PyTorch ARM bindings with CPU-only or
  incompatible builds. This has already caused problems on prior projects —
  don't suggest conda.
- Deployment stack: TensorRT (INT8 primary, FP16 fallback for
  quantization-sensitive heads), ONNX Runtime with TensorRT Execution
  Provider, `nvidia-modelopt` for QAT and 2:4 sparsity-aware fine-tuning.
- Optimization levers identified but not yet used (no physical Jetson
  benchmarking has happened in this repo yet — these are the plan, not
  measured results): DLA offload (e.g. LaneATT on DLA while YOLO11n runs on
  GPU simultaneously), VPI for preprocessing on PVA/VIC hardware blocks, CUDA
  Graphs to cut inter-stage kernel launch overhead, multi-stream concurrent
  execution.

## Datasets

- **CULane** — primary training set for lane detection. Confirmed present
  (`~/Projects/Auto/CuLaneDataset/`, 43 G). Has no depth ground truth — lane
  detection and depth are separate supervision streams, don't conflate them.
- **TuSimple** — confirmed present (`~/Projects/Auto/TUSimple/`, 26 G); noted
  as insufficient alone for lane detection (highway-only, small).
- **OpenLane** — confirmed present (`~/Projects/Auto/OpenLane/`, 7.3 G).
- **BDD100K** — **primary current dataset for YOLO11 object detection**, not
  mentioned in earlier notes at all. Raw source (`100k_images/` 5.6 G,
  `100k_json/` 1.9 G) plus the prepared 2-class derivative (`Bdd100Final/`,
  614 M, via `prepare_bdd100k.py`) — see Object Detection above for training
  status (prepared, not yet trained on).
- An earlier, first-generation **Roboflow COCO export was abandoned** for
  having too few instances (`Preprocess.py`'s own docstring: ~3,893 person
  instances vs. ~257k in full COCO train2017) — this is why the current
  4-class scheme reprocesses raw COCO2017 JSONs directly instead.
- **Comma2k19LD** — referenced in earlier notes as vehicle-state/steering-
  geometry validation data, but **not found on disk** in this pass (not in
  the sibling dataset directory, not referenced by any code). Verify it's
  still intended before relying on this — may have been acquired and later
  removed, or may never have been.

## Training Conventions

- **Augmentation**: wide-range uniform augmentation across all images
  (RandAugment-style) outperforms conditional branching by driving condition.
  Harder conditions (night, glare) get more attention via weighted
  oversampling in the dataloader — not via separate augmentation branches per
  condition. This was a deliberate simplification after Aman pushed back on a
  more complex conditional scheme; don't reintroduce that complexity.
  - Horizontal flips need care for lane detection (left/right semantics
    matter for lane assignment).
  - Grayscale augmentation is incompatible with traffic light color
    classification — don't apply it upstream of that task.

## How Aman Works — match this style

- Wants **mechanical, step-by-step explanations**, not conceptual summaries.
  E.g., for MPC he wanted the exact forward-simulation loop mechanics, not a
  high-level description. Default to precise, implementable detail over
  prose — the Kalman/TTC formulas and the depth-cost table above are the
  right level of concreteness to aim for.
- Will push back when a suggestion seems unnecessarily complex — take that
  seriously and be ready to justify or simplify, don't just defend the
  original suggestion.
- Reads primary papers directly and expects paper-grounded, specific
  analysis — `depthandspeedestimationresearch.md` (top-level, ~135 citations,
  compiled this month) is a live example of the depth this project's own
  docs go to; match that standard, don't hand-wave.
- Validates individual models on real hardware before integrating them into
  the pipeline — so far that's meant the Slurm cluster (L40S/A100), not yet
  the physical Jetson. Don't assume a model choice is settled until real
  throughput is confirmed, and be explicit about *which* hardware a given
  number came from.
- When something is quantifiable (params, FPS, TOPS, mAP), give the number.
  Avoid hand-wavy claims about performance on this project.

## What NOT to suggest (already decided against, with reasons)

- Dense monocular depth models as a category — **outdated.** Depth-Anything-V2
  (both relative and metric variants) is the live, active thread. Don't
  re-raise "too heavy for Jetson" as settled; the real, measured concern is
  the ~2× latency cost shown above, which is a live open problem, not a
  closed one.
- conda on the Jetson — breaks JetPack's TensorRT/CUDA/PyTorch bindings.
- Tracking-ID-based ego-lane assignment — LaneATT's spatial approach already
  solves this more robustly; `lane_utils.py`'s unused `EgoLaneTracker` is
  direct evidence this was tried and consciously not wired in.
- Multi-task/multi-branch detection networks (HybridNets-style) — the
  architecture class is a bad fit for this hardware (irregular ops, hard to
  optimize) even though the specific "~12 FPS" figure isn't independently
  verifiable in this repo — see Hardware Constraint above.
- Conditional augmentation branches by driving condition — oversampling in
  the dataloader already covers this more simply.
- Using raw (untracked) YOLO detections for IDM's Δv term — velocity must
  come from the Kalman-filtered `[Z, Ż]` state, not instantaneous detections
  or instantaneous depth differences (noise amplifies badly under
  differentiation — see the σ_Z → σ_v numbers above).

## Known Issues / Housekeeping

- **`kaggle.json` and a file literally named `ec2-user@54.224.142.22`** are
  both committed to git and pushed to `origin` (`Duck1405/Autonomous-Bicycle`)
  — they contain what look like live Kaggle API credentials. `Jetson.txt`
  (also tracked, currently modified in the working tree) contains a plaintext
  SSH username/password for a remote Jetson/Ubuntu host. **Rotate these
  credentials and remove the files from tracking** (`git rm --cached` +
  `.gitignore`); history-rewriting is a separate, bigger decision — don't do
  it without being asked.
- `README.md`'s stated 3-branch policy (`main`=docs, `ai`=ML work,
  `hardware`=firmware) doesn't match reality: all ML work above lives on
  `main`, and the firmware branch is actually named `S3-code`.
- `skills.md` itself was untracked in git as of this pass — consider adding
  it if you want it to survive a clean clone.
