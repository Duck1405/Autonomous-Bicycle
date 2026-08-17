# TensorRT Benchmark Notes — Jetson Orin Nano Super

Running log of TensorRT and ONNX Runtime inference benchmarks for the Pathfinder
perception stack.

Unless stated otherwise, every run below shares these conditions:

| | |
|---|---|
| Device | Jetson Orin Nano Super (`sm87`), 7.44 GiB unified memory |
| TensorRT | 10.3.0 |
| Timed frames | 500 |
| Execution | Sequential, one frame at a time |
| Repo | `~/aman/Autonomous-Bicycle` |

Console output is reproduced verbatim (ANSI colour codes stripped).

---

## Contents

1. [Serial pipeline sweep — LaneATT + YOLO](#1-serial-pipeline-sweep--laneatt--yolo)
2. [Single-model runs](#2-single-model-runs)
3. [Two processes in separate terminals](#3-two-processes-in-separate-terminals)
4. [ONNX Runtime (CUDA FP32) vs TensorRT (FP16)](#4-onnx-runtime-cuda-fp32-vs-tensorrt-fp16--2026-08-14)
5. [Concurrent vs serial — two models](#5-concurrent-vs-serial--two-models--2026-08-14)
6. [Concurrent vs serial — three models](#6-concurrent-vs-serial--three-models--2026-08-14)
7. [Running ONNX + LaneATT + MaxMode](#7-running-onnx--laneatt--maxmode--2026-08-17)

---

## Headline numbers

**Power mode sweep** (§1, `camera_1.mp4`, LaneATT + YOLO serial):

| Run | Power mode | LaneATT engine | YOLO engine | Engines only | Pipeline |
|---|---|---:|---:|---:|---:|
| Test 1 | base case | 23.6 ms | 18.6 ms | 42.3 ms → 23.7 FPS | 51.7 ms → **19.33 FPS** |
| Test 2 | 25 W | 26.7 ms | 21.1 ms | 47.8 ms → 20.9 FPS | 58.2 ms → **17.20 FPS** |
| Test 3 | MAXN_SUPER | 22.2 ms | 17.6 ms | 39.8 ms → 25.1 FPS | 48.9 ms → **20.45 FPS** |

**ONNX Runtime vs TensorRT** (§4, `IMG_6893_30fps.mp4`, single model per run):

| Model | ORT CUDA FP32 | TensorRT FP16 | Speedup | TRT engine FPS |
|---|---:|---:|---:|---:|
| LaneATT (ResNet-34) | 125.7 ms | 27.6 ms | **4.55×** | 36.2 |
| YOLOv11n | 30.3 ms | 16.6 ms | **1.83×** | 60.3 |

**Serial vs concurrent processes** (§5–6, MAXN_SUPER, `IMG_6893_30fps.mp4`):

| Configuration | Same-frame pipeline | Notes |
|---|---:|---|
| 2 models, one process (serial) | 16.00 FPS | LaneATT 29.0 ms + YOLO 19.9 ms |
| 2 models, two processes | — | 35.75 / 43.07 FPS, not same-frame |
| 3 models, one process (serial) | 10.14 FPS | Depth is the long pole at 34.6 ms |
| 3 models, three processes | 17.81 FPS | **1.76×**, Depth-limited, not same-frame |

---

## 1. Serial pipeline sweep — LaneATT + YOLO

Video: `/home/mlc/aman/Autonomous-Bicycle/Videos2/camera_1.mp4`

Engines used in all three tests:

```text
laneatt: LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.engine
    in  image[1, 3, 360, 640] float32
    out proposals[1, 1000, 77] float32
yolo: LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.engine
    in  images[1, 3, 640, 640] float32
    out output0[1, 300, 6] float32
```

### 1.1 Test 1 — base case

Using FB-32. *(3.28 GiB GPU free of 7.44 GiB)*

```text
=== TensorRT sequential per-frame benchmark: 500 frames of /home/mlc/aman/Autonomous-Bicycle/Videos2/camera_1.mp4 ===
video read: 0.7 ms/frame
laneatt  preprocess    4.0 ms/frame + engine   23.6 ms/frame
yolo     preprocess    4.7 ms/frame + engine   18.6 ms/frame
engines only:      42.3 ms/frame -> 23.7 FPS
preprocess only:    8.7 ms/frame
pipeline wall (read + preprocess + engines, sequential): 51.7 ms/frame -> 19.33 FPS
```

### 1.2 Test 2 — warmup 20, 25 W

*(3.29 GiB GPU free of 7.44 GiB)*

```text
=== TensorRT sequential per-frame benchmark: 500 frames of /home/mlc/aman/Autonomous-Bicycle/Videos2/camera_1.mp4 ===
video read: 0.8 ms/frame
laneatt  preprocess    4.3 ms/frame + engine   26.7 ms/frame
yolo     preprocess    5.3 ms/frame + engine   21.1 ms/frame
engines only:      47.8 ms/frame -> 20.9 FPS
preprocess only:    9.6 ms/frame
pipeline wall (read + preprocess + engines, sequential): 58.2 ms/frame -> 17.20 FPS
```

### 1.3 Test 3 — warmup 20, MAXN_SUPER

*(3.31 GiB GPU free of 7.44 GiB)*

```text
=== TensorRT sequential per-frame benchmark: 500 frames of /home/mlc/aman/Autonomous-Bicycle/Videos2/camera_1.mp4 ===
video read: 0.7 ms/frame
laneatt  preprocess    4.0 ms/frame + engine   22.2 ms/frame
yolo     preprocess    4.4 ms/frame + engine   17.6 ms/frame
engines only:      39.8 ms/frame -> 25.1 FPS
preprocess only:    8.4 ms/frame
pipeline wall (read + preprocess + engines, sequential): 48.9 ms/frame -> 20.45 FPS
```

### Render command

```bash
python trt_video_benchmark.py \
    --video Videos2/camera_1.mp4 \
    --warmup 200 \
    --frames 99999 \
    --render LaneATT/video_output_4/render2.mp4
```

---

## 2. Single-model runs

Video: `Videos2/IMG_2635.mp4` (18506 frames, 1080x1920 @ 29.97 fps), env `LaneNet310`.

### 2.1 YOLOv11n only — with render

```bash
python trt_video_benchmark.py --laneatt-on False --video Videos2/IMG_2635.mp4
```

```text
tensorrt 10.3.0, sm87, 4.63 GiB GPU free of 7.44 GiB
Loading engines for ['laneatt', 'yolo', 'depth'] from ['LaneATT/onnxmodels/YoloN/YoloN_fb16.engine']
yolo: LaneATT/onnxmodels/YoloN/YoloN_fb16.engine
    in  images[1, 3, 640, 640] float32
    out output0[1, 300, 6] float32
rendering 1080x1920 @ 29.97 fps [mp4v] -> LaneATT/video_output_4/render.mp4
video: Videos2/IMG_2635.mp4 (18506 frames), timing 500 frames after 20 warmup

=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_2635.mp4 ===
video read: 42.3 ms/frame
yolo     preprocess    4.1 ms/frame + engine   16.7 ms/frame
engines only:      16.7 ms/frame -> 60.0 FPS
preprocess only:    4.1 ms/frame
render (D2H + decode + draw + write):   23.8 ms/frame
pipeline wall (read + preprocess + engines + render, sequential): 87.0 ms/frame -> 11.50 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_2635.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": "LaneATT/video_output_4/render.mp4",
  "read_ms": 42.27142772998195,
  "models": {
    "yolo": {
      "preprocess_ms": 4.1265655034803785,
      "engine_ms": 16.677420647989493
    }
  },
  "engines_only_ms": 16.677420647989493,
  "engines_only_fps": 59.96131063112284,
  "render_ms": 23.841723663790617,
  "pipeline_ms": 86.9802556859795,
  "pipeline_fps": 11.496862042004686
}
```

</details>

### 2.2 YOLOv11n only — no render

> **Note:** this section was headed "Only LaneATT" in the original notes, but the
> command passes `--laneatt-on False`, loads `YoloN_fb16.engine`, and the output
> reports `yolo` timings. The run measured YOLOv11n, not LaneATT.

```bash
python trt_video_benchmark.py --laneatt-on False --video Videos2/IMG_2635.mp4 --no-render
```

```text
tensorrt 10.3.0, sm87, 4.62 GiB GPU free of 7.44 GiB
Loading engines for ['laneatt', 'yolo', 'depth'] from ['LaneATT/onnxmodels/YoloN/YoloN_fb16.engine']
yolo: LaneATT/onnxmodels/YoloN/YoloN_fb16.engine
    in  images[1, 3, 640, 640] float32
    out output0[1, 300, 6] float32
video: Videos2/IMG_2635.mp4 (18506 frames), timing 500 frames after 20 warmup

=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_2635.mp4 ===
video read: 41.1 ms/frame
yolo     preprocess    3.8 ms/frame + engine   16.8 ms/frame
engines only:      16.8 ms/frame -> 59.7 FPS
preprocess only:    3.8 ms/frame
pipeline wall (read + preprocess + engines, sequential): 61.7 ms/frame -> 16.21 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_2635.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": null,
  "read_ms": 41.11377167433966,
  "models": {
    "yolo": {
      "preprocess_ms": 3.784406661579851,
      "engine_ms": 16.75959924655035
    }
  },
  "engines_only_ms": 16.75959924655035,
  "engines_only_fps": 59.6672978445968,
  "render_ms": 0.0,
  "pipeline_ms": 61.70422625797801,
  "pipeline_fps": 16.20634534527861
}
```

</details>

---

## 3. Two processes in separate terminals

Both models launched concurrently via `./dual.sh` on `Videos2/IMG_2635.mp4`.

```text
tensorrt 10.3.0, sm87, 4.48 GiB GPU free of 7.44 GiB
laneatt: LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine
    in  image[1, 3, 360, 640] float32
    out proposals[1, 1000, 77] float32
yolo: LaneATT/onnxmodels/YoloN/YoloN_fb16.engine
    in  images[1, 3, 640, 640] float32
    out output0[1, 300, 6] float32
```

### 3.1 YOLO process

```text
=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_2635.mp4 ===
video read: 56.0 ms/frame
yolo     preprocess    5.1 ms/frame + engine   29.3 ms/frame
engines only:      29.3 ms/frame -> 34.1 FPS
preprocess only:    5.1 ms/frame
pipeline wall (read + preprocess + engines, sequential): 90.5 ms/frame -> 11.05 FPS
```

### 3.2 LaneATT process

```text
=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_2635.mp4 ===
video read: 53.1 ms/frame
laneatt  preprocess    3.2 ms/frame + engine   39.7 ms/frame
engines only:      39.7 ms/frame -> 25.2 FPS
preprocess only:    3.2 ms/frame
pipeline wall (read + preprocess + engines, sequential): 96.1 ms/frame -> 10.40 FPS
```

<details>
<summary>JSON output — both processes</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_2635.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": null,
  "read_ms": 55.96574765164405,
  "models": {
    "yolo": {
      "preprocess_ms": 5.1496796639985405,
      "engine_ms": 29.29859625082463
    }
  },
  "engines_only_ms": 29.29859625082463,
  "engines_only_fps": 34.131328048587115,
  "render_ms": 0.0,
  "pipeline_ms": 90.47747384995455,
  "pipeline_fps": 11.052474803378944
}
```

```json
{
  "frames": 500,
  "video": "Videos2/IMG_2635.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": null,
  "read_ms": 53.10582197189797,
  "models": {
    "laneatt": {
      "preprocess_ms": 3.1994896807009354,
      "engine_ms": 39.70775455003604
    }
  },
  "engines_only_ms": 39.70775455003604,
  "engines_only_fps": 25.183997718629303,
  "render_ms": 0.0,
  "pipeline_ms": 96.11390623397892,
  "pipeline_fps": 10.404321696858393
}
```

</details>

---

## 4. ONNX Runtime (CUDA FP32) vs TensorRT (FP16) — 2026-08-14

| | |
|---|---|
| Env | conda `LaneNet310Cuda` |
| Video | `Videos2/IMG_6893_30fps.mp4` (8719 frames) |
| Timing | 500 timed frames, start frame 0, 20-frame warmup |
| Execution | Runs were sequential, one at a time |

### 4.1 LaneATT (ResNet-34) — ONNX Runtime CUDA FP32

```bash
python onnx_video_LaneaTT.py --video Videos2/IMG_6893_30fps.mp4 --start-frame 0 --frames 500 --warmup 20
```

<details>
<summary>Startup warnings</summary>

```text
2026-08-14 21:39:15.777228630 [W:onnxruntime:Default, device_discovery.cc:211 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:91 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-08-14 21:39:16.379802345 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-14 21:39:16.379920940 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-14 21:39:16.379953165 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-14 21:39:16.379970414 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-14 21:39:16.379982702 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
```

</details>

```text
onnxruntime 1.24.0, providers ['CUDAExecutionProvider', 'CPUExecutionProvider']
laneatt: LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx
    in  image [1, 3, 360, 640] tensor(float)
    out proposals [1, 1000, 77] tensor(float)

=== ONNX Runtime sequential per-frame benchmark: 500 frames of Videos2/IMG_6893_30fps.mp4 ===
video read: 2.8 ms/frame
laneatt  preprocess    4.0 ms/frame + engine  125.7 ms/frame
engines only:     125.7 ms/frame -> 8.0 FPS
preprocess only:    4.0 ms/frame
pipeline wall (read + preprocess + engines, sequential): 132.5 ms/frame -> 7.55 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_6893_30fps.mp4",
  "runtime": "onnxruntime",
  "onnxruntime": "1.24.0",
  "providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ],
  "model": "LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx",
  "start_frame": 0,
  "warmup": 20,
  "read_ms": 2.7941555439974763,
  "models": {
    "laneatt": {
      "preprocess_ms": 3.9810028260035324,
      "engine_ms": 125.675222004018
    }
  },
  "engines_only_ms": 125.675222004018,
  "engines_only_fps": 7.957017971036715,
  "pipeline_ms": 132.4985135959978,
  "pipeline_fps": 7.547254477503857
}
```

</details>

### 4.2 YOLOv11n (4-class, NMS baked in) — ONNX Runtime CUDA FP32

```bash
python onnx_video_Yolo.py --video Videos2/IMG_6893_30fps.mp4 --start-frame 0 --frames 500 --warmup 20
```

```text
2026-08-14 21:40:26.302132321 [W:onnxruntime:Default, device_discovery.cc:211 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:91 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
onnxruntime 1.24.0, providers ['CUDAExecutionProvider', 'CPUExecutionProvider']
yolo: LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx
    in  images [1, 3, 640, 640] tensor(float)
    out output0 [1, 300, 6] tensor(float)

=== ONNX Runtime sequential per-frame benchmark: 500 frames of Videos2/IMG_6893_30fps.mp4 ===
video read: 4.3 ms/frame
yolo     preprocess    7.8 ms/frame + engine   30.3 ms/frame
engines only:      30.3 ms/frame -> 33.0 FPS
preprocess only:    7.8 ms/frame
pipeline wall (read + preprocess + engines, sequential): 42.5 ms/frame -> 23.52 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_6893_30fps.mp4",
  "runtime": "onnxruntime",
  "onnxruntime": "1.24.0",
  "providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ],
  "model": "LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.onnx",
  "start_frame": 0,
  "warmup": 20,
  "read_ms": 4.310318942043523,
  "models": {
    "yolo": {
      "preprocess_ms": 7.823335941968253,
      "engine_ms": 30.335276196048653
    }
  },
  "engines_only_ms": 30.335276196048653,
  "engines_only_fps": 32.96492154998924,
  "pipeline_ms": 42.51390294800149,
  "pipeline_fps": 23.521717147990255
}
```

</details>

### 4.3 LaneATT (ResNet-34) — TensorRT 10.3.0 FP16

```bash
python trt_video_benchmark.py --video Videos2/IMG_6893_30fps.mp4 --start-frame 0 --frames 500 --warmup 20 \
    --models laneatt --laneatt-on true --yolo-on false --depth-on false --no-render
```

```text
tensorrt 10.3.0, sm87, 4.66 GiB GPU free of 7.44 GiB
Loading engines for ['laneatt'] from ['LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine']
laneatt: LaneATT/onnxmodels/LaneATTresnet34Aug2/models/LaneATT_fb16.engine
    in  image[1, 3, 360, 640] float32
    out proposals[1, 1000, 77] float32

=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_6893_30fps.mp4 ===
video read: 3.0 ms/frame
laneatt  preprocess    4.2 ms/frame + engine   27.6 ms/frame
engines only:      27.6 ms/frame -> 36.2 FPS
preprocess only:    4.2 ms/frame
pipeline wall (read + preprocess + engines, sequential): 34.9 ms/frame -> 28.66 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_6893_30fps.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": null,
  "read_ms": 2.9926889280577598,
  "models": {
    "laneatt": {
      "preprocess_ms": 4.210776427946257,
      "engine_ms": 27.62988268598201
    }
  },
  "engines_only_ms": 27.62988268598201,
  "engines_only_fps": 36.19269800618259,
  "render_ms": 0.0,
  "pipeline_ms": 34.89391111800069,
  "pipeline_fps": 28.65829504231559
}
```

</details>

### 4.4 YOLOv11n (4-class, NMS baked in) — TensorRT 10.3.0 FP16

```bash
python trt_video_benchmark.py --video Videos2/IMG_6893_30fps.mp4 --start-frame 0 --frames 500 --warmup 20 \
    --models yolo --laneatt-on false --yolo-on true --depth-on false --no-render
```

```text
tensorrt 10.3.0, sm87, 4.67 GiB GPU free of 7.44 GiB
Loading engines for ['yolo'] from ['LaneATT/onnxmodels/YoloN/YoloN_fb16.engine']
yolo: LaneATT/onnxmodels/YoloN/YoloN_fb16.engine
    in  images[1, 3, 640, 640] float32
    out output0[1, 300, 6] float32

=== TensorRT sequential per-frame benchmark: 500 frames of Videos2/IMG_6893_30fps.mp4 ===
video read: 2.4 ms/frame
yolo     preprocess    4.3 ms/frame + engine   16.6 ms/frame
engines only:      16.6 ms/frame -> 60.3 FPS
preprocess only:    4.3 ms/frame
pipeline wall (read + preprocess + engines, sequential): 23.3 ms/frame -> 42.96 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_6893_30fps.mp4",
  "tensorrt": "10.3.0",
  "compute_capability": "sm87",
  "render": null,
  "read_ms": 2.375681304012687,
  "models": {
    "yolo": {
      "preprocess_ms": 4.259559268015437,
      "engine_ms": 16.590314691973617
    }
  },
  "engines_only_ms": 16.590314691973617,
  "engines_only_fps": 60.276132102774355,
  "render_ms": 0.0,
  "pipeline_ms": 23.27878891999717,
  "pipeline_fps": 42.95756121320257
}
```

</details>

---

## 5. Concurrent vs serial — two models — 2026-08-14

| | |
|---|---|
| Device | Jetson Orin Nano Super, MAXN_SUPER selected |
| Env | conda `LaneNet310Cuda` |
| Video | `Videos2/IMG_6893_30fps.mp4`, 500 frames, 20 warmup frames, no rendering |

The observation is real and reproducible. Both modes were run on the Jetson in
`LaneNet310Cuda` with MAXN_SUPER selected:

| Mode | LaneATT engine | YOLO engine | Combined / pair pipeline rate |
|---|---:|---:|---|
| One process, current script | 29.0 ms | 19.9 ms | 16.00 FPS |
| Two concurrent processes | 20.7 ms | 16.0 ms | 35.75 LaneATT FPS, 43.07 YOLO FPS |

### Why the serial mode is slower

The current "together" mode is explicitly serial. Its loop does:

1. preprocess LaneATT
2. H2D copy, execute LaneATT, `cudaStreamSynchronize`
3. preprocess YOLO
4. H2D copy, execute YOLO, `cudaStreamSynchronize`

That synchronization is in `jetson_tools/trt_runner.py`, so the next model cannot
begin until the prior engine has fully completed. Therefore 29.0 + 19.9 = 48.9 ms
engine time; preprocessing and video I/O raise the total to 62.5 ms/frame.

The two-terminal test creates two separate CUDA contexts/streams. While one
process is doing CPU video decoding/preprocessing or waiting on its stream, the
other can submit and run GPU work. Their timed intervals overlap, so their
individual FPS values **must not be added**. This gives higher throughput, but each
process independently decodes the video, so the predictions are not synchronized
as one LaneATT+YOLO result for the same frame.

### Resource measurement during the concurrent run

| Metric | Value |
|---|---|
| Unified RAM | 1.51 GB idle/start → 2.16 / 7.62 GB peak |
| Swap | 0 MB used |
| GPU activity | sampled 23–97% GR3D, frequently 70–90% |
| Junction temperature | 49.1–52.2 °C |
| Peak input power | 13.46 W |
| CPU clock | reached 1.728 GHz |

This is not a GPU-memory issue and not thermal throttling. MAXN_SUPER permits high
clocks, but clocks are still dynamically managed; `jetson_clocks --show` requires
sudo and was not applied or verified in this run.

> **Conclusion:** 16 FPS is correct for strict serial, same-frame paired inference.
> The faster two-process result is concurrent throughput. To preserve one shared
> camera/frame stream while executing concurrently, the runtime must submit each
> engine to a separate stream and synchronize only after both have been submitted.
> No source code was changed for this measurement.

---

## 6. Concurrent vs serial — three models — 2026-08-14

| | |
|---|---|
| Device | Jetson Orin Nano Super, MAXN_SUPER selected |
| Env | conda `LaneNet310Cuda` |
| Video | `Videos2/IMG_6893_30fps.mp4`, 500 frames, 20 warmup frames, no rendering |

LaneATT, YOLOv11n, and Depth Anything V2 Small were run concurrently as three
independent TensorRT processes, then compared against a serial one-process run.
All runs completed 500 frames successfully with `render=null`; no output video was
produced.

| Mode | LaneATT | YOLOv11n | Depth | Same-frame pipeline |
|---|---:|---:|---:|---:|
| Serial, one script | 18.8 ms engine | 15.7 ms engine | 34.6 ms | 10.14 FPS |
| Three concurrent processes | 21.6 ms engine | 19.1 ms engine | 34.1 ms | 17.81 FPS (Depth) |

Concurrent-process pipeline rates:

- LaneATT: 33.99 FPS
- YOLOv11n: 36.19 FPS
- Depth: 17.81 FPS

Depth is the slowest process, so it limits effective three-model throughput to
about 17.8 FPS. This is 1.76× the 10.14 FPS serial same-frame pipeline rate. The
individual concurrent-process FPS values overlap and **must not be added**.

### Resource measurement

| Metric | Three processes | Serial script |
|---|---|---|
| Unified RAM peak | 2.75 / 7.62 GB | 2.23 / 7.62 GB |
| Swap | 0 MB | 0 MB |
| GPU activity, average / peak | 66.1% / 99% | 55.5% / 99% |
| CPU per-core use, average / peak | 41.9% / 100% | 26.8% / 100% |
| CPU clock peak | 1.728 GHz | 1.728 GHz |
| Junction temperature peak | 54.9 °C | 53.9 °C |
| Input-power peak | 9.57 W | 9.07 W |

The three-process run is faster because it keeps the GPU busier: average GPU
activity rose from 55.5% to 66.1%. There was no swap use, memory pressure, or
thermal throttling. However, independent processes decode and advance through the
video independently; their outputs are not synchronized as one same-frame
LaneATT+YOLO+Depth result.

> **A production implementation should share one decoder, submit the three engines
> to separate CUDA streams, and synchronize once per frame.** No source code was
> changed for this measurement.

---

## 7. Running ONNX + LaneATT + MaxMode — 2026-08-17

```bash
python onnx_video_LaneaTT.py --video Videos2/IMG_6893_30fps.mp4 --start-frame 0 --frames 500 --warmup 20
```

<details>
<summary>Startup warnings</summary>

```text
2026-08-17 15:13:08.509854322 [W:onnxruntime:Default, device_discovery.cc:211 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:91 ReadFileContents Failed to open file: "/sys/class/drm/card1/device/vendor"
2026-08-17 15:13:09.590128866 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-17 15:13:09.590197252 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-17 15:13:09.590226565 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-17 15:13:09.590241669 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
2026-08-17 15:13:09.590253349 [W:onnxruntime:Default, scatter_nd.h:51 ScatterNDWithAtomicReduction] ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
```

</details>

```text
onnxruntime 1.24.0, providers ['CUDAExecutionProvider', 'CPUExecutionProvider']
laneatt: LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx
    in  image [1, 3, 360, 640] tensor(float)
    out proposals [1, 1000, 77] tensor(float)

=== ONNX Runtime sequential per-frame benchmark: 500 frames of Videos2/IMG_6893_30fps.mp4 ===
video read: 2.9 ms/frame
laneatt  preprocess    4.1 ms/frame + engine  129.5 ms/frame
engines only:     129.5 ms/frame -> 7.7 FPS
preprocess only:    4.1 ms/frame
pipeline wall (read + preprocess + engines, sequential): 136.5 ms/frame -> 7.33 FPS
```

<details>
<summary>JSON output</summary>

```json
{
  "frames": 500,
  "video": "Videos2/IMG_6893_30fps.mp4",
  "runtime": "onnxruntime",
  "onnxruntime": "1.24.0",
  "providers": [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
  ],
  "model": "LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx",
  "start_frame": 0,
  "warmup": 20,
  "read_ms": 2.876383224269375,
  "models": {
    "laneatt": {
      "preprocess_ms": 4.07855007739272,
      "engine_ms": 129.45717089780373
    }
  },
  "engines_only_ms": 129.45717089780373,
  "engines_only_fps": 7.724562440727377,
  "pipeline_ms": 136.45163447997766,
  "pipeline_fps": 7.328604042091821
}
```

</details>

> Note: this is a near-repeat of §4.1 (same model, same video, ONNX Runtime CUDA
> FP32) but run 3 days later at MAXN_SUPER instead of whatever mode was active on
> 2026-08-14 — 129.5 ms vs 125.7 ms engine time, consistent within run-to-run
> noise. No TensorRT counterpart run alongside it this time.
