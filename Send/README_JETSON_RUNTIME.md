# Jetson Runtime Bundle

This folder is a self-contained runtime bundle for the road-guidance HybridNets checkpoint:

- `hybridnets_epoch_030_weights.pth`: your trained model weights
- `InferRoadVideo.py`: offline video inference entrypoint
- `InferRoadVideoTensorRT.py`: offline ONNX Runtime / TensorRT video inference entrypoint
- `InferRoadCamera.py`: live camera / RTSP / file-stream inference entrypoint
- `road_guidance.py`: road-center extraction, Stanley steering command, HUD rendering
- `road_segmentation_model.py`: segmentation-only checkpoint loader
- `road_segmentation_ort.py`: ONNX Runtime session wrapper with TensorRT provider support
- `backbone_runtime.py`: stripped HybridNets backbone that only builds the segmentation path
- `runtime_utils.py`: runtime-only preprocessing helpers, especially `letterbox`
- `video_common.py`: torch-free preprocessing and overlay helpers shared by the PyTorch and TensorRT paths
- `encoders/`: minimal EfficientNet encoder support used by the runtime backbone
- `hybridnets/model_runtime.py`: runtime-only BiFPN, decoder, same-padding, and segmentation head blocks
- `utils/constants.py`: segmentation mode constants
- `requirements-jetson-runtime.txt`: non-PyTorch Python package requirements
- `run_video.sh`: convenience launcher for saved-video inference
- `run_video_tensorrt.sh`: convenience launcher for ONNX Runtime / TensorRT saved-video inference
- `run_camera.sh`: convenience launcher for live camera inference

## Why this bundle exists

The original `HybridNets` repo contains training code, validation code, notebooks, and import chains that pull in extra dependencies. This bundle removes the deployment traps:

- no imports back into `/home/aman/Projects/Auto/HybridNets`
- no runtime dependency on training losses
- no attempt to download ImageNet encoder weights at startup
- no reliance on user-site packages when `hybridnets-local` is active

## Environment

The launch scripts assume a conda env named `hybridnets-local`.

Important:

- keep `PYTHONNOUSERSITE=1`
- keep `unset PYTHONPATH`

That matters because the workstation env was shadowed by `~/.local` PyTorch packages. The launch scripts already enforce the safe settings.

## Jetson package setup

1. Install Jetson-native `torch` and `torchvision` wheels that match your JetPack version.
2. Activate `hybridnets-local`.
3. Install the remaining Python packages:

```bash
pip install -r requirements.txt
```

If you need on-screen preview from `cv2.imshow`, use a Jetson build of OpenCV with GUI support. `opencv-python-headless` is enough for file inference and headless camera runs.

## TensorRT video path

The PyTorch launchers still require a working Jetson-native `torch` install. If Jetson PyTorch is blocked by missing system libraries, use the ONNX Runtime / TensorRT path instead.

Requirements for this path:

1. Export a segmentation-only ONNX model on a machine where the checkpoint already loads.
2. Copy that `.onnx` file into this folder, for example as `hybridnets_road_segmentation.onnx`.
3. Install an `onnxruntime` build on the Jetson that includes `TensorrtExecutionProvider`.

Run it with:

```bash
bash run_video_tensorrt.sh /path/to/model.onnx /path/to/input.mp4 /path/to/output.mp4
```

Useful example:

```bash
bash run_video_tensorrt.sh ./hybridnets_road_segmentation.onnx /home/rayan/aman/Auto/test_videos/input.mp4 ./output_stanley_trt.mp4 --provider tensorrt
```

## Running video inference

```bash
bash run_video.sh /path/to/input.mp4 /path/to/output.mp4
```

Any extra CLI flags after the first two arguments are passed through to `InferRoadVideo.py`.

## Running live camera inference

```bash
bash run_camera.sh --source 0 --display false
```

Useful examples:

```bash
bash run_camera.sh --source 0 --display true
bash run_camera.sh --source rtsp://user:pass@camera/stream --display false --save-video true
```

## Notes about the controller

The guidance overlay now shows:

- `Steering Cmd`: the Stanley steering command
- `Path Heading`: the fitted road-center tangent heading
- `Cross-track`: lateral offset from the fitted path

This is the value you would use downstream for steering logic, not the old heading-only angle.
