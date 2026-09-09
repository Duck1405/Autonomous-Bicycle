# Downloaded checkpoint mapping

| Checkpoint | Matching local architecture | Output/task |
|---|---|---|
| `models/DDRNet23s_imagenet.pth` | `../classification/DDRNet_23_slim.py`, `get_model()` | 1,000 ImageNet class scores per image |
| `models/DDRNet23_imagenet.pth` | `../classification/DDRNet_23.py`, `get_model()` | 1,000 ImageNet class scores per image |
| `models/best_val_smaller.pth` | `DDRNet_23_slim.py`, planes=32, spp_planes=128, head_planes=64 | 19 segmentation scores per spatial location |
| `models/best_val.pth` | `DDRNet_23.py`, planes=64, spp_planes=128, head_planes=128 | 19 segmentation scores per spatial location |

All four matches were verified with `load_state_dict(..., strict=True)`.
The segmentation checkpoints contain a `model.` prefix, an auxiliary training
head (`model.seghead_extra.*`), and a loss weight (`loss.criterion.weight`).
The validation script removes that prefix and explicitly excludes those
training-only entries when constructing the inference model with `augment=False`.
No inference parameters are skipped. The normal model constructors are used
directly, bypassing the hardcoded pretrained helper functions.

The ImageNet checkpoints do not contain trained segmentation heads. They can
initialize a backbone for training, but cannot directly produce a road mask.
Neither segmentation checkpoint should be loaded directly into the speed-test
architecture: that file omits batch-normalization operations and requires proper
conversion/fusion of weights before it can reproduce the trained network.

## Reproduce the video test

From this directory, using the existing Lannet310 environment:

```bash
python validate_models.py
```

The default test renders the first 180 consecutive frames of
`LaneATT/video_input/IMG_5105.mp4` and separately checks nine frames spread across
the entire source video. `--frames 0` renders the entire source video instead.
This is a CPU validation script, not a GPU speed benchmark.

- Continuous clip: inference at width=1024, height=512; output panels at 640x360.
- Nine distributed samples: inference at width=2048, height=1024.
- Input: OpenCV BGR converted to RGB, divided by 255, then normalized with
  ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
- Output: bilinearly resize logits with `align_corners=False`, then take
  `argmax(dim=1)` across the 19 classes. Road uses standard Cityscapes train ID 0.
- Class names are not embedded in these weight dictionaries; the script uses
  the standard Cityscapes ordering associated with the repository's 19-class models.
- Saved `labels_*.png` files contain raw class IDs 0-18, not display colors.

Artifacts are in `output/checkpoint_validation/`: `road_comparison.mp4` shows
original, slim road overlay, and wider-model road overlay from left to right.
`sample_*.jpg` shows distributed full-resolution inference comparisons.
`report.json` records strict-loading results, frame counts, logit ranges,
predicted classes, road fractions, and CPU inference/postprocessing times.
The script decodes every generated video frame to verify the saved artifact.

This test checks compatibility, finite outputs, video generation, and visual
plausibility. It does not measure segmentation accuracy against labeled ground
truth or prove that every road pixel is correctly classified.
