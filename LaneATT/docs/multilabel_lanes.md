# Optional BDD100K lane attributes

Set `model.parameters.multilabel: true` in the YAML. Omitted or `false`
keeps the binary model, parameter names, 77-column targets (S=72), and
four-element forward tuples. YAML `Yes` and `No` also work; do not quote booleans.
`Config.get_dataset()` propagates the model switch to every LaneDataset split.
For direct construction, pass `LaneDataset(..., multilabel=True)` yourself.

Enabled mode adds nine sigmoid logits per anchor, separate from the original
lane/background classifier and geometry. Each lane receives a category label
and, when annotated, a style label:

- Categories: single white, single yellow, single other, double white,
  double yellow, double other, road curb.
- Styles: solid, dashed. Missing style and curb style are masked, not negatives.

Crosswalks and cross-road (`direction: vertical`) annotations are excluded in
enabled mode. The legacy loader behavior is preserved when disabled.
The category order is defined in `lib/lane_attributes.py`; changing it invalidates
attribute checkpoint semantics. Category-map IDs are not network output indices.
Color is derived from the predicted category. Attribute targets are appended to
geometry targets (86 columns for S=72). Crop/flip/perspective transforms preserve
the association between each lane's points and attributes. BDD hue/saturation
shifts are disabled in the supplied config to preserve white/yellow identity.

The attribute head uses masked binary cross-entropy on positive matched anchors.
Set `loss_parameters.attribute_loss_weight` to change its weight (default 1).
Geometric matching, regression, lane confidence, and class-agnostic lane NMS stay
separate. Training logs include `attribute_loss`. Keep training NMS sufficiently
large; `max_lanes` is annotation capacity, not the output cap.

## PyTorch inference

```python
with torch.no_grad():
    outputs = model(images, conf_threshold=0.5, nms_thres=50, nms_topk=4)
    lanes = model.decode(outputs, as_lanes=True)[0]
for lane in lanes:
    print(lane.metadata['lane_category'])       # e.g. 'single yellow'
    print(lane.metadata['lane_color'])          # e.g. 'yellow'
    print(lane.metadata['lane_style'])          # e.g. 'solid'
    print(lane.metadata['image_side'])          # 'left', 'right', 'center'
    print(lane.metadata['attribute_scores'])    # all nine sigmoid probabilities
```

Category and style use the highest score in their respective groups, with a
0.5 acceptance threshold; otherwise the result is `unknown`. These scores are
not calibrated probabilities. `image_side` is a majority vote over ALL lane
points relative to image x=0.5. It is not an ego-lane assignment or a world-frame
position, and yellow does not automatically mean left. The code never invents
a missing opposite boundary based on color.

Enabled forward tuples have a fifth element: selected attribute logits aligned
with the NMS anchor indices. `decode(as_lanes=False)` still returns geometry
only; use `as_lanes=True` to obtain semantic metadata. `LaneATTInference.frame_eval`
preserves this metadata on the returned Lane objects. Converting them to pixel
arrays alone discards the metadata; consumers must retain it explicitly.

## Checkpoints and deployment

Existing binary checkpoints load strictly with `multilabel: false`. Attribute
checkpoints load strictly with `true`. Start a NEW experiment for attribute
training; do not resume a binary experiment's optimizer state.
For explicit initialization from a compatible binary model:

```python
result = model.load_state_dict(binary_state_dict, strict=False)
assert set(result.missing_keys) == {'attribute_layer.weight', 'attribute_layer.bias'}
assert not result.unexpected_keys
```

This initializes the new attribute head randomly; it must then be trained. All
other architecture settings, including anchor count, must match the checkpoint.
No trained attribute checkpoint is supplied by this code change.

CULane/TuSimple targets lack type labels. With multilabel enabled, their appended
attribute targets are masked; geometry training/evaluation still works. Their
F1 scores do not measure lane-type accuracy. The BDD evaluator also currently
reports geometry metrics only; attribute quality needs separate validation.

The current raw ONNX exporter, TensorRT/ROS2 consumers, and ROS lane messages
handle geometry only. `convertonnx.py` explicitly rejects enabled mode rather
than silently exporting a model that loses types. Use PyTorch for attribute
inference until those consumers and the exported output contract are extended.

## Verification

From LaneATT in the Lannet310 environment:

```sh
python -m unittest discover -s tests -p 'test_lane_attributes.py'
python -m unittest discover -s tests -p 'test_bdd100k_loader.py'
```
