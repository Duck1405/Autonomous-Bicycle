"""Export LaneATT to ONNX for onnxruntime — raw proposals only.

The in-graph NMS (lib/models/laneatt.py forward's last step) is a Python loop
with data-dependent shapes, so no ONNX exporter can take it. Same split as the
raw YOLO export: the graph ends at reg_proposals, and at runtime you do
softmax -> conf threshold -> nms_pytorch -> model.decode in Python.

    image (1, 3, 360, 640)  ->  proposals (1, n_anchors, 77)
                                 [cls0, cls1, start_y, start_x, length, 72 x-offsets]

Input must match frame_eval's preprocessing: cv2.resize to 640x360, to_tensor.
"""
import argparse
import torch
import onnxruntime as ort
from lib.config import Config
from pathlib import Path

DEFAULT_MODEL = "experiments/LaneATTresnet34Aug2/models/model_0013.pt"
DEFAULT_YAML_PATH = "experiments/LaneATTresnet34Aug2/config.yaml"
DEFAULT_OUT = "onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="LaneATT training checkpoint (.pt)")
    ap.add_argument("--yaml", default=DEFAULT_YAML_PATH,
                    help="experiment config.yaml (defines the architecture)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output .onnx path")
    args = ap.parse_args()
    MODEL, YAML_PATH, OUT = args.model, args.yaml, args.out

    cfg = Config(YAML_PATH)
    model = cfg.get_model()
    model.load_state_dict(torch.load(MODEL, map_location="cpu")["model"])
    model.eval()

    model.nms = lambda proposals, *args, **kwargs: proposals

    x = torch.randn(1, 3, 360, 640)
    
    
    dir_path = Path(OUT).parent
    print(f"Path: {dir_path}")
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (x,), OUT, input_names=["image"],
                      output_names=["proposals"], opset_version=17)
    print(f"wrote {OUT}")

    # Sanity check: onnxruntime output must match torch on the same input.
    
    # with torch.no_grad():
    #     torch_out = model(x)
    # sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    # (ort_out,) = sess.run(None, {"image": x.numpy()})
    # diff = (torch_out - torch.from_numpy(ort_out)).abs().max().item()
    # print(f"output shape: {ort_out.shape}, max |torch - ort| = {diff:.2e}")


if __name__ == "__main__":
    main()
