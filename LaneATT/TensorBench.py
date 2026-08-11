depth_engine= "LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine"
laneNet = "LaneATT/onnxmodels/depth_onnx/depth_anything_v2_small.engine"
YoloS = "LaneATT/onnxmodels/LaneATTresnet34Aug2/models/model_0013_raw.engine"
YoloM = "LaneATT/onnxmodels/YoloM/yolo11m_coco4_nms.onnx"
YoloN = "LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.engine"



from cuda.bindings import runtime as cudart
import sys

def check(err):
    # cuda-python returns (cudaError_t, *values); raise on any non-success status.
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

def load_engine(runtime, engine_path):
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def main()


if __name__ == "__main__":
    shape = sys.argv[4].split("x")
    main(
        engine_path=sys.argv[1],
        input_path=sys.argv[2],
        output_path=sys.argv[3],
        height=int(shape[0]),
        width=int(shape[1]),
    )