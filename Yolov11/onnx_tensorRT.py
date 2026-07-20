import tensorrt as trt

ONNX = "models/yolo11s_coco4/run5/yolo11s_coco4.onnx"
ENGINE = "yolo11s_coco4_engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
parser = trt.OnnxParser(network, logger)

if not parser.parse_from_file(ONNX):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise SystemExit("ONNX parse failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
config.set_flag(trt.BuilderFlag.FP16)

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise SystemExit("engine build failed")
with open(ENGINE, "wb") as f:
    f.write(serialized)
print("wrote", ENGINE)