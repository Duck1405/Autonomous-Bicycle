"""Build a TensorRT engine from any (static-shape) ONNX model, with progress bars.

Usage:
    python onnx_tensorRT.py --onnx models/yolo11s_coco4/run5/yolo11s_coco4.onnx --fp16
    python onnx_tensorRT.py --onnx ../LaneATT/onnxmodels/LaneATTresnet34Aug2/model.onnx --fp16
    python onnx_tensorRT.py --onnx model.onnx --engine custom_name.engine --workspace 8

Requires: pip install tensorrt==10.3.0 tqdm
Run on a GPU node (salloc/srun) — the builder benchmarks kernels on the device.
"""
import argparse
from pathlib import Path

import tensorrt as trt
from tqdm import tqdm


class TQDMProgressMonitor(trt.IProgressMonitor):
    """Renders TensorRT's build phases as nested tqdm bars.

    TRT calls phase_start/step_complete/phase_finish as it works through
    parsing, graph optimization, and kernel timing. Returning False from
    step_complete cancels the build, so Ctrl-C aborts cleanly instead of
    leaving a half-dead process.
    """

    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active = {}  # phase_name -> {"bar": tqdm, "parent": str | None}
        self._keep_going = True

    def phase_start(self, phase_name, parent_phase, num_steps):
        try:
            self._active[phase_name] = {
                "bar": tqdm(total=num_steps, desc=phase_name,
                            position=self._depth(parent_phase), leave=False),
                "parent": parent_phase,
            }
        except KeyboardInterrupt:
            self._keep_going = False

    def step_complete(self, phase_name, step):
        try:
            entry = self._active.get(phase_name)
            if entry:
                entry["bar"].update(step - entry["bar"].n)
            return self._keep_going
        except KeyboardInterrupt:
            self._keep_going = False
            return False

    def phase_finish(self, phase_name):
        entry = self._active.pop(phase_name, None)
        if entry:
            entry["bar"].close()

    def _depth(self, parent):
        d = 0
        while parent is not None:
            d += 1
            parent = self._active.get(parent, {}).get("parent")
        return d


def build(onnx_path: Path, engine_path: Path, fp16: bool, workspace_gb: int, verbose: bool):
    # WARNING level keeps the bars readable; --verbose restores INFO logs.
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(  logger)
    network = builder.create_network(0)  # explicit batch — only mode in TRT 10
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    monitor = TQDMProgressMonitor()
    config.progress_monitor = monitor

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed (or cancelled)")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path, help="input ONNX model")
    ap.add_argument("--engine", type=Path, default=None,
                    help="output engine path (default: <onnx stem>.engine next to the ONNX)")
    ap.add_argument("--fp16", action="store_true", help="enable FP16 kernels")
    ap.add_argument("--workspace", type=int, default=4, help="workspace pool in GB (default 4)")
    ap.add_argument("--verbose", action="store_true",
                    help="INFO-level TRT logging (noisy alongside the bars)")
    args = ap.parse_args()

    engine = args.engine if args.engine else args.onnx.with_suffix(".engine")
    build(args.onnx, engine, args.fp16, args.workspace, args.verbose)


if __name__ == "__main__":
    main()