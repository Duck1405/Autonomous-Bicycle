"""Build a TensorRT engine from any (static-shape) ONNX model, with progress bars.

Usage:
    python onnx_tensorRT.py --onnx models/yolo11s_coco4/run5/yolo11s_coco4.onnx --fp16
    python onnx_tensorRT.py --onnx ../LaneATT/onnxmodels/LaneATTresnet34Aug2/model.onnx --fp16
    python onnx_tensorRT.py --onnx model.onnx --engine custom_name.engine --workspace 8

Omit --fp16 for an FP32 engine (useful as an accuracy baseline when FP16 output
looks wrong). Engines are locked to the TensorRT version AND the GPU arch that
built them, so every build writes a <engine>.json sidecar recording both; check
it before blaming a model for garbage output.

Cluster (x86_64): conda env LaneNetCuda_12_6, on a GPU node (salloc/srun) — the
builder benchmarks kernels on the real device.
Jetson (aarch64): conda env LaneNet310, TensorRT comes from JetPack via apt
(there is no pip TensorRT for Tegra). Defaults drop to a 1 GB workspace there
because CPU and GPU share one 8 GB pool.

Requires: tensorrt >= 10, tqdm (optional — plain logging without it).
"""
import argparse
import ctypes
import hashlib
import json
import platform
import time
from pathlib import Path

import tensorrt as trt

try:
    from tqdm import tqdm
except ImportError:  # progress bars are a nicety, not a dependency
    tqdm = None


def is_tegra():
    """True on Jetson. /etc/nv_tegra_release only exists on L4T."""
    return Path("/etc/nv_tegra_release").exists()


def cuda_free_mb():
    """Free device memory in MB, or None if libcudart isn't loadable."""
    try:
        rt = ctypes.CDLL("libcudart.so")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return free.value / 1e6
    except OSError:
        return None


def compute_capability():
    """(major, minor) of device 0, or None. Engines are pinned to this."""
    try:
        rt = ctypes.CDLL("libcudart.so")
        major, minor = ctypes.c_int(), ctypes.c_int()
        # cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
        if rt.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0) != 0:
            return None
        rt.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0)
        return major.value, minor.value
    except OSError:
        return None


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


def build(onnx_path: Path, engine_path: Path, fp16: bool, workspace_gb: float, verbose: bool,
          hardware_compat: bool = False, timing_cache: Path = None, opt_level: int = None):
    # WARNING level keeps the bars readable; --verbose restores INFO logs.
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    # Registers EfficientNMS_TRT and friends. Ultralytics' nms=True export uses
    # plain ONNX ops so it parses without this, but --end2end exports don't.
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch — only mode in TRT 10
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        if parser.num_errors == 0:
            # A parse failure with zero recorded errors almost always means the
            # file itself is bad — truncated transfer, LFS pointer, wrong path.
            print(f"parser reported no errors; check the file is a complete ONNX "
                  f"({onnx_path.stat().st_size} bytes on disk)")
        raise SystemExit(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        # A missing BuilderFlag.FP16 is not a script bug — the flag has existed
        # since TRT 6. It means the tensorrt / tensorrt_bindings / tensorrt_libs
        # packages are at mismatched versions (or a system TRT shadows the pip one).
        # Fail with that diagnosis instead of a cryptic AttributeError.
        if not hasattr(trt.BuilderFlag, "FP16"):
            hint = ("reinstall the apt packages (sudo apt install --reinstall "
                    "python3-libnvinfer tensorrt) — there is no pip TensorRT for Tegra"
                    if is_tegra() else
                    "pip uninstall tensorrt tensorrt_bindings tensorrt_libs, then "
                    "reinstall a single matching version")
            raise SystemExit(
                f"This TensorRT build has no BuilderFlag.FP16 — the install is "
                f"mismatched. Fix: {hint}. Or fall back to `trtexec --fp16`.")
        config.set_flag(trt.BuilderFlag.FP16)

    if opt_level is not None:
        config.builder_optimization_level = opt_level

    if hardware_compat:
        # Lets one engine run across different GPUs on the SAME platform (e.g. build
        # on an A100 SM80 node, run on an L40S SM89 node) — without it the engine is
        # pinned to the exact build GPU. This does NOT make cluster engines loadable
        # on the Jetson: engines never cross platforms (x86_64 -> aarch64).
        if is_tegra():
            raise SystemExit(
                "--hardware-compat is not supported on JetPack (NVIDIA documents it as "
                "unavailable on Jetson/DRIVE OS). Jetson engines are built for this exact "
                "device; drop the flag.")
        if not hasattr(trt, "HardwareCompatibilityLevel"):
            raise SystemExit(
                "This TensorRT build has no HardwareCompatibilityLevel — need TRT 8.6+.")
        config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS

    # A timing cache makes a retried build skip kernel autotuning it already did.
    # That matters most on Jetson, where a first build can take many minutes.
    if timing_cache is not None:
        blob = timing_cache.read_bytes() if timing_cache.exists() else b""
        config.set_timing_cache(config.create_timing_cache(blob), ignore_mismatch=False)
        print(f"timing cache: {timing_cache} ({len(blob) / 1e3:.1f} kB loaded)")

    if tqdm is not None:
        config.progress_monitor = TQDMProgressMonitor()

    free_before = cuda_free_mb()
    free_note = f", {free_before:.0f} MB GPU free" if free_before else ""
    print(f"building {'FP16' if fp16 else 'FP32'} engine from {onnx_path.name} "
          f"(workspace {workspace_gb} GB{free_note})")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - t0

    if serialized is None:
        free_now = cuda_free_mb()
        mem = (f" GPU free was {free_before:.0f} MB before the build, {free_now:.0f} MB now."
               if free_before and free_now else "")
        raise SystemExit(
            f"engine build returned nothing after {elapsed:.0f}s (failed or cancelled).{mem} "
            "If the process was killed outright instead of returning, that was the OOM "
            "killer: lower --workspace and/or --builder-opt-level.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    if timing_cache is not None:
        timing_cache.write_bytes(memoryview(config.get_timing_cache().serialize()))

    cc = compute_capability()
    sidecar = {
        "onnx": str(onnx_path),
        "onnx_md5": hashlib.md5(onnx_path.read_bytes()).hexdigest(),
        "tensorrt": trt.__version__,
        "compute_capability": f"sm{cc[0]}{cc[1]}" if cc else None,
        "platform": platform.machine(),
        "precision": "fp16" if fp16 else "fp32",
        "workspace_gb": workspace_gb,
        "builder_optimization_level": opt_level,
        "build_seconds": round(elapsed, 1),
    }
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2) + "\n")

    print(f"wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB) in {elapsed:.0f}s")
    print(f"      TensorRT {trt.__version__} / {sidecar['compute_capability']} / "
          f"{platform.machine()} — engine only loads on a matching runtime")


def main():
    tegra = is_tegra()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path, help="input ONNX model")
    ap.add_argument("--engine", type=Path, default=None,
                    help="output engine path (default: <onnx stem>.engine next to the ONNX)")
    ap.add_argument("--fp16", action="store_true", help="enable FP16 kernels (default: FP32)")
    ap.add_argument("--hardware-compat", action="store_true",
                    help="build for Ampere+ hardware compatibility (one engine usable across "
                         "cluster GPU types, e.g. A100 and L40S). Unsupported on Jetson.")
    ap.add_argument("--workspace", type=float, default=1 if tegra else 4,
                    help="workspace pool in GB (default: 1 on Jetson, 4 elsewhere)")
    ap.add_argument("--timing-cache", type=Path, default=None,
                    help="load/save a kernel timing cache here to speed up repeat builds")
    ap.add_argument("--builder-opt-level", type=int, default=None, choices=range(6),
                    help="0-5, TRT default 3. Lower = faster build, less tuned engine.")
    ap.add_argument("--verbose", action="store_true",
                    help="INFO-level TRT logging (noisy alongside the bars)")
    args = ap.parse_args()

    engine = args.engine if args.engine else args.onnx.with_suffix(".engine")
    build(args.onnx, engine, args.fp16, args.workspace, args.verbose,
          args.hardware_compat, args.timing_cache, args.builder_opt_level)


if __name__ == "__main__":
    main()
