"""Inspect a built .engine and time its raw forward pass.

    python jetson_tools/trt_engine_probe.py LaneATT/onnxmodels/YoloN/yolo11n_coco4_nms.engine
    python jetson_tools/trt_engine_probe.py *.engine --iters 200

Reports the IO contract (so you can confirm it matches the engines built on the
cluster), device memory footprint, the .engine.json sidecar, and latency on
dummy input. Latency here is the pure engine forward — no video decode, no
preprocessing — so it is the floor the real pipeline can never beat.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_runner import CudaRT, TrtEngine  # noqa: E402


def probe(path, iters, warmup, cuda):
    print(f"\n=== {path} ===")
    size_mb = path.stat().st_size / 1e6
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        print(f"built by TensorRT {meta.get('tensorrt')} / {meta.get('compute_capability')} "
              f"/ {meta.get('platform')}, {meta.get('precision')}, "
              f"{meta.get('build_seconds')}s")
    else:
        print("no .engine.json sidecar — provenance unknown")

    free_before, _ = cuda.mem_info()
    eng = TrtEngine(path, cuda=cuda)
    free_after, _ = cuda.mem_info()
    print(f"engine file {size_mb:.1f} MB, device memory +{(free_before - free_after) / 1e6:.0f} MB")
    for line in eng.describe():
        print(f"  {line}")

    arr = np.random.rand(*eng.shapes[eng.input_name]).astype(eng.dtypes[eng.input_name])
    for _ in range(warmup):
        eng.run(arr)

    t0 = time.perf_counter()
    for _ in range(iters):
        eng.run(arr)
    per_iter = (time.perf_counter() - t0) / iters

    outs = eng.infer(arr)
    stats = []
    for name, val in outs.items():
        finite = np.isfinite(val)
        stats.append(f"{name}: min {val[finite].min():.4g} max {val[finite].max():.4g}"
                     + ("" if finite.all() else f"  {(~finite).sum()} NON-FINITE"))
    print(f"forward: {per_iter * 1000:.2f} ms/iter -> {1 / per_iter:.1f} FPS "
          f"(engine only, {iters} iters)")
    for s in stats:
        print(f"  out {s}")
    eng.close()
    return per_iter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engines", nargs="+", type=Path)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cuda = CudaRT()
    free, total = cuda.mem_info()
    cc = cuda.compute_capability()
    arch = f"sm{cc[0]}{cc[1]}" if cc else "sm?"
    print(f"device: {arch}, {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB")

    total_s = 0.0
    for path in args.engines:
        if not path.exists():
            print(f"\n=== {path} ===\nmissing — skipped")
            continue
        total_s += probe(path, args.iters, args.warmup, cuda)

    if total_s:
        print(f"\nall engines sequentially: {total_s * 1000:.2f} ms -> "
              f"{1 / total_s:.1f} FPS (forward passes only)")


if __name__ == "__main__":
    main()
