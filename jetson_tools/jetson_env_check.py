"""Preflight for TensorRT work on the Jetson. Exits nonzero if anything blocks.

    python jetson_tools/jetson_env_check.py
    python jetson_tools/jetson_env_check.py --video LaneATT/video_input/IMG_6540.MOV

Run this inside the LaneNet310 conda env before building or benchmarking. It
checks the things that have actually bitten this project rather than a generic
"is CUDA there" smoke test: the dist-packages numpy stub, a broken system cv2,
an OpenCV without video decode, and whether TensorRT is importable at all
without PYTHONPATH games.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

FAIL, WARN, OK = "FAIL", "WARN", "ok  "
_results = []


def record(level, label, detail):
    _results.append((level, label, detail))
    print(f"[{level}] {label}: {detail}")


def check_platform():
    import platform
    record(OK, "platform", f"{platform.machine()} python {platform.python_version()} "
                           f"({sys.executable})")
    rel = Path("/etc/nv_tegra_release")
    if not rel.exists():
        record(WARN, "L4T", "not a Jetson (/etc/nv_tegra_release missing) — "
                            "engine builds here will not run on the bike")
        return False
    record(OK, "L4T", rel.read_text().splitlines()[0].lstrip("# "))
    try:
        out = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
                             capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            record(OK, "JetPack", out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return True


def check_power():
    if not shutil.which("nvpmodel"):
        return
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=20)
        mode = " ".join(out.stdout.split())
        level = WARN if "15W" in mode or "7W" in mode else OK
        record(level, "power mode", f"{mode}"
               + ("  (MAXN_SUPER is mode 2; benchmarks at 15W understate the bike)"
                  if level == WARN else ""))
    except (OSError, subprocess.SubprocessError) as e:
        record(WARN, "power mode", f"could not query: {e}")


def check_dla():
    # Orin Nano has no DLA; passing --useDLACore anywhere would fail confusingly.
    has = bool(list(Path("/sys/class").glob("nvdla*")))
    record(OK, "DLA", "present" if has else "absent (expected on Orin Nano — never use DLA flags)")


def check_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as e:
        record(FAIL, "tensorrt", f"import failed: {e}. Symlink the apt bindings into this "
                                 f"env's site-packages; there is no pip TensorRT for Tegra.")
        return None
    where = Path(trt.__file__).resolve()
    record(OK, "tensorrt", f"{trt.__version__} ({where})")
    return trt


def check_cuda():
    try:
        rt = ctypes.CDLL("libcudart.so")
    except OSError as e:
        record(FAIL, "libcudart", str(e))
        return
    rt.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.POINTER(ctypes.c_size_t)]
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        record(FAIL, "cudaMemGetInfo", "failed — CUDA runtime is not usable")
        return
    free_gb, total_gb = free.value / 2**30, total.value / 2**30
    level = OK if free_gb > 2.0 else WARN
    record(level, "GPU memory", f"{free_gb:.2f} GiB free of {total_gb:.2f} GiB "
                                f"(unified with CPU RAM)")
    major, minor = ctypes.c_int(), ctypes.c_int()
    if rt.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0) == 0:
        rt.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0)
        record(OK, "compute capability", f"sm{major.value}{minor.value} "
                                         f"(engines are locked to this)")
    # A real allocation, because mem_get_info succeeds even when allocation is broken
    # (L4T 36.4.7 has a reported "unable to allocate CUDA0 buffer" firmware bug).
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaFree.argtypes = [ctypes.c_void_p]
    ptr = ctypes.c_void_p()
    rc = rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(1024 * 1024 * 1024))
    if rc == 0:
        rt.cudaFree(ptr)
        record(OK, "1 GiB cudaMalloc", "succeeded")
    else:
        record(FAIL, "1 GiB cudaMalloc", f"rc={rc} — allocation is broken, not just tight")


def check_python_deps():
    for mod in ("numpy", "cv2", "tqdm", "onnxruntime"):
        try:
            m = __import__(mod)
        except ImportError as e:
            level = FAIL if mod in ("numpy", "cv2") else WARN
            record(level, mod, f"missing ({e})")
            continue
        ver = getattr(m, "__version__", None)
        if ver is None:
            # Exactly the dist-packages numpy-stub failure: a namespace package
            # with no __init__.py imports fine but has no attributes.
            record(FAIL, mod, f"imported from {getattr(m, '__path__', '?')} but has no "
                              f"__version__ — this is a namespace-package stub, not the "
                              f"real module. Check PYTHONPATH.")
            continue
        record(OK, mod, f"{ver} ({getattr(m, '__file__', '?')})")


def check_opencv_video(video):
    try:
        import cv2
    except ImportError:
        return
    info = cv2.getBuildInformation()
    flags = []
    for key in ("FFMPEG", "GStreamer", "CUDA"):
        line = next((l for l in info.splitlines() if l.strip().startswith(key)), "")
        flags.append(f"{key}={'YES' if 'YES' in line else 'NO'}")
    record(OK, "opencv build", ", ".join(flags))

    if video is None:
        return
    if not Path(video).exists():
        record(WARN, "video decode", f"{video} not found — skipped")
        return
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not ok:
        record(FAIL, "video decode", f"cannot read a frame from {video} — OpenCV has no "
                                     f"usable H.264 decoder")
    else:
        record(OK, "video decode", f"{frame.shape[1]}x{frame.shape[0]}, {int(n)} frames")


def check_headroom():
    st = os.statvfs("/")
    free_gb = st.f_bavail * st.f_frsize / 2**30
    record(OK if free_gb > 5 else WARN, "disk free", f"{free_gb:.1f} GiB on /")
    try:
        info = dict(
            (p[0].rstrip(":"), int(p[1]))
            for p in (l.split() for l in Path("/proc/meminfo").read_text().splitlines())
        )
        record(OK, "RAM", f"{info['MemAvailable'] / 2**20:.1f} GiB available of "
                          f"{info['MemTotal'] / 2**20:.1f} GiB, "
                          f"swap {info['SwapFree'] / 2**20:.1f} GiB free")
    except (OSError, KeyError, ValueError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="LaneATT/video_input/IMG_6540.MOV",
                    help="video to test-decode (set to '' to skip)")
    args = ap.parse_args()

    print("=== Jetson / TensorRT environment check ===")
    check_platform()
    check_power()
    check_dla()
    check_tensorrt()
    check_cuda()
    check_python_deps()
    check_opencv_video(args.video or None)
    check_headroom()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print(f"\n{len(_results)} checks: {len(fails)} failed, {len(warns)} warnings")
    for _, label, detail in fails:
        print(f"  BLOCKING - {label}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
