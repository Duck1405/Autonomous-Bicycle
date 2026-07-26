"""Torch-free TensorRT 10 runtime: ctypes + libcudart for buffers.

The cluster benchmark used torch CUDA tensors as engine IO buffers. That is a
multi-GB dependency the Jetson does not need and has historically failed on it
(see test.txt: pip torch 2.11.0/cu13 reports CUDA available: False). Everything
here talks to libcudart directly, so the only requirements are tensorrt and
numpy.

Static shapes only — all Pathfinder models export with fixed input dims, and an
engine with a dynamic dim raises rather than silently allocating the wrong size.
"""
import ctypes

import numpy as np
import tensorrt as trt

# cudaMemcpyKind
_H2D = 1
_D2H = 2

TRT_TO_NUMPY = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool_,
}


class CudaRT:
    """Thin ctypes binding for the handful of libcudart calls we need."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        # Without explicit argtypes, ctypes truncates 64-bit pointers/sizes to int.
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                            ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, rc, what):
        if rc != 0:
            msg = self.lib.cudaGetErrorString(rc)
            raise RuntimeError(f"{what} failed: rc={rc} {msg.decode() if msg else ''}")

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), f"cudaMalloc({nbytes})")
        return ptr

    def free(self, ptr):
        self.lib.cudaFree(ptr)

    def htod(self, dst, arr):
        src = arr.ctypes.data_as(ctypes.c_void_p)
        self.check(self.lib.cudaMemcpy(dst, src, arr.nbytes, _H2D), "cudaMemcpy H2D")

    def dtoh(self, arr, src):
        dst = arr.ctypes.data_as(ctypes.c_void_p)
        self.check(self.lib.cudaMemcpy(dst, src, arr.nbytes, _D2H), "cudaMemcpy D2H")

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def mem_info(self):
        """(free_bytes, total_bytes)."""
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
                   "cudaMemGetInfo")
        return free.value, total.value

    def compute_capability(self, device=0):
        major, minor = ctypes.c_int(), ctypes.c_int()
        # cudaDevAttrComputeCapabilityMajor = 75, Minor = 76
        if self.lib.cudaDeviceGetAttribute(ctypes.byref(major), 75, device) != 0:
            return None
        self.lib.cudaDeviceGetAttribute(ctypes.byref(minor), 76, device)
        return major.value, minor.value


class TrtEngine:
    """A deserialized static-shape TRT 10 engine with preallocated device IO."""

    def __init__(self, path, cuda=None, logger_severity=trt.Logger.ERROR):
        self.cuda = cuda or CudaRT()
        logger = trt.Logger(logger_severity)
        trt.init_libnvinfer_plugins(logger, "")
        blob = open(path, "rb").read()
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                f"could not deserialize {path}. Engines are locked to the TensorRT "
                f"version and GPU arch that built them; this runtime is TensorRT "
                f"{trt.__version__}. Check the .engine.json sidecar.")
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.stream()

        self.inputs, self.outputs, self.shapes, self.dtypes = {}, {}, {}, {}
        self._device = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"{path}: tensor '{name}' has dynamic shape {shape}; this runner "
                    "only handles static-shape engines.")
            dtype = TRT_TO_NUMPY[self.engine.get_tensor_dtype(name)]
            host = np.zeros(shape, dtype=dtype)
            ptr = self.cuda.malloc(host.nbytes)
            self.context.set_tensor_address(name, ptr.value)
            self._device[name] = ptr
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = host
            else:
                self.outputs[name] = host
        self.input_name = next(iter(self.inputs))

    def run(self, arr):
        """H2D copy of a single input + execute + sync. No D2H — benchmark path."""
        if arr.dtype != self.dtypes[self.input_name]:
            arr = arr.astype(self.dtypes[self.input_name])
        self.cuda.htod(self._device[self.input_name], np.ascontiguousarray(arr))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("execute_async_v3 returned False")
        self.cuda.sync(self.stream)

    def infer(self, arr):
        """run() plus D2H of every output. Returns {name: ndarray} (views reused)."""
        self.run(arr)
        for name, host in self.outputs.items():
            self.cuda.dtoh(host, self._device[name])
        return self.outputs

    def close(self):
        for ptr in self._device.values():
            self.cuda.free(ptr)
        self._device.clear()

    def describe(self):
        io = []
        for name, shape in self.shapes.items():
            kind = "in " if name in self.inputs else "out"
            io.append(f"{kind} {name}{list(shape)} {np.dtype(self.dtypes[name]).name}")
        return io
