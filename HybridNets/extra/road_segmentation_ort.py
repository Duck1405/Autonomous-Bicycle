from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def build_provider_stack(
    preferred_provider: str,
    available_providers: list[str],
    engine_cache_dir: Path | None,
    enable_trt_fp16: bool,
    trt_workspace_size_bytes: int,
) -> tuple[list[str], list[dict[str, str]]]:
    providers: list[str] = []
    provider_options: list[dict[str, str]] = []

    if preferred_provider in {"auto", "tensorrt"}:
        if "TensorrtExecutionProvider" in available_providers:
            if engine_cache_dir is not None:
                engine_cache_dir.mkdir(parents=True, exist_ok=True)
            providers.append("TensorrtExecutionProvider")
            provider_options.append(
                {
                    "trt_fp16_enable": "1" if enable_trt_fp16 else "0",
                    "trt_engine_cache_enable": "1",
                    "trt_engine_cache_path": str(engine_cache_dir) if engine_cache_dir is not None else ".",
                    "trt_max_workspace_size": str(max(int(trt_workspace_size_bytes), 1)),
                }
            )
        elif preferred_provider == "tensorrt":
            raise RuntimeError(
                "TensorRT execution was requested, but this onnxruntime build does not provide "
                "TensorrtExecutionProvider."
            )

    if preferred_provider in {"auto", "cuda"}:
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            provider_options.append({})
        elif preferred_provider == "cuda":
            raise RuntimeError(
                "CUDA execution was requested, but this onnxruntime build does not provide "
                "CUDAExecutionProvider."
            )

    if preferred_provider == "cpu":
        providers.append("CPUExecutionProvider")
        provider_options.append({})
    elif "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
        provider_options.append({})

    if not providers:
        raise RuntimeError(
            "No compatible onnxruntime execution providers were available. "
            f"Available providers: {available_providers}"
        )
    return providers, provider_options


class OnnxRoadSegmentationSession:
    def __init__(
        self,
        model_path: Path,
        provider: str = "auto",
        engine_cache_dir: Path | None = None,
        enable_trt_fp16: bool = True,
        trt_workspace_size_bytes: int = 1 << 30,
        intra_op_threads: int = 0,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.available_providers = ort.get_available_providers()
        self.provider, self.engine_cache_dir = provider, engine_cache_dir

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            session_options.intra_op_num_threads = intra_op_threads

        providers, provider_options = build_provider_stack(
            preferred_provider=provider,
            available_providers=self.available_providers,
            engine_cache_dir=engine_cache_dir,
            enable_trt_fp16=enable_trt_fp16,
            trt_workspace_size_bytes=trt_workspace_size_bytes,
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )
        self.active_providers = self.session.get_providers()
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.input_shape = tuple(self.input_meta.shape)
        self.output_shape = tuple(self.output_meta.shape)

    def describe(self) -> str:
        return (
            f"providers={self.active_providers} "
            f"input={self.input_name}{self.input_shape} "
            f"output={self.output_name}{self.output_shape}"
        )

    def get_static_input_hw(self) -> tuple[int, int] | None:
        if len(self.input_shape) != 4:
            return None
        height, width = self.input_shape[2], self.input_shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return int(height), int(width)
        return None

    def infer(self, input_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(input_tensor, torch.Tensor):
            input_array = input_tensor.detach().cpu().numpy()
        else:
            input_array = np.asarray(input_tensor)
        input_array = np.ascontiguousarray(input_array.astype(np.float32, copy=False))
        if input_array.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor, got shape {input_array.shape}")
        return self.session.run([self.output_name], {self.input_name: input_array})[0]

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int) -> None:
        if runs <= 0:
            return
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(runs):
            self.infer(dummy)
