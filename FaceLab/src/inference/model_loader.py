"""Configuração do ONNX Runtime de acordo com o dispositivo escolhido.

Detalhe importante no Windows: o ``onnxruntime-gpu`` precisa das DLLs do
CUDA 12 e do cuDNN 9. Quando executamos no ambiente Python normal, o PyTorch
já traz essas DLLs em ``torch/lib``; quando empacotado pelo instalador,
essas DLLs ficam em ``_internal/torch/lib`` (ou ``_internal/cuda_libs``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Registra os diretórios de DLLs do CUDA/cuDNN antes de importar onnxruntime
_candidate_dll_dirs: list[Path] = []
if getattr(sys, "frozen", False):
    _base = Path(sys.executable).resolve().parent / "_internal"
    _candidate_dll_dirs.extend([_base / "torch" / "lib", _base / "cuda_libs", _base])

for _d in _candidate_dll_dirs:
    if _d.is_dir():
        try:
            os.add_dll_directory(str(_d))
        except Exception:
            pass
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

try:
    import torch  # noqa: F401
except Exception:
    pass

from typing import Any

import numpy as np
import onnxruntime as ort

from config.settings import ORT_CUDNN_CONV_ALGO_SEARCH, ORT_ENABLE_IO_BINDING
from src.inference.device_manager import DeviceManager


def onnx_providers(
    device_manager: DeviceManager,
    cudnn_conv_algo_search: str = ORT_CUDNN_CONV_ALGO_SEARCH,
) -> list[str | tuple[str, dict[str, Any]]]:
    """Lista de providers do ONNX Runtime, em ordem de preferência.

    Quando CUDA está ativo, configura ``cudnn_conv_algo_search="HEURISTIC"`` para evitar
    buscas exaustivas lentas no primeiro quadro e otimiza o reuso de memória da arena CUDA.
    """
    available = ort.get_available_providers()
    if device_manager.info.is_cuda and "CUDAExecutionProvider" in available:
        cuda_options: dict[str, Any] = {
            "device_id": 0,
            "cudnn_conv_algo_search": str(cudnn_conv_algo_search).upper(),
            "arena_extend_strategy": "kSameAsRequested",
        }
        return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


_iobinding_buffer_cache: dict[tuple[int, str, tuple[int, ...]], tuple[np.ndarray, Any]] = {}


def run_with_iobinding(
    session: ort.InferenceSession,
    output_name: str,
    feeds: dict[str, np.ndarray],
    output_shape: tuple[int, ...] | None = None,
    enable_iobinding: bool = ORT_ENABLE_IO_BINDING,
) -> np.ndarray:
    """Executa inferência ONNX usando ``IOBinding`` com buffer de saída pré-alocado e reutilizado."""
    if not enable_iobinding or not hasattr(session, "io_binding"):
        return session.run([output_name], feeds)[0]

    try:
        io_binding = session.io_binding()
        for name, arr in feeds.items():
            contig = np.ascontiguousarray(arr, dtype=np.float32)
            io_binding.bind_cpu_input(name, contig)

        if output_shape is not None:
            cache_key = (id(session), output_name, tuple(output_shape))
            cached = _iobinding_buffer_cache.get(cache_key)
            if cached is None:
                out_buf = np.empty(output_shape, dtype=np.float32)
                ort_out = ort.OrtValue.ortvalue_from_numpy(out_buf)
                _iobinding_buffer_cache[cache_key] = (out_buf, ort_out)
            else:
                out_buf, ort_out = cached
            io_binding.bind_ortvalue_output(output_name, ort_out)
            session.run_with_iobinding(io_binding)
            return out_buf

        io_binding.bind_output(output_name)
        session.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]
    except Exception:
        return session.run([output_name], feeds)[0]

