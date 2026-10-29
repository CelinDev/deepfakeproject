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

import onnxruntime as ort

from src.inference.device_manager import DeviceManager


def onnx_providers(device_manager: DeviceManager) -> list[str]:
    """Lista de providers do ONNX Runtime, em ordem de preferência.

    O TensorRT é ignorado de propósito: exige instalação extra e a GTX 1070
    não tem Tensor Cores, então o ganho seria pequeno.
    """
    available = ort.get_available_providers()
    if device_manager.info.is_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

