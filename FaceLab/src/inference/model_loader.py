"""Configuração do ONNX Runtime de acordo com o dispositivo escolhido.

Detalhe importante no Windows: o ``onnxruntime-gpu`` precisa das DLLs do
CUDA e do cuDNN. O PyTorch com CUDA já traz essas DLLs; importar o
``torch`` antes do ``onnxruntime`` faz com que elas sejam carregadas no
processo e reaproveitadas, sem instalar o CUDA Toolkit separadamente.
"""
from __future__ import annotations

import torch  # noqa: F401  — precisa vir antes do onnxruntime (ver docstring)
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

