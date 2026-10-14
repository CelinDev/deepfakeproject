"""Gerenciamento do dispositivo de processamento (GPU CUDA ou CPU).

Ideia central: o restante do programa nunca pergunta "tem GPU?" diretamente.
Ele pede ao ``DeviceManager`` o dispositivo a usar, e esta classe decide
com base na preferência do usuário e no que realmente está disponível.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class DevicePreference(str, Enum):
    """Preferência escolhida pelo usuário na interface."""

    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


class DeviceUnavailableError(RuntimeError):
    """Lançada quando o usuário exige GPU mas ela não pode ser usada."""


@dataclass(frozen=True)
class DeviceInfo:
    """Resumo do dispositivo selecionado."""

    device: torch.device
    name: str
    total_memory_mb: float | None  # None para CPU

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


class DeviceManager:
    """Seleciona o dispositivo e fornece informações de memória."""

    def __init__(self, preference: DevicePreference = DevicePreference.AUTO) -> None:
        self.preference = preference
        self.info = self._select()

    @staticmethod
    def cuda_usable() -> bool:
        """Retorna True se o PyTorch tem CUDA e a GPU é suportada por este build.

        ``torch.cuda.is_available()`` pode ser True mesmo quando a arquitetura
        da GPU (ex.: sm_61 da GTX 1070) não foi compilada no PyTorch instalado,
        por isso conferimos também a lista de arquiteturas.
        """
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        return arch in torch.cuda.get_arch_list()

    def _select(self) -> DeviceInfo:
        if self.preference is DevicePreference.CPU:
            return self._cpu_info()

        if self.cuda_usable():
            props = torch.cuda.get_device_properties(0)
            return DeviceInfo(
                device=torch.device("cuda:0"),
                name=props.name,
                total_memory_mb=props.total_memory / 1024**2,
            )

        if self.preference is DevicePreference.GPU:
            raise DeviceUnavailableError(
                "GPU solicitada, mas CUDA não está disponível ou a GPU não é "
                "suportada pelo PyTorch instalado. Execute scripts/check_env.py."
            )
        return self._cpu_info()  # AUTO sem GPU -> CPU

    @staticmethod
    def _cpu_info() -> DeviceInfo:
        return DeviceInfo(device=torch.device("cpu"), name="CPU", total_memory_mb=None)

    def memory_used_mb(self) -> float | None:
        """Memória de GPU alocada pelo PyTorch (MB), ou None na CPU."""
        if not self.info.is_cuda:
            return None
        return torch.cuda.memory_allocated(self.info.device) / 1024**2

    def release_memory(self) -> None:
        """Devolve ao driver a memória em cache do PyTorch (útil após cada job)."""
        if self.info.is_cuda:
            torch.cuda.empty_cache()

