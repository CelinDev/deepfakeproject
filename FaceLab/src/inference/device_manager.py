"""Gerenciamento do dispositivo de processamento (GPU CUDA ou CPU).

Ideia central: o restante do programa nunca pergunta "tem GPU?" diretamente.
Ele pede ao ``DeviceManager`` o dispositivo a usar, e esta classe decide
com base na preferência do usuário e no que realmente está disponível.
Funciona tanto com PyTorch instalado quanto em builds enxutos baseados apenas
em ONNX Runtime GPU + driver NVIDIA nativo (``nvcuda.dll``).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import torch  # type: ignore[import-not-found]
except Exception:
    torch = None  # type: ignore[assignment]


class _SimpleDevice:
    """Representação leve de dispositivo compatível com ``torch.device``."""

    def __init__(self, spec: str) -> None:
        parts = spec.split(":", 1)
        self.type: str = parts[0]
        self.index: int | None = int(parts[1]) if len(parts) > 1 else None

    def __repr__(self) -> str:
        return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")

    def __eq__(self, other: object) -> bool:
        return getattr(other, "type", None) == self.type and getattr(other, "index", None) == self.index


def _make_device(spec: str) -> Any:
    if torch is not None:
        return torch.device(spec)
    return _SimpleDevice(spec)


def _query_nvidia_driver() -> tuple[str, float, tuple[int, int]] | None:
    """Consulta diretamente a GPU NVIDIA via Driver API (nvcuda.dll no Windows / libcuda.so no Linux)."""
    for lib_name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            nvcuda = ctypes.CDLL(lib_name)
            if nvcuda.cuInit(0) != 0:
                continue
            count = ctypes.c_int(0)
            if nvcuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value <= 0:
                continue
            dev = ctypes.c_int(0)
            if nvcuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
                continue
            name_buf = ctypes.create_string_buffer(256)
            nvcuda.cuDeviceGetName(name_buf, 256, dev)
            total_mem = ctypes.c_size_t(0)
            nvcuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), dev)
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, MINOR = 76
            nvcuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
            nvcuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)
            gpu_name = name_buf.value.decode("utf-8", errors="replace").strip()
            vram_mb = total_mem.value / (1024**2)
            return gpu_name, vram_mb, (major.value, minor.value)
        except Exception:
            continue
    return None


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

    device: object
    name: str
    total_memory_mb: float | None  # None para CPU

    @property
    def is_cuda(self) -> bool:
        return getattr(self.device, "type", "cpu") == "cuda"


class DeviceManager:
    """Seleciona o dispositivo e fornece informações de memória."""

    def __init__(self, preference: DevicePreference = DevicePreference.AUTO) -> None:
        self.preference = preference
        self.info = self._select()

    @staticmethod
    def cuda_usable() -> bool:
        """Retorna True se há GPU NVIDIA compatível disponível para CUDA."""
        if torch is not None:
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            return arch in torch.cuda.get_arch_list()
        drv = _query_nvidia_driver()
        if drv is not None:
            _, _, (major, minor) = drv
            return major >= 6  # Pascal (GTX 10xx) ou superior
        return False

    def _select(self) -> DeviceInfo:
        if self.preference is DevicePreference.CPU:
            return self._cpu_info()

        if self.cuda_usable():
            if torch is not None and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return DeviceInfo(
                    device=_make_device("cuda:0"),
                    name=props.name,
                    total_memory_mb=props.total_memory / 1024**2,
                )
            drv = _query_nvidia_driver()
            if drv is not None:
                gpu_name, vram_mb, _ = drv
                return DeviceInfo(
                    device=_make_device("cuda:0"),
                    name=gpu_name,
                    total_memory_mb=vram_mb,
                )

        if self.preference is DevicePreference.GPU:
            raise DeviceUnavailableError(
                "GPU solicitada, mas CUDA não está disponível ou a GPU não é "
                "suportada. Verifique o driver NVIDIA."
            )
        return self._cpu_info()  # AUTO sem GPU -> CPU

    @staticmethod
    def _cpu_info() -> DeviceInfo:
        return DeviceInfo(device=_make_device("cpu"), name="CPU", total_memory_mb=None)

    def memory_used_mb(self) -> float | None:
        """Memória de GPU alocada (MB), ou None na CPU."""
        if not self.info.is_cuda:
            return None
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.info.device) / 1024**2
        return 0.0

    def release_memory(self) -> None:
        """Devolve ao driver a memória em cache do PyTorch (útil após cada job)."""
        if self.info.is_cuda and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

