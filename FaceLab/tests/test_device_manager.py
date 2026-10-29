"""Testes do DeviceManager — funcionam com ou sem GPU."""
from __future__ import annotations

import pytest
import torch

from src.inference.device_manager import (
    DeviceManager,
    DevicePreference,
    DeviceUnavailableError,
)


def test_cpu_sempre_disponivel() -> None:
    dm = DeviceManager(DevicePreference.CPU)
    assert dm.info.device.type == "cpu"
    assert dm.memory_used_mb() is None


def test_auto_escolhe_gpu_se_utilizavel() -> None:
    dm = DeviceManager(DevicePreference.AUTO)
    esperado = "cuda" if DeviceManager.cuda_usable() else "cpu"
    assert dm.info.device.type == esperado


def test_gpu_forcada_sem_cuda_gera_erro(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simula uma máquina sem GPU, independentemente do hardware real.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError):
        DeviceManager(DevicePreference.GPU)


@pytest.mark.skipif(not DeviceManager.cuda_usable(), reason="GPU CUDA não disponível")
def test_tensor_na_gpu() -> None:
    dm = DeviceManager(DevicePreference.GPU)
    x = torch.ones(1000, 1000, device=dm.info.device)
    assert (x @ x).sum().item() == 1000.0 * 1000 * 1000
    dm.release_memory()


def test_onnx_providers_cudnn_heuristic_and_iobinding() -> None:
    """Garante que onnx_providers configure cudnn_conv_algo_search='HEURISTIC' e run_with_iobinding funcione."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.inference.model_loader import onnx_providers, run_with_iobinding

    dm = DeviceManager(DevicePreference.AUTO)
    provs = onnx_providers(dm, cudnn_conv_algo_search="HEURISTIC")
    if dm.info.is_cuda and isinstance(provs[0], tuple):
        name, opts = provs[0]
        assert name == "CUDAExecutionProvider"
        assert opts.get("cudnn_conv_algo_search") == "HEURISTIC"

    # Testa fallback limpo e execução do run_with_iobinding em sessão mockada
    fake_sess = MagicMock()
    expected = np.ones((1, 3, 128, 128), dtype=np.float32)
    fake_sess.run.return_value = [expected]
    del fake_sess.io_binding  # Simula sessão sem io_binding -> cai para session.run
    res = run_with_iobinding(fake_sess, "output", {"in": np.zeros((1, 3, 128, 128), dtype=np.float32)})
    assert res.shape == (1, 3, 128, 128)


