"""Diagnóstico do ambiente do FaceLab.

Uso (com o venv ativado, a partir da pasta FaceLab):
    python scripts/check_env.py
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import find_executable  # noqa: E402


def ok(msg: str) -> None:
    print(f"  [OK]    {msg}")


def warn(msg: str) -> None:
    print(f"  [AVISO] {msg}")


def fail(msg: str) -> None:
    print(f"  [ERRO]  {msg}")


def check_python() -> None:
    print("Python")
    v = sys.version_info
    msg = f"{platform.python_version()} ({sys.executable})"
    ok(msg) if (v.major, v.minor) == (3, 11) else warn(msg + " — recomendado 3.11")


def check_torch() -> None:
    print("PyTorch")
    try:
        import torch
    except ImportError:
        fail("torch não instalado")
        return
    ok(f"versão {torch.__version__}, CUDA do build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        warn("CUDA indisponível — o processamento usará CPU")
        return
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    ok(f"GPU: {name} ({arch})")
    if arch in archs:
        ok(f"arquitetura {arch} suportada por este build")
    else:
        fail(f"{arch} NÃO está em {archs} — instale um build compatível (ex.: cu126)")


def check_onnxruntime() -> None:
    print("ONNX Runtime")
    try:
        import onnxruntime as ort
    except ImportError:
        warn("onnxruntime não instalado (necessário a partir da Fase 3)")
        return
    providers = ort.get_available_providers()
    ok(f"versão {ort.__version__}, providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        warn("CUDAExecutionProvider ausente — ONNX rodará em CPU")


def check_opencv() -> None:
    print("OpenCV")
    try:
        import cv2
        ok(f"versão {cv2.__version__}")
    except ImportError:
        fail("opencv-python não instalado")


def check_ffmpeg() -> None:
    print("FFmpeg")
    for exe in ("ffmpeg", "ffprobe"):
        path = find_executable(exe)
        if path is None:
            fail(f"{exe} não encontrado (tools/ffmpeg/bin ou PATH)")
            continue
        out = subprocess.run([str(path), "-version"], capture_output=True, text=True)
        ok(f"{out.stdout.splitlines()[0]}  ->  {path}")


if __name__ == "__main__":
    print(f"=== FaceLab — diagnóstico ({platform.system()} {platform.release()}) ===")
    for check in (check_python, check_torch, check_onnxruntime, check_opencv, check_ffmpeg):
        check()

