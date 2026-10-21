"""Configurações centrais do FaceLab.

Concentra caminhos, formatos aceitos e parâmetros padrão para que nenhum
outro módulo precise "adivinhar" onde estão as coisas.
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Raiz do projeto (pasta FaceLab/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
TEMP_DIR: Path = PROJECT_ROOT / "temp"
TOOLS_DIR: Path = PROJECT_ROOT / "tools"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
ZOMBIE_REFERENCE_IMAGE: Path = ASSETS_DIR / "zombie_reference.png"
BISENET_MODEL_PATH: Path = MODELS_DIR / "bisenet_resnet_34.onnx"

# FFmpeg local do projeto (tools/ffmpeg/bin) tem prioridade sobre o PATH.
LOCAL_FFMPEG_BIN: Path = TOOLS_DIR / "ffmpeg" / "bin"

SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def find_executable(name: str) -> Path | None:
    """Localiza um executável (ex.: 'ffmpeg', 'ffprobe').

    Ordem de busca: pasta local ``tools/ffmpeg/bin`` e depois o PATH do sistema.
    Retorna ``None`` se não encontrar.
    """
    local = LOCAL_FFMPEG_BIN / f"{name}.exe"
    if local.is_file():
        return local
    found = shutil.which(name)
    return Path(found) if found else None

