"""Configurações centrais do FaceLab.

Concentra caminhos, formatos aceitos e parâmetros padrão para que nenhum
outro módulo precise "adivinhar" onde estão as coisas.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Raiz do projeto: quando empacotado em executável (PyInstaller), usa a pasta do .exe
if getattr(sys, "frozen", False):
    PROJECT_ROOT: Path = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
TEMP_DIR: Path = PROJECT_ROOT / "temp"
TOOLS_DIR: Path = PROJECT_ROOT / "tools"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
ZOMBIE_REFERENCE_IMAGE: Path = ASSETS_DIR / "zombie_reference.png"
VAMPIRE_REFERENCE_IMAGE: Path = ASSETS_DIR / "vampire_reference.png"
CYBORG_REFERENCE_IMAGE: Path = ASSETS_DIR / "cyborg_reference.png"
BISENET_MODEL_PATH: Path = MODELS_DIR / "bisenet_resnet_34.onnx"

# FFmpeg local do projeto (tools/ffmpeg/bin) tem prioridade sobre o PATH.
LOCAL_FFMPEG_BIN: Path = TOOLS_DIR / "ffmpeg" / "bin"

SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Parâmetros padrão de restauração (GFPGAN) e suavização temporal (One Euro Filter / Hold)
LIVE_ENHANCER_INTERVAL: int = 2         # Executa GFPGAN no crop a cada N quadros na webcam e interpola nos intermediários
ONE_EURO_MIN_CUTOFF: float = 1.5        # Frequência de corte mínima (Hz) para filtrar tremores com rosto parado
ONE_EURO_BETA: float = 0.05             # Coeficiente de velocidade para eliminar atraso (lag) em movimentos rápidos
ONE_EURO_D_CUTOFF: float = 1.0          # Frequência de corte para a derivada da velocidade (Hz)
SCENE_CUT_FACE_RATIO: float = 0.65      # Deslocamento relativo ao tamanho do rosto que indica corte de cena
SMOOTHER_ALPHA_COLOR: float = 0.15      # Peso da nova amostra nas estatísticas de cor (0.15 = alta inércia/estabilidade)
TARGET_HOLD_FRAMES: int = 6             # Quadros de persistência (hold + fade-out) ao perder temporariamente o rosto


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
