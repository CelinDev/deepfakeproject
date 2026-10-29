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

# Parâmetros de máscara alinhada (Fase 2), oclusão, correção de cor e blending
ALIGNED_MASK_SIZE: int = 256            # Resolução fixa no espaço do crop alinhado para geração da máscara semântica
MASK_EMA_ALPHA: float = 0.45            # Peso da nova máscara no EMA temporal (0.45 = suave sem atraso em piscadas/fala)
USE_LEGACY_BBOX_RESTORE: bool = False   # Flag para comparar com o método antigo (restore_natural_hair_and_ears via bbox)
XSEG_MODEL_PATH: Path = MODELS_DIR / "xseg_2.onnx"
COLOR_STD_GAIN_MIN: float = 0.80        # Ganho mínimo permitido na razão de desvios padrão em LAB
COLOR_STD_GAIN_MAX: float = 1.25        # Ganho máximo permitido na razão de desvios padrão em LAB
ENABLE_BORDER_LOWFREQ_COLOR: bool = False  # Correção opcional de baixa frequência na faixa de borda
COMPOSITOR_BLEND_MODE: str = "laplacian"   # Modo de blending: "alpha", "laplacian" (multi-band) ou "lowfreq_border"
LAPLACIAN_PYRAMID_LEVELS: int = 3          # Níveis da pirâmide laplaciana no multi-band blending
WARP_BACK_INTERPOLATION: str = "cubic"     # Interpolação no warp de volta ao frame: "cubic" ou "lanczos4"
WARP_PREFILTER_SHRINK: bool = True         # Aplica pré-filtro gaussiano anti-aliasing quando o rosto no frame é menor que o crop

# Parâmetros do Tracker (Seção 3), Alinhamento 106p (Seção 4) e Modelo de Swap (Seção 5)
TRACKER_VELOCITY_TOLERANCE_FRAMES: int = 5   # Quadros tolerados usando predição de velocidade da bbox no modo sem embedding
TRACKER_EMBEDDING_EMA_ALPHA: float = 0.05    # Taxa de atualização lenta do embedding do alvo quando similaridade é alta
TRACKER_EMBEDDING_UPDATE_SIM: float = 0.60   # Similaridade mínima para atualizar o embedding do alvo com média móvel lenta
USE_LANDMARK_106: bool = True                # Usa landmark_2d_106 do buffalo_l para estimativa de similaridade mais estável
SMOOTH_TRANSFORM_PARAMS: bool = True         # Suaviza parâmetros da transformação (escala, rotação, translação) com One Euro Filter
SWAPPER_MODEL_NAME: str = "inswapper_128"    # Modelo de swap padrão ("inswapper_128" ou "hyperswap_256")
HYPERSWAP_256_MODEL_PATH: Path = MODELS_DIR / "hyperswap_1a_256.onnx"


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
