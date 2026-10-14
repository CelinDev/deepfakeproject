"""Tipos de dados compartilhados entre os módulos do FaceLab.

São classes simples (dataclasses) que carregam informação de um módulo
para outro sem depender de nenhuma biblioteca específica de vídeo.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Metadados de um vídeo, obtidos com ffprobe.

    ``fps`` é uma ``Fraction`` (ex.: 30000/1001) e não um float, para que a
    taxa de quadros seja repassada ao FFmpeg sem erro de arredondamento.
    """

    path: Path
    width: int
    height: int
    fps: Fraction
    duration_s: float
    frame_count: int | None  # nem todo container informa
    video_codec: str
    has_audio: bool
    audio_codec: str | None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass
class Frame:
    """Um quadro de vídeo.

    Attributes:
        index: posição do quadro na sequência (0, 1, 2...).
        timestamp_s: instante do quadro em segundos.
        image: matriz BGR (altura x largura x 3), formato padrão do OpenCV.
    """

    index: int
    timestamp_s: float
    image: np.ndarray


@dataclass
class DetectedFace:
    """Um rosto detectado num quadro, em formato independente da biblioteca.

    Attributes:
        bbox: caixa (x1, y1, x2, y2) em pixels, float32, shape (4,).
        landmarks: 5 pontos (olho esq., olho dir., nariz, boca esq., boca dir.),
            shape (5, 2). São usados para alinhar o rosto.
        score: confiança da detecção (0 a 1).
        embedding: vetor de identidade normalizado (norma 1), ou None se o
            modelo de reconhecimento não foi carregado. Dois rostos da mesma
            pessoa têm embeddings com produto escalar alto.
    """

    bbox: np.ndarray
    landmarks: np.ndarray | None
    score: float
    embedding: np.ndarray | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

