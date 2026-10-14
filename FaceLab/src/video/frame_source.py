"""Fonte de quadros a partir de um arquivo de vídeo (OpenCV)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from src.core.interfaces import FrameSource
from src.core.types import Frame, VideoInfo
from src.video.media_probe import probe_video


class VideoReadError(RuntimeError):
    """Falha ao abrir ou decodificar o vídeo."""


class VideoFileSource(FrameSource):
    """Lê os quadros de um arquivo em ordem, um por vez.

    Ler quadro a quadro (em vez de carregar tudo na memória) permite
    processar vídeos longos: 10 s em 1080p já ocupam ~1,8 GB descomprimidos.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.info: VideoInfo = probe_video(self.path)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoReadError(f"OpenCV não conseguiu abrir {self.path.name}.")
        self._cap = cap

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise VideoReadError("Chame open() antes de frames().")
        index = 0
        fps = float(self.info.fps)
        while True:
            ok, image = self._cap.read()
            if not ok or image is None:
                break  # fim do vídeo (ou quadro corrompido no final)
            # O timestamp é calculado pelo índice para manter a mesma base de
            # tempo usada pelo FFmpeg ao gravar (taxa constante).
            yield Frame(index=index, timestamp_s=index / fps, image=image)
            index += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

