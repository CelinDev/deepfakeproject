"""Fonte de captura para Webcam em tempo real (FrameSource).

Implementa a interface abstrata FrameSource utilizando OpenCV VideoCapture com DirectShow (Windows).
Desacoplada de arquivos de vídeo e do FFmpeg, permitindo transmissão contínua de quadros da câmera.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from fractions import Fraction

import cv2

from src.core.interfaces import FrameSource
from src.core.types import Frame


class WebcamSource(FrameSource):
    """Captura quadros ao vivo de uma webcam conectada."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._is_running = False

    def open(self) -> None:
        # No Windows, cv2.CAP_MSMF (Media Foundation) conecta perfeitamente a drivers virtuais como DroidCam e OBS
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera (índice {self.camera_index}). "
                "Verifique se o DroidCam Client está conectado (Start) no PC."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._is_running = True

    def frames(self) -> Iterator[Frame]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Chame open() antes de capturar quadros.")

        index = 0
        t_start = time.perf_counter()
        while self._is_running:
            ret, image = self._cap.read()
            if not ret or image is None:
                time.sleep(0.01)
                continue

            timestamp = time.perf_counter() - t_start
            yield Frame(index=index, timestamp_s=timestamp, image=image)
            index += 1

    def close(self) -> None:
        self._is_running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

