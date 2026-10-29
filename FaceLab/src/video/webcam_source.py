"""Fonte de captura Assíncrona para Webcam em tempo real (FrameSource).

Implementa a interface abstrata FrameSource utilizando OpenCV VideoCapture com
uma thread leitora dedicada (zero I/O blocking no loop da GPU).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from src.core.interfaces import FrameSource
from src.core.types import Frame


class WebcamSource(FrameSource):
    """Captura quadros ao vivo de uma webcam conectada em thread dedicada (baixa latência)."""

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
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_seq: int = 0
        self._reader_thread: threading.Thread | None = None

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

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._is_running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Lê continuamente da câmera em segundo plano para manter sempre o quadro mais fresco."""
        while self._is_running and self._cap is not None:
            ret, img = self._cap.read()
            if ret and img is not None:
                with self._lock:
                    self._latest_frame = img
                    self._frame_seq += 1
            else:
                time.sleep(0.005)

    def frames(self) -> Iterator[Frame]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Chame open() antes de capturar quadros.")

        index = 0
        last_seq = -1
        t_start = time.perf_counter()
        while self._is_running:
            with self._lock:
                seq = self._frame_seq
                img = None if self._latest_frame is None else self._latest_frame.copy()

            if img is None or seq == last_seq:
                time.sleep(0.002)
                continue

            last_seq = seq
            timestamp = time.perf_counter() - t_start
            yield Frame(index=index, timestamp_s=timestamp, image=img)
            index += 1

    def close(self) -> None:
        self._is_running = False
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

