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
        prefer_mjpg: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.prefer_mjpg = bool(prefer_mjpg)
        self._cap: cv2.VideoCapture | None = None
        self._is_running = False
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp_s: float = 0.0
        self._t_start: float = 0.0
        self._frame_seq: int = 0
        self._dropped_frames: int = 0
        self._last_consumed_seq: int = 0
        self._reader_thread: threading.Thread | None = None

    @property
    def dropped_frames(self) -> int:
        """Total de quadros antigos descartados para manter sempre apenas o último quadro."""
        return self._dropped_frames

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
        if self.prefer_mjpg:
            mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._t_start = time.perf_counter()
        self._is_running = True
        self._new_frame_event.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Lê continuamente da câmera em segundo plano e mantém somente o último quadro (descarta antigos)."""
        while self._is_running and self._cap is not None:
            ret, img = self._cap.read()
            if ret and img is not None:
                now_ts = time.perf_counter() - self._t_start
                with self._lock:
                    if self._frame_seq > self._last_consumed_seq:
                        self._dropped_frames += 1
                    self._latest_frame = img
                    self._latest_timestamp_s = now_ts
                    self._frame_seq += 1
                self._new_frame_event.set()
            else:
                time.sleep(0.003)

    def frames(self) -> Iterator[Frame]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Chame open() antes de capturar quadros.")

        index = 0
        last_seq = -1
        while self._is_running:
            if not self._new_frame_event.wait(timeout=0.05):
                continue
            self._new_frame_event.clear()

            with self._lock:
                seq = self._frame_seq
                if self._latest_frame is None or seq == last_seq:
                    continue
                img = self._latest_frame.copy()
                timestamp = self._latest_timestamp_s
                self._last_consumed_seq = seq

            last_seq = seq
            yield Frame(index=index, timestamp_s=timestamp, image=img)
            index += 1

    def close(self) -> None:
        self._is_running = False
        self._new_frame_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.5)
        self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

