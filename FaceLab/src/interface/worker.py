"""Worker em segundo plano para o processamento de vídeo no PySide6.

Executa o VideoPipeline em uma QThread separada para:
1. Nunca travar ou congelar a interface gráfica (GUI).
2. Transmitir atualizações de progresso através de Qt Signals de forma thread-safe.
3. Permitir cancelamento gracioso e seguro da tarefa.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from src.core.types import PipelineCancelled
from src.inference.device_manager import DevicePreference
from src.pipeline.video_pipeline import PipelineProgress, VideoPipeline

logger = logging.getLogger(__name__)


def resize_maintain_aspect(img: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    """Redimensiona ``img`` para caber em ``max_size = (max_w, max_h)`` mantendo a proporção."""
    h, w = img.shape[:2]
    max_w, max_h = max(1, int(max_size[0])), max(1, int(max_size[1]))
    scale = min(float(max_w) / float(w), float(max_h) / float(h))
    if abs(scale - 1.0) < 1e-3:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


class VideoWorker(QObject):
    """Executa o pipeline e emite sinais para a GUI."""

    progress_updated = Signal(PipelineProgress)
    finished = Signal(str)  # Caminho do vídeo final
    error = Signal(str)     # Mensagem de erro

    def __init__(
        self,
        video_path: Path | str,
        image_path: Path | str | None = None,
        output_path: Path | str | None = None,
        device_preference: DevicePreference = DevicePreference.AUTO,
        enable_enhancer: bool = False,
        blend_intensity: float = 1.0,
        color_correct: bool = True,
        feather_amount: int = 15,
        enable_zombie: bool = False,
        target_face_index: int | None = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.image_path = image_path
        self.output_path = output_path
        self.device_preference = device_preference
        self.enable_enhancer = enable_enhancer
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.feather_amount = feather_amount
        self.enable_zombie = enable_zombie
        self.target_face_index = target_face_index
        self.pipeline: VideoPipeline | None = None
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Solicita o cancelamento seguro da execução (mesmo se chamado antes de self.pipeline existir)."""
        self._cancel_event.set()
        if self.pipeline:
            self.pipeline.cancel()

    def run(self) -> None:
        """Ponto de entrada executado dentro da QThread."""
        try:
            if self._cancel_event.is_set():
                raise PipelineCancelled("Processamento cancelado pelo usuário.")

            self.pipeline = VideoPipeline(
                device_preference=self.device_preference,
                on_progress=lambda p: self.progress_updated.emit(p),
                enable_enhancer=self.enable_enhancer,
                blend_intensity=self.blend_intensity,
                color_correct=self.color_correct,
                feather_amount=self.feather_amount,
                enable_zombie=self.enable_zombie,
            )
            if self._cancel_event.is_set():
                self.pipeline.cancel()

            out_file = self.pipeline.process(
                video_path=self.video_path,
                source_image_path=self.image_path,
                output_path=self.output_path,
                target_face_index=self.target_face_index,
            )
            self.finished.emit(str(out_file))
        except PipelineCancelled as exc:
            self.error.emit(str(exc) or "Processamento cancelado pelo usuário.")
        except Exception as exc:
            logger.exception("Erro durante execução do worker:")
            self.error.emit(str(exc))


class LiveWebcamWorker(QObject):
    """Executa o LivePipeline em tempo real dentro de uma QThread separada sem acumular fila na UI."""

    frame_ready = Signal(object, float)  # (np.ndarray BGR já redimensionado, fps)
    error = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        image_path: Path | str | None = None,
        camera_index: int = 0,
        device_preference: DevicePreference = DevicePreference.AUTO,
        blend_intensity: float = 1.0,
        color_correct: bool = True,
        feather_amount: int = 15,
        enable_zombie: bool = False,
        target_face_index: int | None = None,
        display_size: tuple[int, int] = (800, 480),
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.camera_index = camera_index
        self.device_preference = device_preference
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.feather_amount = feather_amount
        self.enable_zombie = enable_zombie
        self.target_face_index = target_face_index
        self.display_size = (int(display_size[0]), int(display_size[1]))
        self.pipeline: LivePipeline | None = None
        self._stop_event = threading.Event()
        self._ui_busy = False
        self.dropped_ui_frames: int = 0

    def stop(self) -> None:
        """Interrompe a captura ao vivo (funciona mesmo se chamado antes de self.pipeline ser instanciado)."""
        self._stop_event.set()
        if self.pipeline:
            self.pipeline.stop()

    def mark_ui_ready(self) -> None:
        """Sinaliza que a thread principal da GUI terminou de desenhar o último quadro emitido."""
        self._ui_busy = False

    def set_display_size(self, width: int, height: int) -> None:
        """Atualiza o tamanho alvo de exibição para o cv2.resize feito na thread do worker."""
        self.display_size = (max(64, int(width)), max(64, int(height)))

    def _emit_frame_if_ui_free(self, frame_bgr: np.ndarray, fps: float) -> None:
        """Redimensiona na thread do worker e descarta o quadro se a UI ainda estiver ocupada."""
        if self._stop_event.is_set():
            if self.pipeline:
                self.pipeline.stop()
            return
        if self._ui_busy:
            self.dropped_ui_frames += 1
            return
        self._ui_busy = True
        resized = resize_maintain_aspect(frame_bgr, self.display_size)
        self.frame_ready.emit(resized, fps)

    def update_params(
        self,
        blend_intensity: float | None = None,
        color_correct: bool | None = None,
        feather_amount: int | None = None,
        enable_zombie: bool | None = None,
    ) -> None:
        """Repassa parâmetros dinâmicos em tempo real para o pipeline ativo."""
        if self.pipeline:
            self.pipeline.update_params(
                blend_intensity=blend_intensity,
                color_correct=color_correct,
                feather_amount=feather_amount,
                enable_zombie=enable_zombie,
            )

    def run(self) -> None:
        """Inicia a transmissão ao vivo."""
        try:
            if self._stop_event.is_set():
                self.stopped.emit()
                return

            from src.pipeline.live_pipeline import LivePipeline

            self.pipeline = LivePipeline(
                device_preference=self.device_preference,
                on_frame_rendered=self._emit_frame_if_ui_free,
                blend_intensity=self.blend_intensity,
                color_correct=self.color_correct,
                feather_amount=self.feather_amount,
                enable_zombie=self.enable_zombie,
            )
            if self._stop_event.is_set():
                self.pipeline.stop()
                self.stopped.emit()
                return

            self.pipeline.run(
                source_image_path=self.image_path,
                camera_index=self.camera_index,
                target_face_index=self.target_face_index,
            )
            self.stopped.emit()
        except Exception as exc:
            logger.exception("Erro durante execução da webcam ao vivo:")
            self.error.emit(str(exc))


