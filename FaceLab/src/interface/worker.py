"""Worker em segundo plano para o processamento de vídeo no PySide6.

Executa o VideoPipeline em uma QThread separada para:
1. Nunca travar ou congelar a interface gráfica (GUI).
2. Transmitir atualizações de progresso através de Qt Signals de forma thread-safe.
3. Permitir cancelamento gracioso e seguro da tarefa.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from src.inference.device_manager import DevicePreference
from src.pipeline.video_pipeline import PipelineProgress, VideoPipeline

logger = logging.getLogger(__name__)


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
        self.pipeline: VideoPipeline | None = None

    def cancel(self) -> None:
        """Solicita o cancelamento seguro da execução."""
        if self.pipeline:
            self.pipeline.cancel()

    def run(self) -> None:
        """Ponto de entrada executado dentro da QThread."""
        try:
            self.pipeline = VideoPipeline(
                device_preference=self.device_preference,
                on_progress=lambda p: self.progress_updated.emit(p),
                enable_enhancer=self.enable_enhancer,
                blend_intensity=self.blend_intensity,
                color_correct=self.color_correct,
                feather_amount=self.feather_amount,
                enable_zombie=self.enable_zombie,
            )
            out_file = self.pipeline.process(
                video_path=self.video_path,
                source_image_path=self.image_path,
                output_path=self.output_path,
            )
            self.finished.emit(str(out_file))
        except KeyboardInterrupt:
            self.error.emit("Processamento cancelado pelo usuário.")
        except Exception as exc:
            logger.exception("Erro durante execução do worker:")
            self.error.emit(str(exc))


class LiveWebcamWorker(QObject):
    """Executa o LivePipeline em tempo real dentro de uma QThread separada."""

    frame_ready = Signal(object, float)  # (np.ndarray BGR, fps)
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
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.camera_index = camera_index
        self.device_preference = device_preference
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.feather_amount = feather_amount
        self.enable_zombie = enable_zombie
        self.pipeline: LivePipeline | None = None

    def stop(self) -> None:
        """Interrompe a captura ao vivo."""
        if self.pipeline:
            self.pipeline.stop()

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
            from src.pipeline.live_pipeline import LivePipeline

            self.pipeline = LivePipeline(
                device_preference=self.device_preference,
                on_frame_rendered=lambda frame, fps: self.frame_ready.emit(frame, fps),
                blend_intensity=self.blend_intensity,
                color_correct=self.color_correct,
                feather_amount=self.feather_amount,
                enable_zombie=self.enable_zombie,
            )
            self.pipeline.run(
                source_image_path=self.image_path,
                camera_index=self.camera_index,
            )
            self.stopped.emit()
        except Exception as exc:
            logger.exception("Erro durante execução da webcam ao vivo:")
            self.error.emit(str(exc))


