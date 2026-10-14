"""LivePipeline — Pipeline Modular para Processamento de Vídeo em Tempo Real (Webcam).

Diferenças arquiteturais em relação ao VideoPipeline:
1. Fonte / Destino: utiliza WebcamSource (ou gerador de stream) em vez de arquivos MP4.
2. Latência Zero de Gravação: Não há FFmpeg ou codificação H.264 pesada no laço crítico;
   os quadros processados são enviados via callback em tempo real para a UI ou exibidos via OpenCV.
3. Rastreamento e Alinhamento Otimizados: Reutiliza o TargetFaceTracker, FaceAligner,
   INSwapperModel e o TemporalSmoother desenvolvidos nas Fases anteriores.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from src.core.interfaces import FrameSource
from src.core.types import DetectedFace, Frame
from src.face.face_compositor import FaceCompositor
from src.face.face_detector import InsightFaceDetector
from src.face.face_swapper import INSwapperModel
from src.face.face_tracker import TargetFaceTracker
from src.face.temporal_smoother import TemporalSmoother
from src.inference.device_manager import DeviceManager, DevicePreference
from src.utils.validators import validate_image_path
from src.video.webcam_source import WebcamSource

logger = logging.getLogger(__name__)


class LivePipeline:
    """Orquestrador de face swap ao vivo para webcam com baixa latência."""

    def __init__(
        self,
        device_preference: DevicePreference = DevicePreference.AUTO,
        on_frame_rendered: Callable[[np.ndarray, float], None] | None = None,
        blend_intensity: float = 1.0,
        color_correct: bool = True,
        feather_amount: int = 15,
    ) -> None:
        """
        Args:
            device_preference: Preferência de hardware (GPU/CPU).
            on_frame_rendered: Callback que recebe (frame_bgr, fps_atual) para atualização em tempo real.
            blend_intensity: Força de mistura do rosto [0.0 a 1.0].
            color_correct: Ajuste automático de iluminação e cor.
            feather_amount: Suavização da borda da máscara.
        """
        self.device_manager = DeviceManager(device_preference)
        # Modo rápido: detecção com landmarks e sem cálculo obrigatório de embedding em todo quadro
        self.detector = InsightFaceDetector(self.device_manager)
        compositor = FaceCompositor(blur_amount=feather_amount)
        self.swapper = INSwapperModel(self.device_manager, compositor=compositor)
        self.smoother = TemporalSmoother(alpha_landmarks=0.65, alpha_color=0.8)
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.on_frame_rendered = on_frame_rendered
        self._is_running = False

    def stop(self) -> None:
        """Interrompe a execução contínua da webcam."""
        self._is_running = False

    def load_source_face(self, source_image_path: str | Path) -> DetectedFace:
        """Extrai o embedding e marcos do rosto novo de referência."""
        path = validate_image_path(source_image_path)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {path}")

        faces = self.detector.detect(img)
        if not faces:
            raise ValueError(f"Nenhum rosto foi detectado na imagem de referência: {path.name}")

        source_face = TargetFaceTracker.select_largest(faces)
        assert source_face is not None
        if source_face.embedding is None:
            raise ValueError("Não foi possível extrair o embedding facial da imagem de referência.")
        return source_face

    def run(
        self,
        source_image_path: str | Path,
        camera_source: FrameSource | None = None,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """Inicia o laço de captura, troca facial e renderização em tempo real."""
        source_face = self.load_source_face(source_image_path)
        source = camera_source or WebcamSource(camera_index=camera_index, width=width, height=height)

        tracker: TargetFaceTracker | None = None
        self._is_running = True
        self.smoother.reset()

        frame_count = 0
        fps = 0.0
        t_last = time.perf_counter()

        logger.info("Iniciando LivePipeline (Webcam)...")
        with source:
            for frame in source.frames():
                if not self._is_running:
                    break

                t0 = time.perf_counter()
                # Em tempo real, busca embedding apenas a cada 8 quadros para economizar tempo
                need_embedding = (tracker is None) or (frame_count % 8 == 0) or (tracker._missing > 0)
                faces = self.detector.detect(frame.image, compute_embeddings=need_embedding)

                # Inicializa tracker com a pessoa que estiver em frente à câmera
                if tracker is None and faces:
                    alvo = TargetFaceTracker.select_largest(faces)
                    if alvo and alvo.embedding is not None:
                        tracker = TargetFaceTracker(alvo.embedding)

                target = tracker.update(faces) if tracker else None

                if target is not None and target.landmarks is not None:
                    smoothed_lmk = self.smoother.smooth_landmarks(target.landmarks)
                    target.landmarks = smoothed_lmk

                    out_frame = self.swapper.swap(
                        target_frame=frame.image,
                        target_face=target,
                        source_face=source_face,
                        smoother=self.smoother,
                        blend_intensity=self.blend_intensity,
                        color_correct=self.color_correct,
                    )
                else:
                    self.smoother.reset()
                    out_frame = frame.image

                # Cálculo de FPS instantâneo suavizado
                t_now = time.perf_counter()
                dt = t_now - t0
                instant_fps = 1.0 / dt if dt > 0 else 0.0
                fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

                if self.on_frame_rendered:
                    self.on_frame_rendered(out_frame, fps)

                frame_count += 1

