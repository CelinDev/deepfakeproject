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

from config.settings import ZOMBIE_REFERENCE_IMAGE
from src.core.interfaces import FrameSource
from src.core.types import DetectedFace, Frame
from src.face.face_aligner import FaceAligner
from src.face.face_compositor import FaceCompositor
from src.face.face_detector import InsightFaceDetector
from src.face.face_enhancer import FaceEnhancer
from src.face.face_parser import NATURAL_HAIR_AND_OCCLUSION_CLASSES, FaceParser
from src.face.face_swapper import INSwapperModel
from src.face.face_tracker import TargetFaceTracker
from src.face.temporal_smoother import TemporalSmoother
from src.face.zombie_transformer import ZombieTransformer
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
        enable_zombie: bool = False,
        enable_head_swap: bool = True,
        enable_enhancer: bool = True,
    ) -> None:
        """
        Args:
            device_preference: Preferência de hardware (GPU/CPU).
            on_frame_rendered: Callback que recebe (frame_bgr, fps_atual) para atualização em tempo real.
            blend_intensity: Força de mistura do rosto [0.0 a 1.0].
            color_correct: Ajuste automático de iluminação e cor.
            feather_amount: Suavização da borda da máscara.
            enable_zombie: Ativa transformação e ferimentos de Halloween/Zumbi.
            enable_head_swap: Ativa transferência de cabeça completa (cabelo e orelhas).
            enable_enhancer: Ativa restauração de super-resolução e texturas faciais.
        """
        self.device_manager = DeviceManager(device_preference)
        # Modo rápido: detecção com landmarks e sem cálculo obrigatório de embedding em todo quadro
        self.detector = InsightFaceDetector(self.device_manager)
        compositor = FaceCompositor(blur_amount=feather_amount)
        self.swapper = INSwapperModel(self.device_manager, compositor=compositor)
        self.parser = FaceParser(self.device_manager) if enable_head_swap else None
        self.enhancer = FaceEnhancer(self.device_manager) if enable_enhancer else None
        self.smoother = TemporalSmoother(alpha_landmarks=0.65, alpha_color=0.8)
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.enable_zombie = enable_zombie
        self.enable_head_swap = enable_head_swap
        self.on_frame_rendered = on_frame_rendered
        self._is_running = False

    def stop(self) -> None:
        """Interrompe a execução contínua da webcam."""
        self._is_running = False

    def update_params(
        self,
        blend_intensity: float | None = None,
        color_correct: bool | None = None,
        feather_amount: int | None = None,
        enable_zombie: bool | None = None,
    ) -> None:
        """Atualiza parâmetros de renderização dinamicamente em tempo real sem reiniciar o stream."""
        if blend_intensity is not None:
            self.blend_intensity = float(blend_intensity)
        if color_correct is not None:
            self.color_correct = bool(color_correct)
        if feather_amount is not None:
            k = int(feather_amount)
            if hasattr(self.swapper, "compositor"):
                self.swapper.compositor.blur_amount = k if k % 2 != 0 else k + 1
        if enable_zombie is not None:
            self.enable_zombie = bool(enable_zombie)

    def load_source_face(self, source_image_path: str | Path, det_thresh: float = 0.5) -> DetectedFace:
        """Extrai o embedding e marcos do rosto novo de referência."""
        path = validate_image_path(source_image_path)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {path}")

        if det_thresh != 0.5:
            self.detector.set_detection_threshold(det_thresh)

        faces = self.detector.detect(img)

        # Se não achou na primeira tentativa, tenta com limiar baixo automático (ex: próteses pesadas / zumbi)
        if not faces and det_thresh >= 0.5:
            self.detector.set_detection_threshold(0.2)
            faces = self.detector.detect(img)

        # Restaura o limiar padrão para o processamento de quadros da webcam
        self.detector.set_detection_threshold(0.5)

        if not faces:
            raise ValueError(f"Nenhum rosto foi detectado na imagem de referência: {path.name}")

        source_face = TargetFaceTracker.select_largest(faces)
        assert source_face is not None
        if source_face.embedding is None:
            raise ValueError("Não foi possível extrair o embedding facial da imagem de referência.")
        return source_face

    def run(
        self,
        source_image_path: str | Path | None = None,
        camera_source: FrameSource | None = None,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        """Inicia o laço de captura, troca facial e renderização em tempo real."""
        source_face: DetectedFace | None = None
        source_crop: np.ndarray | None = None
        ref_path: Path | None = None
        if self.enable_zombie and source_image_path is None:
            if ZOMBIE_REFERENCE_IMAGE.is_file():
                ref_path = ZOMBIE_REFERENCE_IMAGE
                logger.info("Carregando identidade zumbi hiper-realista para a webcam: %s", ref_path)
                source_face = self.load_source_face(ref_path, det_thresh=0.2)
        elif source_image_path is not None:
            ref_path = validate_image_path(source_image_path)
            source_face = self.load_source_face(ref_path)

        # Transfere a paleta de cores característica da fonte APENAS se for o preset de Zumbi
        is_zombie_ref = (ref_path is not None) and ("zombie" in str(ref_path).lower())
        if is_zombie_ref and ref_path is not None and source_face is not None and source_face.landmarks is not None:
            ref_img = cv2.imread(str(ref_path))
            if ref_img is not None:
                source_crop, _ = FaceAligner.align_crop(
                    ref_img, source_face.landmarks, crop_size=self.swapper.input_size[0]
                )
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

                    if source_face is not None:
                        out_frame = self.swapper.swap(
                            target_frame=frame.image,
                            target_face=target,
                            source_face=source_face,
                            smoother=self.smoother,
                            blend_intensity=self.blend_intensity,
                            color_correct=self.color_correct,
                            source_crop=source_crop,
                        )

                        # Super-resolução e Nitidez de Alta Frequência em Tempo Real (<1ms)
                        if self.enhancer is not None:
                            out_frame = self.enhancer.enhance_fast(out_frame, amount=0.35)

                        # Preservação e Oclusão Natural dos Cabelos/Orelhas Reais (elimina aspecto de 'peruca/robô')
                        if self.parser is not None:
                            out_frame = self.parser.restore_natural_hair_and_ears(
                                swapped_frame=out_frame,
                                original_frame=frame.image,
                                target_face=target,
                                feather=5,
                            )
                    else:
                        out_frame = frame.image.copy()

                    # Refinamento de Halloween: se ativado junto com source customizado
                    if self.enable_zombie and source_image_path is not None:
                        out_frame = ZombieTransformer.transform(
                            out_frame,
                            landmarks=target.landmarks,
                            intensity=1.0,
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

