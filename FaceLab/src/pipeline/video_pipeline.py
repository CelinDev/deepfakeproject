"""Pipeline de Processamento de Vídeo (Video Pipeline).

Coordena todo o fluxo de ponta a ponta:
1. Validação de entradas (vídeo e imagem de referência).
2. Extração do rosto de origem (source_face).
3. Abertura do FrameSource e FrameSink (com cópia de áudio via FFmpeg).
4. Para cada quadro:
   a. Detecção de rostos (InsightFaceDetector).
   b. Rastreamento e identificação do alvo (TargetFaceTracker).
   c. Execução do Face Swap (INSwapperModel).
   d. Gravação no sink.
5. Emissão de callbacks de progresso (compatível com CLI e UI futura).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config.settings import OUTPUT_DIR, ZOMBIE_REFERENCE_IMAGE
from src.core.types import DetectedFace
from src.face.face_compositor import FaceCompositor
from src.face.face_detector import InsightFaceDetector
from src.face.face_enhancer import FaceEnhancer
from src.face.face_parser import HAIR_AND_EARS_CLASSES, FaceParser
from src.face.face_swapper import INSwapperModel
from src.face.face_tracker import TargetFaceTracker
from src.face.temporal_smoother import TemporalSmoother
from src.face.zombie_transformer import ZombieTransformer
from src.inference.device_manager import DeviceManager, DevicePreference
from src.utils.validators import validate_image_path, validate_output_path, validate_video_path
from src.video.frame_source import VideoFileSource
from src.video.video_writer import VideoFileSink

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    current_frame: int
    total_frames: int | None
    fps: float
    elapsed_seconds: float
    faces_detected: int
    target_found: bool


class VideoPipeline:
    """Executa a substituição facial em arquivos de vídeo."""

    def __init__(
        self,
        device_preference: DevicePreference = DevicePreference.AUTO,
        on_progress: Callable[[PipelineProgress], None] | None = None,
        enable_enhancer: bool = False,
        blend_intensity: float = 1.0,
        color_correct: bool = True,
        feather_amount: int = 15,
        enable_zombie: bool = False,
        enable_head_swap: bool = True,
    ) -> None:
        self.device_manager = DeviceManager(device_preference)
        self.detector = InsightFaceDetector(self.device_manager)
        compositor = FaceCompositor(blur_amount=feather_amount)
        self.swapper = INSwapperModel(self.device_manager, compositor=compositor)
        self.enhancer = FaceEnhancer(self.device_manager) if enable_enhancer else None
        self.parser = FaceParser(self.device_manager) if enable_head_swap else None
        self.blend_intensity = blend_intensity
        self.color_correct = color_correct
        self.enable_zombie = enable_zombie
        self.enable_head_swap = enable_head_swap
        self.on_progress = on_progress
        self._is_cancelled = False

    def cancel(self) -> None:
        """Sinaliza para interromper o pipeline com segurança."""
        self._is_cancelled = True

    def extract_source_face(self, image_path: str | Path, det_thresh: float = 0.5) -> DetectedFace:
        """Detecta e retorna o rosto principal da imagem de referência."""
        path = validate_image_path(image_path)
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

        # Restaura o limiar padrão para o processamento de quadros do vídeo
        self.detector.set_detection_threshold(0.5)

        if not faces:
            raise ValueError(f"Nenhum rosto foi detectado na imagem de referência: {path.name}")

        # Seleciona o maior rosto presente na foto
        source_face = TargetFaceTracker.select_largest(faces)
        assert source_face is not None
        if source_face.embedding is None:
            raise ValueError("Não foi possível extrair o embedding facial da imagem de referência.")
        return source_face

    def process(
        self,
        video_path: str | Path,
        source_image_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Executa o processamento completo do vídeo."""
        self._is_cancelled = False
        v_path = validate_video_path(video_path)
        out_path = validate_output_path(
            output_path or (OUTPUT_DIR / f"{v_path.stem}_swapped.mp4"),
            source=v_path,
        )

        source_face: DetectedFace | None = None
        source_image: np.ndarray | None = None
        source_hair_mask: np.ndarray | None = None

        ref_path: Path | None = None
        if self.enable_zombie and source_image_path is None:
            if ZOMBIE_REFERENCE_IMAGE.is_file():
                ref_path = ZOMBIE_REFERENCE_IMAGE
                logger.info("Carregando identidade zumbi hiper-realista de referência: %s", ref_path)
                source_face = self.extract_source_face(ref_path, det_thresh=0.2)
        elif source_image_path is not None:
            ref_path = validate_image_path(source_image_path)
            logger.info("Extraindo rosto de referência...")
            source_face = self.extract_source_face(ref_path)

        if ref_path is not None and self.parser is not None and source_face is not None:
            source_image = cv2.imread(str(ref_path))
            if source_image is not None:
                source_hair_mask = self.parser.extract_mask(
                    source_image, classes=HAIR_AND_EARS_CLASSES, feather=7
                )

        source = VideoFileSource(v_path)
        info = source.info
        sink = VideoFileSink(
            out_path,
            width=info.width,
            height=info.height,
            fps=info.fps,
            audio_source=v_path if info.has_audio else None,
        )

        tracker: TargetFaceTracker | None = None
        smoother = TemporalSmoother(alpha_landmarks=0.65, alpha_color=0.8)
        start_time = time.perf_counter()
        processed_count = 0

        logger.info("Iniciando pipeline de processamento de quadros...")
        with source, sink:
            for frame in source.frames():
                if self._is_cancelled:
                    logger.warning("Processamento cancelado pelo usuário.")
                    sink.abort()
                    raise KeyboardInterrupt("Processamento cancelado.")

                t0 = time.perf_counter()
                need_embedding = (tracker is None) or (processed_count % 5 == 0) or (tracker._missing > 0)
                faces = self.detector.detect(frame.image, compute_embeddings=need_embedding)

                # Inicializa o tracker com o rosto principal do vídeo no primeiro quadro com rostos
                if tracker is None and faces:
                    primeiro_alvo = TargetFaceTracker.select_largest(faces)
                    if primeiro_alvo and primeiro_alvo.embedding is not None:
                        tracker = TargetFaceTracker(primeiro_alvo.embedding)

                # Rastreia o alvo consistente ao longo do vídeo
                target = tracker.update(faces) if tracker else None

                # Se o alvo foi localizado no quadro atual, processa
                if target is not None and target.landmarks is not None:
                    # REFINAMENTO TEMPORAL: suaviza os 5 pontos faciais para eliminar micro-tremores (jitter)
                    smoothed_lmk = smoother.smooth_landmarks(target.landmarks)
                    target.landmarks = smoothed_lmk

                    if source_face is not None:
                        out_image = self.swapper.swap(
                            target_frame=frame.image,
                            target_face=target,
                            source_face=source_face,
                            smoother=smoother,
                            enhancer=self.enhancer,
                            blend_intensity=self.blend_intensity,
                            color_correct=self.color_correct,
                        )

                        # Segmentação e Fusão da Cabeça Completa (Cabelo, Orelhas, Brincos)
                        if self.parser is not None and source_image is not None:
                            out_image = self.parser.blend_hair_and_ears(
                                target_frame=out_image,
                                source_image=source_image,
                                target_face=target,
                                source_face=source_face,
                                source_hair_mask=source_hair_mask,
                                blend_intensity=self.blend_intensity,
                            )
                    else:
                        out_image = frame.image.copy()

                    # Refinamento de Halloween: se ativado sem swap zumbi (ou pós-swap), aplica grading fotográfico
                    if self.enable_zombie and source_image_path is not None:
                        out_image = ZombieTransformer.transform(
                            out_image,
                            landmarks=target.landmarks,
                            intensity=1.0,
                        )
                else:
                    smoother.reset()
                    out_image = frame.image

                frame.image = out_image
                sink.write(frame)
                processed_count += 1

                # Métricas e callback
                elapsed = time.perf_counter() - start_time
                fps_calc = processed_count / elapsed if elapsed > 0 else 0.0

                if self.on_progress:
                    self.on_progress(
                        PipelineProgress(
                            current_frame=processed_count,
                            total_frames=info.frame_count,
                            fps=fps_calc,
                            elapsed_seconds=elapsed,
                            faces_detected=len(faces),
                            target_found=target is not None,
                        )
                    )

        total_time = time.perf_counter() - start_time
        logger.info(
            "Concluído! %d quadros em %.1fs (%.1f fps). Saída: %s",
            processed_count,
            total_time,
            processed_count / total_time if total_time > 0 else 0,
            out_path,
        )
        return out_path

