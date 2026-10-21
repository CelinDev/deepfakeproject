"""Módulo de Troca Facial (Face Swapper).

Implementação do modelo pré-treinado INSwapper 128 (InsightFace).
Recebe:
- Frame de destino (imagem BGR)
- Rosto de destino detectado (com marcos faciais / landmarks)
- Rosto de origem de referência (com embedding ArcFace normalizado de 512 dimensões)

Executa:
1. Alinhamento e recorte canônico 128x128 do rosto de destino.
2. Projeção latente do embedding facial através da matriz emap do modelo.
3. Inferência ONNX acelerada por CUDA (ou CPU fallback).
4. Composição com o quadro original preservando iluminação e bordas suaves.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from config.settings import MODELS_DIR
from src.core.interfaces import FaceSwapper
from src.core.types import DetectedFace
from src.face.face_aligner import FaceAligner
from src.face.face_compositor import FaceCompositor
from src.face.face_enhancer import FaceEnhancer
from src.face.temporal_smoother import TemporalSmoother
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)


class INSwapperModel(FaceSwapper):
    """Implementa FaceSwapper utilizando o modelo ONNX inswapper_128."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
        compositor: FaceCompositor | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else (MODELS_DIR / "inswapper_128.onnx")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modelo de face swap não encontrado em: {self.model_path}. "
                "Certifique-se de que o download foi concluído."
            )

        self.compositor = compositor or FaceCompositor()

        # Extrai a matriz emap dos initializers do grafo ONNX
        model_proto = onnx.load(str(self.model_path))
        self.emap = numpy_helper.to_array(model_proto.graph.initializer[-1])
        del model_proto

        # Cria a sessão ONNX Runtime com os providers do DeviceManager
        providers = onnx_providers(device_manager)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=providers
        )
        self.active_providers = self.session.get_providers()
        logger.info("INSwapper carregado. Providers ativos: %s", self.active_providers)

        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (128, 128)

    def swap(
        self,
        target_frame: np.ndarray,
        target_face: DetectedFace,
        source_face: DetectedFace,
        smoother: TemporalSmoother | None = None,
        enhancer: FaceEnhancer | None = None,
        blend_intensity: float = 1.0,
        color_correct: bool = True,
        source_crop: np.ndarray | None = None,
    ) -> np.ndarray:
        if target_face.landmarks is None:
            return target_frame  # Sem marcos faciais, impossível alinhar
        if source_face.embedding is None:
            raise ValueError("O rosto de origem (source_face) precisa conter um embedding facial válido.")

        # 1. Alinha o rosto de destino
        aligned_crop, affine_m = FaceAligner.align_crop(
            target_frame, target_face.landmarks, crop_size=self.input_size[0]
        )

        # 2. Prepara o blob de imagem (BGR -> RGB, escala 1/255, NCHW)
        blob = cv2.dnn.blobFromImage(
            aligned_crop,
            1.0 / 255.0,
            self.input_size,
            (0.0, 0.0, 0.0),
            swapRB=True,
        )

        # 3. Prepara o vetor latente do rosto de origem
        latent = source_face.embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        norm = np.linalg.norm(latent)
        if norm > 1e-6:
            latent /= norm

        # 4. Inferência ONNX
        preds = self.session.run(
            [self.output_name],
            {self.input_names[0]: blob, self.input_names[1]: latent},
        )[0]

        # 5. Converte a saída NCHW (RGB) para HWC (BGR [0..255])
        img_fake = preds.transpose((0, 2, 3, 1))[0]
        fake_bgr = np.clip(img_fake * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]

        # 6. Se fornecido source_crop (ex: tom esverdeado/mortiço de zumbi da fonte),
        # transfere a paleta de cores característica da fonte antes da composição
        if source_crop is not None:
            fake_bgr = self.compositor.match_color_hist(fake_bgr, source_crop, smoother=smoother)
            # Evita que o compositor sobreponha a iluminação do alvo em cima do tom da fonte
            color_correct = False

        # 7. Restauração facial / Super-resolução opcional (GFPGAN)
        if enhancer is not None:
            fake_bgr = enhancer.enhance(fake_bgr)

        # 8. Composição sobre o frame original
        return self.compositor.composite(
            original_frame=target_frame,
            fake_crop=fake_bgr,
            aligned_original_crop=aligned_crop,
            affine_matrix=affine_m,
            color_correct=color_correct,
            smoother=smoother,
            blend_intensity=blend_intensity,
        )

