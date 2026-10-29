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

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from config.settings import (
    ALIGNED_MASK_SIZE,
    HYPERSWAP_256_MODEL_PATH,
    MODELS_DIR,
    SMOOTH_TRANSFORM_PARAMS,
    SWAPPER_MODEL_NAME,
    SWAPPER_WORK_SIZE,
)
from src.core.interfaces import FaceOccluder, FaceSwapper
from src.core.types import DetectedFace
from src.face.face_aligner import FaceAligner
from src.face.face_compositor import FaceCompositor
from src.face.face_enhancer import FaceEnhancer
from src.face.face_parser import FaceParser
from src.face.temporal_smoother import TemporalSmoother
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers, run_with_iobinding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwapperModelSpec:
    """Especificação de pré/pós-processamento e alinhamento para um modelo ONNX de face swap."""

    name: str
    filename: str
    input_size: tuple[int, int]
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    align_mode: str = "arcface_128"  # "arcface_128" ou "arcface_112"
    uses_emap: bool = True


SWAPPER_MODEL_SPECS: dict[str, SwapperModelSpec] = {
    "inswapper_128": SwapperModelSpec(
        name="inswapper_128",
        filename="inswapper_128.onnx",
        input_size=(128, 128),
        norm_mean=(0.0, 0.0, 0.0),
        norm_std=(1.0, 1.0, 1.0),
        align_mode="arcface_128",
        uses_emap=True,
    ),
    "hyperswap_256": SwapperModelSpec(
        name="hyperswap_256",
        filename=HYPERSWAP_256_MODEL_PATH.name,
        input_size=(256, 256),
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
        align_mode="arcface_128",
        uses_emap=False,
    ),
}


class INSwapperModel(FaceSwapper):
    """Implementa FaceSwapper configurável para inswapper_128 (128x128) e hyperswap_256 (256x256)."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
        compositor: FaceCompositor | None = None,
        model_name: str = SWAPPER_MODEL_NAME,
    ) -> None:
        self.spec, self.model_path = self._resolve_spec_and_path(model_name, model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Modelo de face swap não encontrado em: {self.model_path}. "
                "Certifique-se de que o download foi concluído."
            )

        self.compositor = compositor or FaceCompositor()

        # Extrai a matriz emap dos initializers do grafo ONNX (se presente, como no inswapper_128)
        self.emap: np.ndarray | None = None
        if self.spec.uses_emap:
            model_proto = onnx.load(str(self.model_path))
            if len(model_proto.graph.initializer) > 0:
                last_init = numpy_helper.to_array(model_proto.graph.initializer[-1])
                if last_init.shape == (512, 512):
                    self.emap = last_init
            del model_proto

        # Cria a sessão ONNX Runtime com os providers do DeviceManager
        providers = onnx_providers(device_manager)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=providers
        )
        self.active_providers = self.session.get_providers()
        logger.info("Swapper '%s' carregado (%s). Providers ativos: %s", self.spec.name, self.model_path.name, self.active_providers)

        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        self.target_input_name, self.source_input_name = self._identify_inputs(inputs)
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = self.spec.input_size
        self.work_size: int = SWAPPER_WORK_SIZE  # Super-amostragem de alinhamento e composição (512x512)
        self._latent_cache_key: bytes | None = None
        self._latent_cache_val: np.ndarray | None = None

    @staticmethod
    def _resolve_spec_and_path(
        model_name: str, model_path: str | Path | None
    ) -> tuple[SwapperModelSpec, Path]:
        requested_key = model_name.strip().lower()
        spec = SWAPPER_MODEL_SPECS.get(requested_key, SWAPPER_MODEL_SPECS["inswapper_128"])

        if model_path is not None:
            p = Path(model_path)
            if "256" in p.name or "hyperswap" in p.name.lower():
                spec = SWAPPER_MODEL_SPECS["hyperswap_256"]
            return spec, p

        candidate = MODELS_DIR / spec.filename
        if candidate.is_file():
            return spec, candidate

        # Fallback automático para inswapper_128 se o modelo 256px não estiver baixado
        fallback_spec = SWAPPER_MODEL_SPECS["inswapper_128"]
        fallback_path = MODELS_DIR / fallback_spec.filename
        if requested_key != "inswapper_128":
            logger.warning(
                "Modelo '%s' (%s) não encontrado em models/; usando '%s' como padrão.",
                requested_key,
                candidate.name,
                fallback_spec.name,
            )
        return fallback_spec, fallback_path

    @staticmethod
    def _identify_inputs(inputs: list[Any]) -> tuple[str, str]:
        """Identifica qual tensor de entrada é o crop de imagem (4D) e qual é o embedding (2D)."""
        if len(inputs) < 2:
            name = inputs[0].name if inputs else "target"
            return name, name
        img_inp = inputs[0].name
        emb_inp = inputs[1].name
        for inp in inputs:
            shape = getattr(inp, "shape", None) or []
            if len(shape) == 4 or inp.name in ("target", "input"):
                img_inp = inp.name
            elif len(shape) == 2 or inp.name in ("source", "latent"):
                emb_inp = inp.name
        return img_inp, emb_inp

    def preprocess_crop(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Redimensiona para ``self.input_size`` e aplica a normalização RGB da especificação."""
        resized = (
            aligned_crop
            if aligned_crop.shape[:2] == self.input_size
            else cv2.resize(aligned_crop, self.input_size, interpolation=cv2.INTER_AREA)
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(self.spec.norm_mean, dtype=np.float32)
        std = np.array(self.spec.norm_std, dtype=np.float32)
        norm = (rgb - mean) / std
        return norm.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def postprocess_output(self, raw_pred: np.ndarray) -> np.ndarray:
        """Converte a saída NCHW normalizada da rede de volta para imagem BGR uint8 [0..255]."""
        img_rgb = raw_pred.transpose((0, 2, 3, 1))[0].astype(np.float32)
        mean = np.array(self.spec.norm_mean, dtype=np.float32)
        std = np.array(self.spec.norm_std, dtype=np.float32)
        restored_01 = img_rgb * std + mean
        bgr = np.clip(restored_01 * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        return bgr

    def prepare_latent(self, embedding: np.ndarray) -> np.ndarray:
        """Prepara e cacheia o vetor latente de identidade (``latent @ emap`` normalizado por rosto de origem)."""
        emb_f = np.ascontiguousarray(embedding, dtype=np.float32).reshape((1, -1))
        key = emb_f.tobytes()
        if self._latent_cache_key == key and self._latent_cache_val is not None:
            return self._latent_cache_val

        latent = emb_f
        if self.emap is not None:
            latent = np.dot(latent, self.emap)
        norm = float(np.linalg.norm(latent))
        if norm > 1e-6:
            latent = latent / norm
        out = latent.astype(np.float32)
        self._latent_cache_key = key
        self._latent_cache_val = out
        return out

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
        enhancer_interval: int = 1,
        parser: FaceParser | None = None,
        parser_interval: int = 1,
        occluder: FaceOccluder | None = None,
    ) -> np.ndarray:
        if target_face.landmarks is None:
            return target_frame  # Sem marcos faciais, impossível alinhar
        if source_face.embedding is None:
            raise ValueError("O rosto de origem (source_face) precisa conter um embedding facial válido.")

        # 1. Alinha o rosto de destino em alta definição (512x512) para preservar poros, dentes e nitidez.
        # Se landmarks_106 estiver disponível, usa os centroides do contorno de 106 pontos e suaviza os
        # parâmetros da transformação de similaridade (escala, rotação, translação).
        raw_affine_hi = FaceAligner.estimate_affine_matrix(
            target_face.landmarks, crop_size=self.work_size, landmarks_106=target_face.landmarks_106
        )
        if smoother is not None and SMOOTH_TRANSFORM_PARAMS:
            affine_hi = smoother.smooth_similarity_matrix(raw_affine_hi)
        else:
            affine_hi = raw_affine_hi

        aligned_hi, affine_hi = FaceAligner.align_crop(
            target_frame,
            target_face.landmarks,
            crop_size=self.work_size,
            landmarks_106=target_face.landmarks_106,
            affine_matrix=affine_hi,
        )

        # 2. Prepara o blob de imagem e o vetor latente conforme a especificação do modelo (128px ou 256px)
        blob = self.preprocess_crop(aligned_hi)
        latent = self.prepare_latent(source_face.embedding)

        # 3. Inferência ONNX acelerada (com IOBinding quando suportado)
        out_shape = (1, 3, self.input_size[1], self.input_size[0])
        preds = run_with_iobinding(
            self.session,
            self.output_name,
            {self.target_input_name: blob, self.source_input_name: latent},
            output_shape=out_shape,
        )

        # 4. Converte a saída NCHW para BGR [0..255] e faz upsampling Lanczos4 para 512x512
        fake_native = self.postprocess_output(preds)
        fake_bgr = (
            fake_native
            if fake_native.shape[:2] == (self.work_size, self.work_size)
            else cv2.resize(fake_native, (self.work_size, self.work_size), interpolation=cv2.INTER_LANCZOS4)
        )

        # 6. Se fornecido source_crop (ex: tom esverdeado/mortiço de zumbi da fonte),
        # transfere a paleta de cores característica da fonte antes da composição
        if source_crop is not None:
            src_hi = (
                cv2.resize(source_crop, (self.work_size, self.work_size), interpolation=cv2.INTER_LANCZOS4)
                if source_crop.shape[0] != self.work_size
                else source_crop
            )
            fake_bgr = self.compositor.match_color_hist(fake_bgr, src_hi, smoother=smoother)
            # Evita que o compositor sobreponha a iluminação do alvo em cima do tom da fonte
            color_correct = False

        # 7. Restauração facial / Super-resolução opcional no recorte do rosto (GFPGAN 512x512)
        if enhancer is not None:
            fake_bgr = enhancer.enhance(fake_bgr, reuse_interval=enhancer_interval)

        # 8. Máscara semântica contínua (softmax) no espaço do crop alinhado (inclui oclusão de cabelo/orelhas/pescoço/óculos)
        parser_mask: np.ndarray | None = None
        skin_mask: np.ndarray | None = None
        if parser is not None:
            parser_mask = parser.parse_probabilities(
                aligned_hi, out_size=ALIGNED_MASK_SIZE, reuse_interval=parser_interval
            )
            skin_mask = parser.last_skin_mask

        # 8b. Máscara opcional de oclusão genérica (XSeg: mãos, copos, microfone, objetos arbitrários)
        occluder_mask: np.ndarray | None = None
        if occluder is not None and occluder.available:
            occluder_mask = occluder.predict_mask(aligned_hi, out_size=ALIGNED_MASK_SIZE)

        # 9. Composição em alta resolução (512x512) sobre o frame original
        return self.compositor.composite(
            original_frame=target_frame,
            fake_crop=fake_bgr,
            aligned_original_crop=aligned_hi,
            affine_matrix=affine_hi,
            color_correct=color_correct,
            smoother=smoother,
            blend_intensity=blend_intensity,
            parser_mask=parser_mask,
            occluder_mask=occluder_mask,
            skin_mask=skin_mask,
        )

