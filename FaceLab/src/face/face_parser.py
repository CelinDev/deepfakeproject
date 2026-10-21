"""Módulo de Segmentação Facial e de Cabeça (Face & Head Parser).

Utiliza a rede neural BiSeNet ResNet-34 para segmentar semanticamente as 19 classes
anatômicas da cabeça humana (pele, cabelo, orelhas, sobrancelhas, olhos, lábios, pescoço, chapéu, etc.).
Permite isolar com precisão o cabelo e as orelhas para composição completa de cabeça.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

from config.settings import BISENET_MODEL_PATH
from src.core.types import DetectedFace
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)

# Mapeamento oficial do modelo BiSeNet (CelebAMask-HQ 19 classes)
FACE_PARSER_CLASSES = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# Conjunto padrão de classes para a "Cabeça Completa" (Rosto + Cabelo + Orelhas + Acessórios)
HEAD_CLASSES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18})

# Conjunto exclusivo de Cabelo, Orelhas, Brincos e Adornos de Cabeça
HAIR_AND_EARS_CLASSES = frozenset({7, 8, 9, 17, 18})


class FaceParser:
    """Segmentador semântico de anatomia facial e capilar baseado em BiSeNet."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else BISENET_MODEL_PATH
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Modelo BiSeNet não encontrado em: {self.model_path}")

        providers = onnx_providers(device_manager)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name
        logger.info("FaceParser (BiSeNet) carregado com sucesso. Providers: %s", self._session.get_providers())

    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Executa a segmentação semântica e retorna uma matriz 2D com IDs de classe (0 a 18).

        Args:
            image: Imagem de entrada BGR (H, W, 3).

        Returns:
            Matriz int32 (H, W) com a classe predita de cada pixel.
        """
        h, w = image.shape[:2]
        # BiSeNet foi treinado com entrada 512x512 normalizada por ImageNet
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255.0,
            size=(512, 512),
            mean=(0.485, 0.456, 0.406),
            swapRB=True,
            crop=False,
        )
        blob = blob / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        blob = blob.astype(np.float32)

        preds = self._session.run([self.output_name], {self.input_name: blob})[0]
        mask_512 = np.argmax(preds[0], axis=0).astype(np.uint8)

        # Reamostra a máscara para as dimensões originais da imagem com vizinho mais próximo
        return cv2.resize(mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

    def extract_mask(
        self,
        image: np.ndarray,
        classes: Sequence[int] | frozenset[int] = HEAD_CLASSES,
        feather: int = 7,
    ) -> np.ndarray:
        """
        Gera uma máscara contínua float32 [0.0, 1.0] contendo as classes solicitadas.

        Args:
            image: Imagem BGR (H, W, 3).
            classes: Índices das classes a incluir na máscara.
            feather: Raio de desfoque gaussiano para transição suave de borda.

        Returns:
            Máscara 2D float32 (H, W) entre 0.0 e 1.0.
        """
        parsed = self.parse(image)
        mask = np.isin(parsed, list(classes)).astype(np.float32)

        if feather > 0:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def blend_hair_and_ears(
        self,
        target_frame: np.ndarray,
        source_image: np.ndarray,
        target_face: DetectedFace,
        source_face: DetectedFace,
        source_hair_mask: np.ndarray | None = None,
        blend_intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Transfere e funde o cabelo, orelhas e adereços da imagem de origem para a pose do alvo.

        Args:
            target_frame: Frame de destino após o face swap facial.
            source_image: Imagem de origem original completa.
            target_face: Dados e marcos faciais detectados no alvo.
            source_face: Dados e marcos faciais detectados na origem.
            source_hair_mask: Máscara pré-computada de cabelo/orelhas da origem (opcional para performance).
            blend_intensity: Intensidade de mistura do cabelo (0.0 a 1.0).

        Returns:
            Frame final com cabeça e cabelos integrados.
        """
        if blend_intensity <= 0.01 or target_face.landmarks is None or source_face.landmarks is None:
            return target_frame

        h_tgt, w_tgt = target_frame.shape[:2]

        # Estima a transformação afim de similaridade entre os marcos da origem e do alvo
        tform = SimilarityTransform.from_estimate(source_face.landmarks, target_face.landmarks)
        matrix = tform.params[0:2, :]

        # Projeta a imagem de origem e a máscara para a geometria e escala da cabeça de destino
        warped_source = cv2.warpAffine(
            source_image, matrix, (w_tgt, h_tgt), borderMode=cv2.BORDER_REFLECT_101
        )

        if source_hair_mask is None:
            source_hair_mask = self.extract_mask(source_image, classes=HAIR_AND_EARS_CLASSES, feather=7)

        warped_mask = cv2.warpAffine(
            source_hair_mask, matrix, (w_tgt, h_tgt), flags=cv2.INTER_LINEAR, borderValue=0.0
        )

        alpha = np.expand_dims(np.clip(warped_mask * blend_intensity, 0.0, 1.0), axis=-1)
        blended = np.clip(
            target_frame.astype(np.float32) * (1.0 - alpha) + warped_source.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

        return blended

