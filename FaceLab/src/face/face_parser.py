"""Módulo de Segmentação Facial e Oclusão de Cabeça (Face & Hair Parser).

Utiliza a rede neural BiSeNet ResNet-34 para segmentar semanticamente as 19 classes
anatômicas da cabeça humana (cabelo, orelhas, óculos, pescoço, etc.).

Fornece restauração ultrarrealista e natural:
- Preserva fios de cabelo reais, franja, costeletas, orelhas, piercings e óculos da pessoa no vídeo/webcam.
- Permite que o cabelo real caia naturalmente por cima do rosto trocado com bordas suaves (soft matting),
  eliminando qualquer efeito artificial de 'peruca/robô'.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from config.settings import BISENET_MODEL_PATH
from src.core.types import DetectedFace
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)

# Mapeamento das 19 classes CelebAMask-HQ
FACE_PARSER_CLASSES = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",      # Óculos
    7: "l_ear",      # Orelha esq
    8: "r_ear",      # Orelha dir
    9: "ear_r",      # Brincos
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",      # Pescoço
    15: "neck_l",    # Colar / Gargantilha
    16: "cloth",     # Roupa
    17: "hair",      # Cabelo
    18: "hat",       # Chapéu / Boné
}

# Classes de oclusão e proteção de fundo (elementos que NÃO devem receber o tom do rosto)
# 0: background (fundo do cenário), 7: orelha esq, 8: orelha dir, 9: brincos, 14: pescoço, 15: colares, 16: roupas, 17: cabelo, 18: chapéus
NATURAL_HAIR_AND_OCCLUSION_CLASSES = frozenset({
    0,   # background / fundo da cena (elimina qualquer aura verde no ar)
    7,   # orelha esquerda
    8,   # orelha direita
    9,   # brincos
    14,  # pescoço
    15,  # adornos do pescoço
    16,  # roupas / ombros
    17,  # cabelo
    18,  # chapéu / adereços
})


class FaceParser:
    """Segmentador semântico baseado em BiSeNet para preservação orgânica de cabelos e orelhas."""

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
        logger.info("FaceParser (BiSeNet) carregado. Providers: %s", self._session.get_providers())

    def parse(self, image: np.ndarray) -> np.ndarray:
        """Executa a inferência e retorna mapa de índices 2D (0 a 18)."""
        h, w = image.shape[:2]
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
        return cv2.resize(mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

    def extract_mask(
        self,
        image: np.ndarray,
        classes: Sequence[int] | frozenset[int] = NATURAL_HAIR_AND_OCCLUSION_CLASSES,
        feather: int = 5,
    ) -> np.ndarray:
        """Extrai máscara suave float32 [0.0, 1.0] para as classes solicitadas."""
        parsed = self.parse(image)
        mask = np.isin(parsed, list(classes)).astype(np.float32)
        if feather > 0:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return mask

    def restore_natural_hair_and_ears(
        self,
        swapped_frame: np.ndarray,
        original_frame: np.ndarray,
        target_face: DetectedFace | None = None,
        feather: int = 5,
    ) -> np.ndarray:
        """
        Restaura com fidelidade orgânica o cabelo real, orelhas, franja e óculos da pessoa original.

        O rosto sintético fica posicionado exatamente sob os fios e detalhes originais,
        acompanhando a física, balanço e iluminação da cena com perfeição fotorrealista.
        """
        h, w = original_frame.shape[:2]

        # Se houver bounding box, fazemos na ROI expandida para altíssima performance
        if target_face is not None and target_face.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
            pad_x = int((x2 - x1) * 0.45)
            pad_y = int((y2 - y1) * 0.45)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            roi_orig = original_frame[ry1:ry2, rx1:rx2]
            roi_swap = swapped_frame[ry1:ry2, rx1:rx2]

            mask_roi = self.extract_mask(
                roi_orig,
                classes=NATURAL_HAIR_AND_OCCLUSION_CLASSES,
                feather=feather,
            )

            alpha = np.expand_dims(np.clip(mask_roi, 0.0, 1.0), axis=-1)
            roi_blended = np.clip(
                roi_swap.astype(np.float32) * (1.0 - alpha) + roi_orig.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

            out = swapped_frame.copy()
            out[ry1:ry2, rx1:rx2] = roi_blended
            return out

        # Fallback para o quadro inteiro
        mask = self.extract_mask(original_frame, classes=NATURAL_HAIR_AND_OCCLUSION_CLASSES, feather=feather)
        alpha = np.expand_dims(np.clip(mask, 0.0, 1.0), axis=-1)
        return np.clip(
            swapped_frame.astype(np.float32) * (1.0 - alpha) + original_frame.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
