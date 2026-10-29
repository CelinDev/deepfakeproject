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

from config.settings import ALIGNED_MASK_SIZE, BISENET_MODEL_PATH
from src.core.types import DetectedFace
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)

# Mapeamento das 19 classes CelebAMask-HQ
FACE_PARSER_CLASSES: dict[int, str] = {
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

# Classes que compõem a face interna que DEVE ser trocada (pele, sobrancelhas, olhos, nariz e lábios)
FACE_SWAP_FOREGROUND_CLASSES: tuple[int, ...] = (
    1,   # skin (pele facial)
    2,   # l_brow (sobrancelha esq)
    3,   # r_brow (sobrancelha dir)
    4,   # l_eye (olho esq)
    5,   # r_eye (olho dir)
    10,  # nose (nariz)
    12,  # u_lip (lábio superior)
    13,  # l_lip (lábio inferior)
)

# Classes de oclusão na máscara do crop alinhado (cabelo, orelhas, pescoço, óculos, dentes/interior da boca, chapéu, roupa)
ALIGNED_OCCLUSION_CLASSES: tuple[int, ...] = (
    0,   # background
    6,   # eye_g (óculos)
    7,   # l_ear (orelha esq)
    8,   # r_ear (orelha dir)
    9,   # ear_r (brincos)
    11,  # mouth (interior da boca / dentes reais)
    14,  # neck (pescoço)
    15,  # neck_l (colar)
    16,  # cloth (roupa)
    17,  # hair (cabelo)
    18,  # hat (chapéu)
)

# Classes de oclusão e proteção de fundo (usadas pelo método legado restore_natural_hair_and_ears)
NATURAL_HAIR_AND_OCCLUSION_CLASSES = frozenset(ALIGNED_OCCLUSION_CLASSES)


class FaceParser:
    """Segmentador semântico baseado em BiSeNet para preservação orgânica de cabelos e orelhas."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else BISENET_MODEL_PATH
        if not self.model_path.is_file():
            self._download_model()

        providers = onnx_providers(device_manager)
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self._session.get_inputs()[0].name
        self.output_name = self._session.get_outputs()[0].name
        self._cached_mask_roi: np.ndarray | None = None
        self._cached_aligned_prob: np.ndarray | None = None
        self._cached_skin_prob: np.ndarray | None = None
        self._frame_counter: int = 0
        logger.info("FaceParser (BiSeNet) carregado. Providers: %s", self._session.get_providers())

    def reset_cache(self) -> None:
        """Limpa os caches de máscara temporal."""
        self._cached_mask_roi = None
        self._cached_aligned_prob = None
        self._cached_skin_prob = None
        self._frame_counter = 0

    @property
    def last_skin_mask(self) -> np.ndarray | None:
        """Retorna a última máscara contínua [0.0, 1.0] apenas da classe pele (skin=1)."""
        return self._cached_skin_prob

    @staticmethod
    def _softmax_channels(logits: np.ndarray) -> np.ndarray:
        """Calcula softmax numericamente estável ao longo do eixo de canais (C, H, W)."""
        shifted = logits - np.max(logits, axis=0, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-8)

    def parse_probabilities(
        self,
        aligned_crop: np.ndarray,
        out_size: int = ALIGNED_MASK_SIZE,
        reuse_interval: int = 1,
    ) -> np.ndarray:
        """Calcula o mapa de máscara contínua [0.0, 1.0] no espaço do crop alinhado usando probabilidades softmax.

        Soma as probabilidades das classes faciais internas (pele, sobrancelhas, olhos, nariz e lábios)
        e desconta explicitamente as probabilidades de oclusão (cabelo, orelhas, pescoço, óculos, dentes),
        sem usar ``argmax`` nem ``INTER_NEAREST``.
        """
        self._frame_counter += 1
        if (
            reuse_interval > 1
            and self._cached_aligned_prob is not None
            and (self._frame_counter - 1) % reuse_interval != 0
        ):
            if self._cached_aligned_prob.shape[:2] == (out_size, out_size):
                return self._cached_aligned_prob
            return cv2.resize(self._cached_aligned_prob, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

        blob = self.preprocess(aligned_crop, size=(512, 512))
        logits = self._session.run([self.output_name], {self.input_name: blob})[0][0]  # (19, 512, 512)
        probs = self._softmax_channels(logits)

        face_prob = np.sum(probs[list(FACE_SWAP_FOREGROUND_CLASSES)], axis=0)
        occ_prob = np.sum(probs[list(ALIGNED_OCCLUSION_CLASSES)], axis=0)
        # Combina a probabilidade contínua da face com a supressão explícita das oclusões
        soft_mask_512 = np.clip(face_prob * (1.0 - occ_prob), 0.0, 1.0).astype(np.float32)
        skin_mask_512 = np.clip(probs[1] * (1.0 - occ_prob), 0.0, 1.0).astype(np.float32)

        if out_size != 512:
            soft_mask = cv2.resize(soft_mask_512, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            skin_mask = cv2.resize(skin_mask_512, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        else:
            soft_mask = soft_mask_512
            skin_mask = skin_mask_512

        self._cached_aligned_prob = soft_mask
        self._cached_skin_prob = skin_mask
        return soft_mask

    def _download_model(self) -> None:
        """Baixa o modelo BiSeNet ResNet-34 automaticamente se não estiver presente."""
        import urllib.request

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx"
        logger.info("Baixando modelo BiSeNet (segmentação de cabeça e cabelo)...")
        urllib.request.urlretrieve(url, str(self.model_path))
        logger.info("Download do BiSeNet concluído!")

    @staticmethod
    def preprocess(image_bgr: np.ndarray, size: tuple[int, int] = (512, 512)) -> np.ndarray:
        """Pré-processa uma imagem BGR para o tensor NCHW float32 esperado pelo BiSeNet (ImageNet RGB).

        Nota: anteriormente ``cv2.dnn.blobFromImage`` recebia ``mean=(0.485, 0.456, 0.406)`` na escala
        0..255 antes de multiplicar por ``1/255``, o que quase não subtraía a média. Aqui normalizamos
        corretamente em ``[0, 1]`` e aplicamos ``(rgb - mean) / std``.
        """
        resized = (
            image_bgr
            if image_bgr.shape[:2] == size
            else cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        return norm.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def parse(self, image: np.ndarray) -> np.ndarray:
        """Executa a inferência e retorna mapa de índices 2D (0 a 18)."""
        h, w = image.shape[:2]
        blob = self.preprocess(image, size=(512, 512))

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
        reuse_interval: int = 1,
    ) -> np.ndarray:
        """
        Restaura com fidelidade orgânica o cabelo real, orelhas, franja e óculos da pessoa original.

        O rosto sintético fica posicionado exatamente sob os fios e detalhes originais,
        acompanhando a física, balanço e iluminação da cena com perfeição fotorrealista.
        """
        h, w = original_frame.shape[:2]
        self._frame_counter += 1

        # Se houver bounding box, fazemos na ROI expandida para altíssima performance
        if target_face is not None and target_face.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
            pad_x = int((x2 - x1) * 0.40)
            pad_y = int((y2 - y1) * 0.40)
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            roi_orig = original_frame[ry1:ry2, rx1:rx2]
            roi_swap = swapped_frame[ry1:ry2, rx1:rx2]
            roi_h, roi_w = roi_orig.shape[:2]

            if (
                reuse_interval > 1
                and self._cached_mask_roi is not None
                and self._frame_counter % reuse_interval != 0
            ):
                mask_roi = cv2.resize(self._cached_mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            else:
                mask_roi = self.extract_mask(
                    roi_orig,
                    classes=NATURAL_HAIR_AND_OCCLUSION_CLASSES,
                    feather=feather,
                )
                self._cached_mask_roi = mask_roi

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
