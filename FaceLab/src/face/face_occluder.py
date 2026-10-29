"""Módulo Opcional de Oclusão Facial Genérica (XSeg / Face Occluder).

Enquanto o ``FaceParser`` (BiSeNet) conhece classes semânticas fixas (cabelo, óculos, orelhas),
o ``XSegOccluder`` detecta qualquer objeto arbitrário que passe na frente do rosto
(mãos, dedos, microfone, copos, celular) diretamente no espaço do crop alinhado.

Design resiliente (opcional):
- Se o arquivo ``models/xseg_2.onnx`` (ou ``models/dfl_xseg.onnx``) não estiver instalado,
  ``available`` será ``False`` e ``predict_mask`` retornará uma máscara neutra de ``1.0``
  sem interromper o funcionamento do programa.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config.settings import ALIGNED_MASK_SIZE, MODELS_DIR, XSEG_MODEL_PATH
from src.core.interfaces import FaceOccluder
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)


class XSegOccluder(FaceOccluder):
    """Implementação opcional de ``FaceOccluder`` compatível com XSeg ONNX (256x256)."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self._session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.input_size: tuple[int, int] = (256, 256)
        self._is_nhwc: bool = False

        if self.model_path is not None and self.model_path.is_file():
            try:
                providers = onnx_providers(device_manager)
                self._session = ort.InferenceSession(str(self.model_path), providers=providers)
                inp = self._session.get_inputs()[0]
                self.input_name = inp.name
                self.output_name = self._session.get_outputs()[0].name
                shape = inp.shape
                if len(shape) == 4:
                    # Detecta automaticamente se o grafo ONNX espera NHWC (1, 256, 256, 3) ou NCHW (1, 3, 256, 256)
                    if shape[-1] == 3:
                        self._is_nhwc = True
                        h = int(shape[1]) if isinstance(shape[1], int) else 256
                        w = int(shape[2]) if isinstance(shape[2], int) else 256
                        self.input_size = (w, h)
                    elif shape[1] == 3:
                        self._is_nhwc = False
                        h = int(shape[2]) if isinstance(shape[2], int) else 256
                        w = int(shape[3]) if isinstance(shape[3], int) else 256
                        self.input_size = (w, h)
                logger.info("XSegOccluder carregado (%s). Providers: %s", self.model_path.name, self._session.get_providers())
            except Exception as exc:
                logger.warning("Não foi possível carregar XSegOccluder (%s): %s", self.model_path, exc)
                self._session = None
        else:
            logger.debug("Modelo XSeg não encontrado em models/; oclusão de mãos/objetos ficará desativada.")

    @staticmethod
    def _resolve_model_path(model_path: str | Path | None) -> Path | None:
        if model_path is not None:
            return Path(model_path)
        for candidate in (XSEG_MODEL_PATH, MODELS_DIR / "dfl_xseg.onnx", MODELS_DIR / "face_occluder.onnx"):
            if candidate.is_file():
                return candidate
        return XSEG_MODEL_PATH

    @property
    def available(self) -> bool:
        return self._session is not None

    def preprocess(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Converte o crop BGR para tensor normalizado [0, 1] no formato esperado pelo XSeg."""
        resized = (
            aligned_crop
            if aligned_crop.shape[:2] == self.input_size
            else cv2.resize(aligned_crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._is_nhwc:
            return rgb[np.newaxis, ...].astype(np.float32)
        return rgb.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def predict_mask(self, aligned_crop: np.ndarray, out_size: int = ALIGNED_MASK_SIZE) -> np.ndarray:
        """Retorna máscara float32 (out_size, out_size) em [0.0, 1.0]."""
        if self._session is None:
            return np.ones((out_size, out_size), dtype=np.float32)

        blob = self.preprocess(aligned_crop)
        raw = self._session.run([self.output_name], {self.input_name: blob})[0]
        pred = np.squeeze(raw).astype(np.float32)

        # Se a saída vier em logits não limitados a [0, 1], aplica sigmoide
        if float(pred.min()) < -0.05 or float(pred.max()) > 1.05:
            pred = 1.0 / (1.0 + np.exp(-np.clip(pred, -20.0, 20.0)))

        mask = np.clip(pred, 0.0, 1.0)
        if mask.shape[:2] != (out_size, out_size):
            mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return mask.astype(np.float32)
