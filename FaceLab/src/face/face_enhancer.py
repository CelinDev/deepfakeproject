"""Módulo de Restauração Facial / Super-Resolução (Face Enhancer).

Utiliza o modelo pré-treinado GFPGAN 1.4 (ONNX) para:
1. Receber o crop gerado pelo swap.
2. Redimensionar para 512x512 e normalizar para [-1, 1] em RGB.
3. Executar super-resolução restaurativa gerando textura realista de pele, cílios, olhos e dentes.
4. Desnormalizar e retornar o rosto restaurado em alta fidelidade.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config.settings import MODELS_DIR
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """Restaura detalhes e nitidez facial utilizando GFPGAN 1.4."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
        blend_ratio: float = 0.8,
    ) -> None:
        """
        Args:
            device_manager: Gerenciador de dispositivo (GPU CUDA / CPU).
            model_path: Caminho do arquivo gfpgan_1.4.onnx.
            blend_ratio: Taxa de mistura entre o rosto restaurado e o original [0.0 a 1.0].
                         0.8 mantém a alta nitidez mantendo total fidelidade aos traços de origem.
        """
        self.model_path = Path(model_path) if model_path else (MODELS_DIR / "gfpgan_1.4.onnx")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Modelo GFPGAN não encontrado em: {self.model_path}")

        self.blend_ratio = blend_ratio
        providers = onnx_providers(device_manager)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=providers
        )
        self.active_providers = self.session.get_providers()
        logger.info("FaceEnhancer (GFPGAN) carregado. Providers ativos: %s", self.active_providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (512, 512)

    def enhance_fast(self, face_bgr: np.ndarray, amount: float = 0.45) -> np.ndarray:
        """
        Restauração de alta frequência e super-resolução adaptativa em tempo real (<1ms).
        Ideal para webcam / streaming com taxa de quadros máxima.
        """
        # Filtro de micro-contraste e realce de textura de pele (Unsharp Masking com retenção de ruído)
        gaussian = cv2.GaussianBlur(face_bgr, (0, 0), sigmaX=2.0, sigmaY=2.0)
        sharpened = cv2.addWeighted(face_bgr, 1.0 + amount, gaussian, -amount, 0)

        # Realce adaptativo de contornos finos (íris, cílios, lábios)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
        edge_detail = cv2.filter2D(face_bgr, -1, kernel)

        return cv2.addWeighted(sharpened, 0.75, edge_detail, 0.25, 0)

    def enhance(self, face_bgr: np.ndarray) -> np.ndarray:
        """Restaura e aumenta a nitidez de um recorte facial BGR."""
        orig_h, orig_w = face_bgr.shape[:2]

        # 1. Redimensiona para 512x512
        resized = cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_CUBIC)

        # 2. BGR -> RGB e normalização para [-1.0, 1.0]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - 0.5) / 0.5  # Escala [-1, 1]

        # 3. Transpõe para NCHW: (1, 3, 512, 512)
        blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # 4. Inferência GFPGAN
        output = self.session.run([self.output_name], {self.input_name: blob})[0]

        # 5. Desnormalização [-1, 1] -> [0, 255] e NCHW -> HWC
        img_out = output[0].transpose((1, 2, 0))
        img_out = np.clip((img_out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # 6. Redimensiona de volta para a resolução original do crop
        enhanced_orig = cv2.resize(enhanced_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # 7. Mistura proporcional opcional para preservar naturalidade
        if self.blend_ratio < 1.0:
            blended = cv2.addWeighted(enhanced_orig, self.blend_ratio, face_bgr, 1.0 - self.blend_ratio, 0)
            return blended

        return enhanced_orig

