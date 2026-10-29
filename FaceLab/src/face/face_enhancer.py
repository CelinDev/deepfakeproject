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

from config.settings import ENHANCER_DEFAULT_BLEND_RATIO, MODELS_DIR, SWAPPER_WORK_SIZE
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)


class FaceEnhancer:
    """Restaura detalhes e nitidez facial utilizando GFPGAN 1.4."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_path: str | Path | None = None,
        blend_ratio: float = ENHANCER_DEFAULT_BLEND_RATIO,
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
        self.input_size: tuple[int, int] = (SWAPPER_WORK_SIZE, SWAPPER_WORK_SIZE)
        self._frame_count: int = 0
        self._cached_delta_512: np.ndarray | None = None

    def reset_cache(self) -> None:
        """Limpa o cache temporal de detalhes do GFPGAN."""
        self._frame_count = 0
        self._cached_delta_512 = None

    def enhance_fast(self, face_bgr: np.ndarray, amount: float = 0.35) -> np.ndarray:
        """
        Realce leve de micro-contraste restrito exclusivamente ao recorte (crop) do rosto.
        Operação rápida (~1 a 3 ms em um crop 512x512 na CPU).
        """
        gaussian = cv2.GaussianBlur(face_bgr, (0, 0), sigmaX=1.8, sigmaY=1.8)
        return cv2.addWeighted(face_bgr, 1.0 + amount, gaussian, -amount, 0)

    def enhance(self, face_bgr: np.ndarray, reuse_interval: int = 1) -> np.ndarray:
        """Restaura e retorna o recorte facial BGR em 512x512 (resolução nativa do GFPGAN).

        Args:
            face_bgr: Recorte facial BGR alinhado.
            reuse_interval: Se > 1, executa a rede GFPGAN a cada ``reuse_interval`` quadros
                            e reutiliza/interpola o residual de alta frequência nos quadros
                            intermediários com realce leve no crop.
        """
        self._frame_count += 1
        resized_512 = (
            face_bgr
            if face_bgr.shape[:2] == self.input_size
            else cv2.resize(face_bgr, self.input_size, interpolation=cv2.INTER_CUBIC)
        )

        # Se estivermos em um quadro intermediário do modo live, reaproveita o detalhe residual do GFPGAN
        if (
            reuse_interval > 1
            and self._cached_delta_512 is not None
            and (self._frame_count - 1) % reuse_interval != 0
        ):
            interpolated = np.clip(
                resized_512.astype(np.float32) + self._cached_delta_512, 0.0, 255.0
            ).astype(np.uint8)
            return self.enhance_fast(interpolated, amount=0.20)

        # 2. BGR -> RGB e normalização para [-1.0, 1.0]
        rgb = cv2.cvtColor(resized_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - 0.5) / 0.5  # Escala [-1, 1]

        # 3. Transpõe para NCHW: (1, 3, 512, 512)
        blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # 4. Inferência GFPGAN
        output = self.session.run([self.output_name], {self.input_name: blob})[0]

        # 5. Desnormalização [-1, 1] -> [0, 255] e NCHW -> HWC (512x512)
        img_out = output[0].transpose((1, 2, 0))
        img_out = np.clip((img_out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        enhanced_512 = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # 6. Mistura proporcional em 512x512 para preservar naturalidade sem perder resolução
        if self.blend_ratio < 1.0:
            final_512 = cv2.addWeighted(
                enhanced_512, self.blend_ratio, resized_512, 1.0 - self.blend_ratio, 0
            )
        else:
            final_512 = enhanced_512

        # Armazena o residual de super-resolução para interpolar nos quadros intermediários
        self._cached_delta_512 = final_512.astype(np.float32) - resized_512.astype(np.float32)
        return final_512

