"""Detector facial baseado no InsightFace (pacote de modelos buffalo_l).

Licença: o código do InsightFace é MIT, mas os pesos do buffalo_l são de uso
**não comercial** (pesquisa/educação). Ver models/README.md.

Modelos carregados:
- ``detection`` (SCRFD): caixas + 5 pontos faciais.
- ``recognition`` (ArcFace): embedding de identidade, usado para seguir
  sempre a mesma pessoa no vídeo e, na Fase 4, pelo face swap.
"""
from __future__ import annotations

import logging

import numpy as np

from config.settings import MODELS_DIR
from src.core.interfaces import FaceDetector
from src.core.types import DetectedFace
from src.inference.device_manager import DeviceManager
from src.inference.model_loader import onnx_providers

logger = logging.getLogger(__name__)


class InsightFaceDetector(FaceDetector):
    """Implementação de ``FaceDetector`` usando ``insightface.app.FaceAnalysis``."""

    def __init__(
        self,
        device_manager: DeviceManager,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        with_embedding: bool = True,
    ) -> None:
        # Import tardio: os testes que não usam o modelo não precisam do insightface.
        from insightface.app import FaceAnalysis

        modules = ["detection", "recognition"] if with_embedding else ["detection"]
        providers = onnx_providers(device_manager)
        # root=MODELS_DIR -> os pesos ficam em FaceLab/models/models/buffalo_l
        # (baixados automaticamente na primeira execução).
        self._app = FaceAnalysis(
            name=model_name, root=str(MODELS_DIR),
            allowed_modules=modules, providers=providers,
        )
        ctx_id = 0 if device_manager.info.is_cuda else -1
        self._app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        self.active_providers = self._session_providers()
        logger.info("Detector pronto. Providers ativos: %s", self.active_providers)

    def _session_providers(self) -> list[str]:
        """Providers que a sessão ONNX realmente está usando (não só os pedidos)."""
        session = getattr(self._app.det_model, "session", None)
        return list(session.get_providers()) if session is not None else []

    def detect(self, image: np.ndarray, compute_embeddings: bool = True) -> list[DetectedFace]:
        if compute_embeddings or "recognition" not in self._app.models:
            faces = self._app.get(image)
        else:
            # OTIMIZAÇÃO (Fase 6): executa somente detecção de caixas e 5 marcos faciais (SCRFD),
            # sem rodar a rede neural profunda ArcFace (que só é necessária quando rastreamos por identidade).
            bboxes, kpss = self._app.det_model.detect(image, max_num=0, metric="default")
            faces = []
            if bboxes.shape[0] > 0:
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    kps = kpss[i] if kpss is not None else None
                    from insightface.app.common import Face
                    faces.append(Face(bbox=bbox, kps=kps, det_score=det_score))

        result: list[DetectedFace] = []
        for f in faces:
            emb = f.normed_embedding if getattr(f, "embedding", None) is not None else None
            result.append(DetectedFace(
                bbox=np.asarray(f.bbox, dtype=np.float32),
                landmarks=None if f.kps is None else np.asarray(f.kps, dtype=np.float32),
                score=float(f.det_score),
                embedding=None if emb is None else np.asarray(emb, dtype=np.float32),
            ))
        return result

