"""Acompanhamento do rosto-alvo ao longo do vídeo.

Problema: num quadro com várias pessoas, qual rosto deve ser trocado?
E como garantir que seja sempre a mesma pessoa, quadro após quadro?

Estratégia (simples e explicável):
1. O usuário escolhe o alvo uma vez -> guardamos o embedding de identidade.
2. Em cada quadro, cada candidato recebe uma pontuação combinando:
   - semelhança de identidade (produto escalar dos embeddings), e
   - sobreposição (IoU) com a caixa do alvo no quadro anterior.
3. Se nenhum candidato for bom o suficiente, o alvo é considerado ausente
   naquele quadro (oculto, de costas, fora de cena) e o quadro não é alterado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import DetectedFace


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union entre duas caixas (x1, y1, x2, y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class TrackerConfig:
    identity_threshold: float = 0.35  # similaridade mínima de identidade
    identity_weight: float = 0.7
    iou_weight: float = 0.3
    max_missing_frames: int = 15  # após isso, a posição anterior deixa de ajudar


class TargetFaceTracker:
    """Mantém a seleção do rosto-alvo consistente durante o vídeo."""

    def __init__(self, target_embedding: np.ndarray, config: TrackerConfig | None = None) -> None:
        self.target_embedding = target_embedding / np.linalg.norm(target_embedding)
        self.config = config or TrackerConfig()
        self._last_bbox: np.ndarray | None = None
        self._missing = 0

    @staticmethod
    def select_largest(faces: list[DetectedFace]) -> DetectedFace | None:
        """Heurística padrão para escolher o alvo: o maior rosto."""
        return max(faces, key=lambda f: f.area, default=None)

    def update(self, faces: list[DetectedFace]) -> DetectedFace | None:
        """Retorna o rosto-alvo neste quadro, ou None se ele não aparece."""
        cfg = self.config
        best: DetectedFace | None = None
        best_score = -1.0
        use_position = self._last_bbox is not None and self._missing <= cfg.max_missing_frames

        for face in faces:
            overlap = iou(face.bbox, self._last_bbox) if use_position else 0.0
            if face.embedding is not None:
                similarity = float(np.dot(face.embedding, self.target_embedding))
                if similarity < cfg.identity_threshold:
                    continue  # identidade diferente: nunca troca a pessoa errada
                score = cfg.identity_weight * similarity + cfg.iou_weight * overlap
            else:
                # Se o embedding não foi calculado neste frame (modo rápido), confia no IoU com a última posição conhecida
                if overlap < 0.2:
                    continue
                score = overlap

            if score > best_score:
                best, best_score = face, score

        if best is None:
            self._missing += 1
        else:
            self._last_bbox = best.bbox.copy()
            self._missing = 0
        return best

