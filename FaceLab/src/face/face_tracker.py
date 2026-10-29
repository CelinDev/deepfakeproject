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

from config.settings import (
    TRACKER_EMBEDDING_EMA_ALPHA,
    TRACKER_EMBEDDING_UPDATE_SIM,
    TRACKER_VELOCITY_TOLERANCE_FRAMES,
)
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
    identity_threshold: float = 0.18  # mais flexível para rotações, expressões fortes e mudanças de luz
    identity_weight: float = 0.55
    iou_weight: float = 0.45
    max_missing_frames: int = 30  # mantém a memória espacial por até ~1 segundo
    grace_frames: int = 6         # segura o último alvo válido em movimentos bruscos/oclusões curtas
    velocity_tolerance_frames: int = TRACKER_VELOCITY_TOLERANCE_FRAMES
    embedding_ema_alpha: float = TRACKER_EMBEDDING_EMA_ALPHA
    embedding_update_similarity: float = TRACKER_EMBEDDING_UPDATE_SIM


class TargetFaceTracker:
    """Mantém a seleção do rosto-alvo consistente durante o vídeo."""

    def __init__(self, target_embedding: np.ndarray, config: TrackerConfig | None = None) -> None:
        self.target_embedding = target_embedding / np.linalg.norm(target_embedding)
        self.config = config or TrackerConfig()
        self._last_bbox: np.ndarray | None = None
        self._bbox_velocity: np.ndarray = np.zeros(4, dtype=np.float32)
        self._last_face: DetectedFace | None = None
        self._missing = 0

    @property
    def missing(self) -> int:
        """Quantidade de quadros consecutivos em que o rosto-alvo esteve ausente."""
        return self._missing

    def predicted_bbox(self) -> np.ndarray | None:
        """Estima a bbox atual extrapolando a velocidade linear da caixa anterior."""
        if self._last_bbox is None:
            return None
        steps = min(self._missing + 1, self.config.velocity_tolerance_frames)
        return (self._last_bbox + self._bbox_velocity * float(steps)).astype(np.float32)

    @staticmethod
    def select_largest(faces: list[DetectedFace]) -> DetectedFace | None:
        """Heurística padrão para escolher o alvo: o maior rosto."""
        return max(faces, key=lambda f: f.area, default=None)

    @staticmethod
    def sort_left_to_right(faces: list[DetectedFace]) -> list[DetectedFace]:
        """Ordena os rostos detectados da esquerda para a direita (coordenada x central)."""
        return sorted(faces, key=lambda f: f.center[0])

    @classmethod
    def select_by_index(cls, faces: list[DetectedFace], index: int) -> DetectedFace | None:
        """Seleciona o rosto pelo índice na lista ordenada da esquerda para a direita."""
        if not faces:
            return None
        ordered = cls.sort_left_to_right(faces)
        idx = int(np.clip(index, 0, len(ordered) - 1))
        return ordered[idx]

    @staticmethod
    def select_by_point(faces: list[DetectedFace], point_xy: tuple[float, float]) -> DetectedFace | None:
        """Seleciona o rosto clicado em ``point_xy = (x, y)`` (contido na bbox ou mais próximo do centro)."""
        if not faces:
            return None
        px, py = float(point_xy[0]), float(point_xy[1])
        inside = [
            f for f in faces
            if f.bbox[0] <= px <= f.bbox[2] and f.bbox[1] <= py <= f.bbox[3]
        ]
        if inside:
            return min(inside, key=lambda f: np.hypot(f.center[0] - px, f.center[1] - py))
        return min(faces, key=lambda f: np.hypot(f.center[0] - px, f.center[1] - py))

    @classmethod
    def select_target(
        cls,
        faces: list[DetectedFace],
        target_embedding: np.ndarray | None = None,
        target_index: int | None = None,
        target_point: tuple[float, float] | None = None,
    ) -> DetectedFace | None:
        """Seleciona manualmente o rosto-alvo inicial por embedding, clique (x, y), índice ou maior área."""
        if not faces:
            return None
        if target_embedding is not None:
            norm_t = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
            with_emb = [f for f in faces if f.embedding is not None]
            if with_emb:
                return max(with_emb, key=lambda f: float(np.dot(f.embedding, norm_t)))
        if target_point is not None:
            return cls.select_by_point(faces, target_point)
        if target_index is not None:
            return cls.select_by_index(faces, target_index)
        return cls.select_largest(faces)

    def update(self, faces: list[DetectedFace]) -> DetectedFace | None:
        """Retorna o rosto-alvo neste quadro, ou None se ele não aparece."""
        cfg = self.config
        best: DetectedFace | None = None
        best_score = -1.0
        best_similarity = -1.0
        use_position = self._last_bbox is not None and self._missing <= cfg.max_missing_frames
        pred_bbox = self.predicted_bbox() if use_position else None

        for face in faces:
            raw_overlap = iou(face.bbox, self._last_bbox) if use_position and self._last_bbox is not None else 0.0
            pred_overlap = iou(face.bbox, pred_bbox) if pred_bbox is not None else 0.0
            overlap = max(raw_overlap, pred_overlap)

            if face.embedding is not None:
                similarity = float(np.dot(face.embedding, self.target_embedding))
                # Se a pessoa já está na mesma posição espacial (IoU alto) ou é a única pessoa na cena,
                # relaxa o guardrail de identidade (ex: 0.10 em vez de 0.18) para não piscar ao virar o rosto de lado,
                # mas ainda bloqueia outra pessoa diferente (similaridade próxima de 0 ou negativa).
                effective_thresh = cfg.identity_threshold * 0.55 if (overlap >= 0.25 or len(faces) == 1) else cfg.identity_threshold
                if similarity < effective_thresh:
                    continue
                score = cfg.identity_weight * similarity + cfg.iou_weight * overlap
            else:
                similarity = -1.0
                # Modo rápido sem embedding: usa IoU com predição de velocidade e distância ao centro predito
                prox_score = 0.0
                if pred_bbox is not None and self._missing <= cfg.velocity_tolerance_frames:
                    pred_cx = 0.5 * float(pred_bbox[0] + pred_bbox[2])
                    pred_cy = 0.5 * float(pred_bbox[1] + pred_bbox[3])
                    pred_diag = float(np.hypot(pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1])) + 1e-6
                    fcx, fcy = face.center
                    dist_ratio = float(np.hypot(fcx - pred_cx, fcy - pred_cy)) / pred_diag
                    # Tolera deslocamentos de até ~1.2x a diagonal prevista durante velocity_tolerance_frames
                    if dist_ratio <= 1.25:
                        prox_score = max(0.0, 1.0 - 0.75 * dist_ratio)

                effective_spatial = max(overlap, prox_score * 0.5)
                if effective_spatial < 0.06 and len(faces) > 1:
                    continue
                score = effective_spatial if effective_spatial > 0 else 0.1

            if score > best_score:
                best, best_score, best_similarity = face, score, similarity

        if best is None:
            self._missing += 1
        else:
            new_bbox = best.bbox.astype(np.float32).copy()
            if self._last_bbox is not None and self._missing <= cfg.velocity_tolerance_frames:
                dt_frames = float(max(1, self._missing + 1))
                inst_vel = (new_bbox - self._last_bbox) / dt_frames
                self._bbox_velocity = 0.60 * self._bbox_velocity + 0.40 * inst_vel
            else:
                self._bbox_velocity = np.zeros(4, dtype=np.float32)

            self._last_bbox = new_bbox
            self._last_face = best
            self._missing = 0
            # Atualiza o embedding do alvo com média móvel lenta quando a similaridade for alta,
            # permitindo tolerar óculos, chapéus ou variações graduais de ângulo/iluminação
            if (
                best.embedding is not None
                and best_similarity >= cfg.embedding_update_similarity
                and best.score >= 0.55
            ):
                alpha_emb = float(np.clip(cfg.embedding_ema_alpha, 0.01, 0.50))
                blended = (1.0 - alpha_emb) * self.target_embedding + alpha_emb * best.embedding
                norm = np.linalg.norm(blended)
                if norm > 1e-6:
                    self.target_embedding = (blended / norm).astype(np.float32)
        return best

