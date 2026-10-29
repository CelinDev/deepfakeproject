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

import cv2
import numpy as np

from config.settings import (
    OPTICAL_FLOW_MIN_CONFIDENCE,
    TRACKER_EMBEDDING_EMA_ALPHA,
    TRACKER_EMBEDDING_UPDATE_SIM,
    TRACKER_GRACE_FRAMES,
    TRACKER_IDENTITY_THRESHOLD,
    TRACKER_IDENTITY_WEIGHT,
    TRACKER_IOU_WEIGHT,
    TRACKER_MAX_MISSING_FRAMES,
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
    identity_threshold: float = TRACKER_IDENTITY_THRESHOLD  # mais flexível para rotações, expressões fortes e mudanças de luz
    identity_weight: float = TRACKER_IDENTITY_WEIGHT
    iou_weight: float = TRACKER_IOU_WEIGHT
    max_missing_frames: int = TRACKER_MAX_MISSING_FRAMES  # mantém a memória espacial por até ~1 segundo
    grace_frames: int = TRACKER_GRACE_FRAMES              # segura o último alvo válido em movimentos bruscos/oclusões curtas
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


class OpticalFlowPropagator:
    """Propaga landmarks (5 e 106 pontos) e bbox nos quadros intermediários usando Lucas-Kanade piramidal.

    Inclui verificação forward-backward de erro para medir a confiança da propagação;
    se a confiança cair abaixo de ``min_confidence``, retorna ``(None, 0.0)`` para forçar
    redetecção imediata pelo SCRFD.
    """

    def __init__(self, min_confidence: float = OPTICAL_FLOW_MIN_CONFIDENCE) -> None:
        self.min_confidence = float(min_confidence)
        self._prev_gray: np.ndarray | None = None
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )

    def reset(self) -> None:
        self._prev_gray = None

    def update_reference(self, frame_bgr: np.ndarray) -> None:
        """Atualiza o quadro cinza de referência após uma detecção SCRFD completa."""
        self._prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def propagate(
        self,
        frame_bgr: np.ndarray,
        prev_face: DetectedFace | None,
    ) -> tuple[DetectedFace | None, float]:
        """Propaga ``prev_face`` para ``frame_bgr`` e retorna ``(propagated_face, confidence)``."""
        if prev_face is None or prev_face.landmarks is None or self._prev_gray is None:
            return None, 0.0

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if curr_gray.shape != self._prev_gray.shape:
            self._prev_gray = curr_gray
            return None, 0.0

        pts5 = prev_face.landmarks.astype(np.float32)
        has_106 = prev_face.landmarks_106 is not None and prev_face.landmarks_106.shape == (106, 2)
        if has_106:
            all_pts = np.vstack([pts5, prev_face.landmarks_106.astype(np.float32)])
        else:
            all_pts = pts5

        p0 = all_pts.reshape(-1, 1, 2)
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, curr_gray, p0, None, **self._lk_params)
        if p1 is None or st_fwd is None:
            self._prev_gray = curr_gray
            return None, 0.0

        # Consistência forward-backward para detectar oclusão repentina ou perda de rastreio
        p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self._prev_gray, p1, None, **self._lk_params)
        self._prev_gray = curr_gray
        if p0_back is None or st_bwd is None:
            return None, 0.0

        valid_mask = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1)
        valid_ratio = float(np.mean(valid_mask[:5]))
        if valid_ratio < 0.80:
            return None, 0.0

        fb_err = np.linalg.norm((p0 - p0_back).reshape(-1, 2)[:5], axis=1)
        face_w = max(float(prev_face.bbox[2] - prev_face.bbox[0]), 16.0)
        mean_rel_err = float(np.mean(fb_err)) / face_w
        # Confiança decai suavemente conforme o erro forward-backward relativo cresce
        confidence = float(np.clip(valid_ratio * np.exp(-12.0 * mean_rel_err), 0.0, 1.0))
        if confidence < self.min_confidence:
            return None, confidence

        tracked_pts = p1.reshape(-1, 2).astype(np.float32)
        new_pts5 = tracked_pts[:5]
        new_pts106 = tracked_pts[5:] if has_106 else None

        # Atualiza a bbox aplicando a translação e escala mediana dos pontos rastreados
        delta = np.median(new_pts5 - pts5, axis=0)
        new_bbox = prev_face.bbox.astype(np.float32).copy()
        new_bbox[0] += float(delta[0])
        new_bbox[2] += float(delta[0])
        new_bbox[1] += float(delta[1])
        new_bbox[3] += float(delta[1])

        propagated = DetectedFace(
            bbox=new_bbox,
            landmarks=new_pts5,
            score=float(prev_face.score * confidence),
            embedding=prev_face.embedding,
            landmarks_106=new_pts106,
        )
        return propagated, confidence

