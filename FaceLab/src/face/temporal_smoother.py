"""Suavização Temporal de Marcos Faciais (One Euro Filter) e Iluminação (EMA).

Corrige dois problemas clássicos de vídeo em tempo real:
1. Nos marcos faciais (landmarks): utiliza o **One Euro Filter** (Casiez et al., CHI 2012)
   com velocidade normalizada pelo tamanho do rosto e ``timestamp_s`` real de cada quadro.
   Isso elimina o tremor (jitter) quando o rosto está parado sem introduzir atraso (lag)
   quando a pessoa se move rapidamente.
2. Nas estatísticas de cor/iluminação: utiliza EMA onde ``alpha_color`` (padrão 0.15)
   representa o **peso da nova amostra** e ``1 - alpha_color`` (0.85) representa a inércia
   histórica, evitando flicker de brilho entre quadros.
"""
from __future__ import annotations

import math

import numpy as np

from config.settings import (
    MASK_EMA_ALPHA,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    ONE_EURO_MIN_CUTOFF,
    SCENE_CUT_FACE_RATIO,
    SMOOTHER_ALPHA_COLOR,
    TARGET_HOLD_FRAMES,
)
from src.core.types import DetectedFace


def _smoothing_factor(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
    """Calcula o coeficiente alpha de um filtro passa-baixa de 1ª ordem para intervalo ``dt``."""
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """Implementação vetorial do One Euro Filter para pontos 2D normalizados pelo tamanho do rosto."""

    def __init__(
        self,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter(
        self,
        x: np.ndarray,
        timestamp_s: float | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Filtra um array ``x`` usando a derivada normalizada por ``scale`` (tamanho do rosto)."""
        x_f = np.asarray(x, dtype=np.float32)
        if self._x_prev is None:
            self._x_prev = x_f.copy()
            self._dx_prev = np.zeros_like(x_f)
            self._t_prev = timestamp_s
            return x_f

        if timestamp_s is not None and self._t_prev is not None:
            dt = float(timestamp_s - self._t_prev)
            if dt <= 1e-6:
                dt = 1.0 / 30.0
        else:
            dt = 1.0 / 30.0
        self._t_prev = timestamp_s

        safe_scale = max(float(scale), 1.0)
        # Velocidade normalizada pelo tamanho do rosto (unidades de rosto por segundo)
        dx = ((x_f - self._x_prev) / safe_scale) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        assert self._dx_prev is not None
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Frequência de corte adaptativa: parado -> min_cutoff; rápido -> sobe com |dx_hat|
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a_x = _smoothing_factor(dt, cutoff)
        x_hat = a_x * x_f + (1.0 - a_x) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat.astype(np.float32)


class TemporalSmoother:
    """Suaviza marcos faciais (via One Euro Filter) e parâmetros de cor (via EMA)."""

    def __init__(
        self,
        alpha_color: float = SMOOTHER_ALPHA_COLOR,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
        scene_cut_ratio: float = SCENE_CUT_FACE_RATIO,
        hold_frames: int = TARGET_HOLD_FRAMES,
        alpha_mask: float = MASK_EMA_ALPHA,
        alpha_landmarks: float | None = None,  # Mantido por compatibilidade retroativa
    ) -> None:
        """
        Args:
            alpha_color: Peso da nova amostra nas estatísticas de cor [0..1].
                         Valores baixos (~0.10 a 0.20) dão alta inércia (80–90% do histórico),
                         suavizando variações bruscas de brilho e cor entre quadros.
            min_cutoff: Frequência de corte mínima (Hz) do One Euro Filter (suavização em repouso).
            beta: Ganho de velocidade do One Euro Filter (reduz latência em movimento rápido).
            d_cutoff: Frequência de corte (Hz) para estimativa da derivada.
            scene_cut_ratio: Limiar de deslocamento relativo ao tamanho do rosto para detectar
                             corte de cena e reiniciar o filtro sem arrasto.
            hold_frames: Quantidade de quadros para manter o último alvo com fade-out progressivo
                         do alpha antes de resetar quando o rosto é perdido.
        """
        self.alpha_color = float(np.clip(alpha_color, 0.01, 1.0))
        self.alpha_mask = float(np.clip(alpha_mask, 0.01, 1.0))
        self.scene_cut_ratio = float(scene_cut_ratio)
        self.hold_frames = max(0, int(hold_frames))
        self.one_euro = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.prev_landmarks: np.ndarray | None = None
        self.prev_bbox: np.ndarray | None = None
        self.prev_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.prev_mask: np.ndarray | None = None
        self.missing_count: int = 0

    @staticmethod
    def estimate_face_size(landmarks: np.ndarray, bbox: np.ndarray | None = None) -> float:
        """Estima a escala do rosto (em pixels) a partir da bbox ou da extensão dos landmarks."""
        if bbox is not None and len(bbox) >= 4:
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])
            size = math.hypot(w, h) / math.sqrt(2.0)
            if size > 1.0:
                return size
        span = np.ptp(landmarks, axis=0)
        # Os 5 pontos internos (olhos ao canto da boca) cobrem ~45% da altura total da face
        return max(float(np.linalg.norm(span)) * 2.0, 1.0)

    def smooth_landmarks(
        self,
        landmarks: np.ndarray,
        timestamp_s: float | None = None,
        bbox: np.ndarray | None = None,
    ) -> np.ndarray:
        """Suaviza os 5 pontos faciais usando One Euro Filter normalizado pelo tamanho do rosto."""
        lmk = np.asarray(landmarks, dtype=np.float32)
        face_size = self.estimate_face_size(lmk, bbox=bbox)

        if self.prev_landmarks is not None:
            max_disp = float(np.max(np.linalg.norm(lmk - self.prev_landmarks, axis=1)))
            # Detecta corte de cena de forma relativa ao tamanho do rosto (independente da resolução)
            if max_disp > self.scene_cut_ratio * face_size:
                self.one_euro.reset()

        smoothed = self.one_euro.filter(lmk, timestamp_s=timestamp_s, scale=face_size)
        self.prev_landmarks = smoothed.copy()
        if bbox is not None:
            self.prev_bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.missing_count = 0
        return smoothed

    def hold_or_reset(self) -> tuple[DetectedFace | None, float]:
        """Mantém o último rosto/landmarks por ``hold_frames`` quadros com fade-out de alpha.

        Returns:
            (held_face, fade_factor): se dentro da janela de hold, retorna um ``DetectedFace``
            com os últimos landmarks conhecidos e um fator de opacidade decrescente (0..1).
            Após ``hold_frames`` quadros ausentes, executa ``reset()`` e retorna ``(None, 0.0)``.
        """
        if self.prev_landmarks is None or self.hold_frames <= 0:
            self.reset()
            return None, 0.0

        self.missing_count += 1
        if self.missing_count > self.hold_frames:
            self.reset()
            return None, 0.0

        # Fade-out linear suave: ex. para hold_frames=6 -> 6/7, 5/7, ..., 1/7
        fade_factor = float(self.hold_frames - self.missing_count + 1) / float(self.hold_frames + 1)
        if self.prev_bbox is not None:
            bbox = self.prev_bbox.copy()
        else:
            min_xy = self.prev_landmarks.min(axis=0) - 20.0
            max_xy = self.prev_landmarks.max(axis=0) + 20.0
            bbox = np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)

        held_face = DetectedFace(
            bbox=bbox,
            landmarks=self.prev_landmarks.copy(),
            score=fade_factor,
            embedding=None,
        )
        return held_face, fade_factor

    def smooth_color_stats(
        self, s_mean: np.ndarray, s_std: np.ndarray, t_mean: np.ndarray, t_std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Suaviza médias e desvios de cor com EMA (alpha_color = peso da nova amostra)."""
        if self.prev_stats is None:
            self.prev_stats = (s_mean.copy(), s_std.copy(), t_mean.copy(), t_std.copy())
            return s_mean, s_std, t_mean, t_std

        a = self.alpha_color
        inertia = 1.0 - a
        p_sm, p_ss, p_tm, p_ts = self.prev_stats
        sm = a * s_mean + inertia * p_sm
        ss = a * s_std + inertia * p_ss
        tm = a * t_mean + inertia * p_tm
        ts = a * t_std + inertia * p_ts

        self.prev_stats = (sm, ss, tm, ts)
        return sm, ss, tm, ts

    def smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """Aplica EMA temporal na máscara no espaço do crop alinhado (evita flicker de bordas/oclusões)."""
        m = np.asarray(mask, dtype=np.float32)
        if self.prev_mask is None or self.prev_mask.shape != m.shape:
            self.prev_mask = m.copy()
            return m
        a = self.alpha_mask
        smoothed = a * m + (1.0 - a) * self.prev_mask
        self.prev_mask = smoothed.copy()
        return smoothed.astype(np.float32)

    def reset(self) -> None:
        """Reinicia os estados anteriores dos filtros de landmarks, bbox, máscara, hold e cor."""
        self.one_euro.reset()
        self.prev_landmarks = None
        self.prev_bbox = None
        self.prev_stats = None
        self.prev_mask = None
        self.missing_count = 0
