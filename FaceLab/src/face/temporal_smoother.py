"""Filtro Temporal Exponencial (EMA - Exponential Moving Average) para Suavização Facial.

Evita o "jitter" (trepidação de micro-pixels entre quadros consecutivos)
nos marcos faciais (landmarks) e na iluminação média, eliminando o efeito de
flicker / estroboscópico no vídeo final.
"""
from __future__ import annotations

import numpy as np


class TemporalSmoother:
    """Suaviza marcos faciais e parâmetros de cor ao longo do tempo."""

    def __init__(self, alpha_landmarks: float = 0.65, alpha_color: float = 0.8) -> None:
        """
        Args:
            alpha_landmarks: Fator de inércia para marcos faciais [0..1].
                             Valores em torno de 0.6-0.7 mantêm movimentos rápidos da cabeça
                             enquanto absorvem tremores de sub-pixel.
            alpha_color: Fator de inércia para a média/desvio de cor [0..1].
                         Valores altos evitam piscadas de brilho entre quadros.
        """
        self.alpha_landmarks = alpha_landmarks
        self.alpha_color = alpha_color
        self.prev_landmarks: np.ndarray | None = None
        self.prev_stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def smooth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Aplica filtro passa-baixa temporal nos 5 pontos faciais."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.copy()
            return landmarks

        # Se o rosto se movimentou bruscamente (corte de cena ou virada rápida), reseta
        max_dist = np.max(np.linalg.norm(landmarks - self.prev_landmarks, axis=1))
        if max_dist > 45.0:  # Mudança de mais de 45px entre quadros = movimento grande / corte
            self.prev_landmarks = landmarks.copy()
            return landmarks

        smoothed = self.alpha_landmarks * landmarks + (1.0 - self.alpha_landmarks) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed.astype(np.float32)

    def smooth_color_stats(
        self, s_mean: np.ndarray, s_std: np.ndarray, t_mean: np.ndarray, t_std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Suaviza médias e desvios de cor para impedir variações bruscas de iluminação."""
        if self.prev_stats is None:
            self.prev_stats = (s_mean, s_std, t_mean, t_std)
            return s_mean, s_std, t_mean, t_std

        p_sm, p_ss, p_tm, p_ts = self.prev_stats
        sm = self.alpha_color * s_mean + (1.0 - self.alpha_color) * p_sm
        ss = self.alpha_color * s_std + (1.0 - self.alpha_color) * p_ss
        tm = self.alpha_color * t_mean + (1.0 - self.alpha_color) * p_tm
        ts = self.alpha_color * t_std + (1.0 - self.alpha_color) * p_ts

        self.prev_stats = (sm, ss, tm, ts)
        return sm, ss, tm, ts

    def reset(self) -> None:
        """Reinicia os estados anteriores."""
        self.prev_landmarks = None
        self.prev_stats = None

