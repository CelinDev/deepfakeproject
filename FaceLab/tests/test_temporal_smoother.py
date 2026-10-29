"""Testes unitários do TemporalSmoother e OneEuroFilter."""
from __future__ import annotations

import numpy as np
import pytest

from src.face.temporal_smoother import OneEuroFilter, TemporalSmoother


def _base_landmarks() -> np.ndarray:
    return np.array(
        [
            [100.0, 120.0],
            [160.0, 120.0],
            [130.0, 150.0],
            [110.0, 180.0],
            [150.0, 180.0],
        ],
        dtype=np.float32,
    )


def test_color_stats_alpha_semantics() -> None:
    """Garante que alpha_color (ex.: 0.15) pese 15% a nova amostra e 85% o histórico."""
    smoother = TemporalSmoother(alpha_color=0.15)
    z = np.zeros(3, dtype=np.float32)
    ones = np.full(3, 100.0, dtype=np.float32)

    smoother.smooth_color_stats(z, z, z, z)
    sm, _, _, _ = smoother.smooth_color_stats(ones, ones, ones, ones)
    assert np.allclose(sm, 15.0, atol=1e-4)


def test_one_euro_filter_reduces_stationary_jitter() -> None:
    """Com rosto parado + micro-tremor, a variância após o One Euro Filter deve cair drasticamente."""
    rng = np.random.default_rng(42)
    smoother = TemporalSmoother(min_cutoff=1.2, beta=0.05)
    base = _base_landmarks()
    bbox = np.array([80.0, 80.0, 180.0, 200.0], dtype=np.float32)

    raw_history = []
    smoothed_history = []
    for i in range(30):
        jitter = rng.normal(0.0, 1.2, size=base.shape).astype(np.float32)
        noisy = base + jitter
        raw_history.append(noisy)
        out = smoother.smooth_landmarks(noisy, timestamp_s=i / 30.0, bbox=bbox)
        smoothed_history.append(out)

    raw_std = float(np.std(np.stack(raw_history[5:]), axis=0).mean())
    smooth_std = float(np.std(np.stack(smoothed_history[5:]), axis=0).mean())
    assert smooth_std < raw_std * 0.6


def test_scene_cut_relative_to_face_size_resets_immediately() -> None:
    """Um deslocamento grande relativo ao tamanho do rosto (corte de cena) não deve arrastar."""
    smoother = TemporalSmoother(scene_cut_ratio=0.6)
    lmk1 = _base_landmarks()
    bbox1 = np.array([80.0, 80.0, 180.0, 200.0], dtype=np.float32)  # face_size ~ 110px
    smoother.smooth_landmarks(lmk1, timestamp_s=0.0, bbox=bbox1)

    # Desloca 300px (> 0.6 * 110px) -> corte de cena, deve assumir exatamente lmk2
    lmk2 = lmk1 + 300.0
    bbox2 = bbox1 + 300.0
    out = smoother.smooth_landmarks(lmk2, timestamp_s=0.033, bbox=bbox2)
    assert np.allclose(out, lmk2)
