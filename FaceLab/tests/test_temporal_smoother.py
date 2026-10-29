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


def test_hold_and_fade_out_before_reset() -> None:
    """Garante que o smoother segure o último transform por hold_frames com fade-out antes de resetar."""
    smoother = TemporalSmoother(hold_frames=6)
    lmk = _base_landmarks()
    bbox = np.array([80.0, 80.0, 180.0, 200.0], dtype=np.float32)
    smoother.smooth_landmarks(lmk, timestamp_s=0.0, bbox=bbox)

    fades = []
    for _ in range(6):
        held, fade = smoother.hold_or_reset()
        assert held is not None
        assert held.landmarks is not None and np.allclose(held.landmarks, lmk)
        fades.append(fade)

    # Os fatores de fade-out devem ser estritamente decrescentes e entre (0, 1)
    assert all(0.0 < f < 1.0 for f in fades)
    assert fades == sorted(fades, reverse=True)

    # No 7º quadro ausente (após hold_frames=6), deve resetar de vez
    held_after, fade_after = smoother.hold_or_reset()
    assert held_after is None
    assert fade_after == 0.0
    assert smoother.prev_landmarks is None


def test_landmark_106_and_transform_param_smoothing_reduces_jitter() -> None:
    """Compara o jitter da matriz de alinhamento entre 5 pontos ruidosos vs 106 pontos + suavização dos parâmetros."""
    from src.face.face_aligner import FaceAligner

    rng = np.random.default_rng(123)
    base_5 = _base_landmarks()
    # Constrói malha base sintética de 106 pontos alinhada com as regiões anatômicas de base_5
    base_106 = np.zeros((106, 2), dtype=np.float32)
    base_106[33:43] = base_5[0]
    base_106[87:97] = base_5[1]
    base_106[[80, 85, 86]] = base_5[2]
    base_106[[52, 64, 65]] = base_5[3]
    base_106[[61, 68, 69]] = base_5[4]

    smoother_5 = TemporalSmoother(min_cutoff=1.5, beta=0.05)
    smoother_106 = TemporalSmoother(min_cutoff=1.5, beta=0.05)
    bbox = np.array([80.0, 80.0, 180.0, 200.0], dtype=np.float32)

    tx_5_list: list[float] = []
    tx_106_param_list: list[float] = []

    for i in range(40):
        t = i / 30.0
        noisy_5 = base_5 + rng.normal(0.0, 1.8, size=base_5.shape).astype(np.float32)
        noisy_106 = base_106 + rng.normal(0.0, 1.8, size=base_106.shape).astype(np.float32)

        # Abordagem anterior: suaviza apenas os 5 pontos e estima a matriz
        sm_5 = smoother_5.smooth_landmarks(noisy_5, timestamp_s=t, bbox=bbox)
        m_5 = FaceAligner.estimate_affine_matrix(sm_5, crop_size=512)
        tx_5_list.append(float(m_5[0, 2]))

        # Nova abordagem (Seção 4): extrai de 106 pontos + suaviza os parâmetros da transformação de similaridade
        sm_5_for_106 = smoother_106.smooth_landmarks(noisy_5, timestamp_s=t, bbox=bbox)
        m_raw_106 = FaceAligner.estimate_affine_matrix(sm_5_for_106, crop_size=512, landmarks_106=noisy_106)
        m_smooth_106 = smoother_106.smooth_similarity_matrix(m_raw_106, timestamp_s=t)
        tx_106_param_list.append(float(m_smooth_106[0, 2]))

    jitter_5 = float(np.std(tx_5_list[5:]))
    jitter_106_param = float(np.std(tx_106_param_list[5:]))
    assert jitter_106_param < jitter_5 * 0.65


def test_non_causal_two_pass_smoothing_zero_phase_and_scene_cut() -> None:
    """Garante que smooth_sequence_non_causal suavize sem atraso de fase (zero-phase) e respeite cortes de cena."""
    from src.core.types import DetectedFace

    rng = np.random.default_rng(77)
    base = _base_landmarks()
    bbox = np.array([80.0, 80.0, 180.0, 200.0], dtype=np.float32)

    # Cria uma trajetória senoidal com pico exatamente no quadro 15 + ruído
    seq: list[DetectedFace | None] = []
    true_x: list[float] = []
    for i in range(31):
        shift_x = float(20.0 * np.sin(np.pi * i / 30.0))
        true_x.append(float(base[0, 0] + shift_x))
        noise = rng.normal(0.0, 2.0, size=base.shape).astype(np.float32)
        lmk_i = base.copy()
        lmk_i[:, 0] += shift_x
        seq.append(DetectedFace(bbox=bbox + np.array([shift_x, 0, shift_x, 0], np.float32), landmarks=lmk_i + noise, score=0.9))

    # Adiciona um corte de cena abrupto no quadro 31 (+400px)
    cut_lmk = base + 400.0
    seq.append(DetectedFace(bbox=bbox + 400.0, landmarks=cut_lmk, score=0.9))

    smoothed = TemporalSmoother.smooth_sequence_non_causal(seq, half_window=4)
    assert len(smoothed) == len(seq)

    # O pico da trajetória suavizada deve continuar centrado em torno do quadro 15 (zero-phase, sem atraso)
    sm_x = [float(f.landmarks[0, 0]) for f in smoothed[:31] if f is not None and f.landmarks is not None]
    peak_idx = int(np.argmax(sm_x))
    assert 13 <= peak_idx <= 17

    # E o quadro 31 após o corte de cena não deve ter sido puxado pela janela anterior
    assert smoothed[31] is not None and smoothed[31].landmarks is not None
    assert np.allclose(smoothed[31].landmarks, cut_lmk, atol=1e-3)


