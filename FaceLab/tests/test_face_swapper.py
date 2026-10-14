"""Testes automatizados da Fase 4: Face Aligner, Compositor e Swapper."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from config.settings import MODELS_DIR
from src.core.types import DetectedFace
from src.face.face_aligner import ARCFACE_DST, FaceAligner
from src.face.face_compositor import FaceCompositor
from src.face.face_swapper import INSwapperModel
from src.inference.device_manager import DeviceManager, DevicePreference


def test_face_aligner_matrix_shape() -> None:
    # 5 pontos faciais dummy
    landmarks = np.array(
        [[30.0, 50.0], [70.0, 50.0], [50.0, 70.0], [40.0, 90.0], [60.0, 90.0]],
        dtype=np.float32,
    )
    m = FaceAligner.estimate_affine_matrix(landmarks, crop_size=128)
    assert m.shape == (2, 3)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop, m_used = FaceAligner.align_crop(img, landmarks, crop_size=128)
    assert crop.shape == (128, 128, 3)
    assert np.allclose(m, m_used)


def test_face_compositor_preserves_dimensions() -> None:
    compositor = FaceCompositor()
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    fake_crop = np.full((128, 128, 3), 200, dtype=np.uint8)
    aligned_crop = np.full((128, 128, 3), 120, dtype=np.uint8)

    # Matriz afim centrada
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    result = compositor.composite(
        original_frame=frame,
        fake_crop=fake_crop,
        aligned_original_crop=aligned_crop,
        affine_matrix=m,
        color_correct=True,
    )
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


MODEL_PATH = MODELS_DIR / "inswapper_128.onnx"


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Modelo inswapper_128.onnx não presente")
def test_inswapper_initialization() -> None:
    dm = DeviceManager(DevicePreference.AUTO)
    swapper = INSwapperModel(dm, model_path=MODEL_PATH)
    assert swapper.session is not None
    assert "CUDAExecutionProvider" in swapper.active_providers or "CPUExecutionProvider" in swapper.active_providers

