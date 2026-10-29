"""Testes unitários do pré-processamento e extração de máscara do FaceParser."""
from __future__ import annotations

import numpy as np
import pytest

from src.face.face_parser import FaceParser


def test_bisenet_preprocess_imagenet_normalization() -> None:
    """Garante que FaceParser.preprocess subtraia corretamente a média ImageNet na escala [0, 1]."""
    # Cria uma imagem BGR cujo RGB equivalente em [0, 1] é exatamente a média ImageNet (0.485, 0.456, 0.406)
    r = int(round(0.485 * 255))
    g = int(round(0.456 * 255))
    b = int(round(0.406 * 255))
    img_bgr = np.empty((256, 256, 3), dtype=np.uint8)
    img_bgr[..., 0] = b
    img_bgr[..., 1] = g
    img_bgr[..., 2] = r

    blob = FaceParser.preprocess(img_bgr, size=(512, 512))
    assert blob.shape == (1, 3, 512, 512)
    assert blob.dtype == np.float32
    # Como o pixel equivale à média ImageNet, o tensor normalizado deve ficar próximo de zero (|x| < 0.02)
    assert float(np.abs(blob).max()) < 0.02


def test_bisenet_preprocess_black_and_white_range() -> None:
    """Verifica os limites inferior (preto) e superior (branco) após normalização ImageNet."""
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)

    b_blob = FaceParser.preprocess(black, size=(512, 512))
    w_blob = FaceParser.preprocess(white, size=(512, 512))

    # Canal R (índice 0): (0 - 0.485)/0.229 ≈ -2.1179; (1 - 0.485)/0.229 ≈ +2.2489
    assert float(b_blob[0, 0, 0, 0]) == pytest.approx(-0.485 / 0.229, rel=1e-3)
    assert float(w_blob[0, 0, 0, 0]) == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-3)
