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


def test_parse_probabilities_and_aligned_mask_with_ema() -> None:
    """Garante que parse_probabilities some softmax de pele/olhos/boca, desconte oclusões e que o compositor aplique elipse cacheada + EMA."""
    from unittest.mock import MagicMock
    from src.face.face_compositor import FaceCompositor
    from src.face.temporal_smoother import TemporalSmoother

    parser = FaceParser.__new__(FaceParser)
    parser.input_name = "input"
    parser.output_name = "output"
    parser._cached_mask_roi = None
    parser._cached_aligned_prob = None
    parser._frame_counter = 0

    # Cria logits sintéticos (1, 19, 512, 512): metade superior = cabelo (17, oclusão), metade inferior = pele (1, foreground)
    logits = np.full((1, 19, 512, 512), -10.0, dtype=np.float32)
    logits[0, 17, :256, :] = 10.0  # Cabelo no topo -> probabilidade ~ 1.0 de oclusão
    logits[0, 1, 256:, :] = 10.0   # Pele na base -> probabilidade ~ 1.0 de face interna

    parser._session = MagicMock()
    parser._session.run.return_value = [logits]

    crop = np.zeros((512, 512, 3), dtype=np.uint8)
    prob_mask = parser.parse_probabilities(crop, out_size=256)
    assert prob_mask.shape == (256, 256)
    assert prob_mask.dtype == np.float32
    # Na metade superior (cabelo), a máscara deve ser ~0; na metade inferior (pele), ~1
    assert float(prob_mask[60, 128]) < 0.01
    assert float(prob_mask[190, 128]) > 0.99

    compositor = FaceCompositor(blur_amount=7, erode_amount=1, mask_size=256)
    smoother = TemporalSmoother(alpha_mask=0.5)
    m1 = compositor.build_aligned_mask(512, 512, 512, parser_mask=prob_mask, smoother=smoother)
    m2 = compositor.build_aligned_mask(512, 512, 512, parser_mask=np.zeros_like(prob_mask), smoother=smoother)
    assert m1.shape == (512, 512)
    # Verifica cache da elipse e inércia do EMA na máscara
    assert (256, 256) in FaceCompositor._ellipse_cache
    assert 0.05 < float(m2[380, 256]) < float(m1[380, 256])

