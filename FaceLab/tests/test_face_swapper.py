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


@pytest.mark.parametrize("fake_size", [128, 256, 512])
def test_face_compositor_supports_multi_resolution_crops(fake_size: int) -> None:
    """Garante que o compositor escale a matriz afim para crops de 128, 256 e 512."""
    compositor = FaceCompositor(blur_amount=15, erode_amount=3)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    landmarks = np.array(
        [[110.0, 120.0], [190.0, 120.0], [150.0, 155.0], [120.0, 195.0], [180.0, 195.0]],
        dtype=np.float32,
    )
    aligned_128, m_128 = FaceAligner.align_crop(frame, landmarks, crop_size=128)
    fake_crop = np.full((fake_size, fake_size, 3), 240, dtype=np.uint8)

    result = compositor.composite(
        original_frame=frame,
        fake_crop=fake_crop,
        aligned_original_crop=aligned_128,
        affine_matrix=m_128,
        color_correct=False,
    )
    assert result.shape == frame.shape
    # O centro do rosto (150, 155) deve ter recebido o valor alto do fake_crop em todas as resoluções
    assert result[155, 150, 0] > 200
    # As bordas distantes do quadro (10, 10) devem permanecer intactas (0)
    assert result[10, 10, 0] == 0


MODEL_PATH = MODELS_DIR / "inswapper_128.onnx"


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Modelo inswapper_128.onnx não presente")
def test_inswapper_initialization() -> None:
    dm = DeviceManager(DevicePreference.AUTO)
    swapper = INSwapperModel(dm, model_path=MODEL_PATH)
    assert swapper.session is not None
    assert "CUDAExecutionProvider" in swapper.active_providers or "CPUExecutionProvider" in swapper.active_providers


def test_match_color_hist_skin_mask_and_clamped_gain() -> None:
    """Garante que match_color_hist calcule estatísticas apenas na máscara de pele e limite o ganho entre 0.8 e 1.25."""
    compositor = FaceCompositor()

    # Source tem contraste baixo (desvio ~8); Target tem pele clara na metade inferior, mas barba/cabelo preto (0) na metade superior
    rng = np.random.default_rng(7)
    src = np.clip(120 + rng.normal(0.0, 8.0, size=(128, 128, 3)), 0, 255).astype(np.uint8)
    tgt = np.zeros((128, 128, 3), dtype=np.uint8)
    # Metade inferior do target = pele clara (valor médio ~200, alto contraste std ~45)
    tgt[64:, :, :] = np.clip(200 + rng.normal(0.0, 45.0, size=(64, 128, 3)), 0, 255).astype(np.uint8)

    # Máscara de pele indica apenas a metade inferior (ignora a metade superior preta)
    skin_mask = np.zeros((128, 128), dtype=np.float32)
    skin_mask[70:120, 20:108] = 1.0

    # Sem máscara de pele (elipse pega metade preta + metade clara -> escurece o resultado)
    out_no_mask = compositor.match_color_hist(src, tgt, skin_mask=None)
    # Com máscara de pele restrita à metade inferior -> resultado é claramente mais iluminado
    out_skin_mask = compositor.match_color_hist(
        src, tgt, skin_mask=skin_mask, std_gain_min=0.80, std_gain_max=1.25, enable_border_lowfreq=False
    )

    assert float(out_skin_mask[90, 64].mean()) > float(out_no_mask[90, 64].mean()) + 20.0

    # Verifica que o ganho de desvio padrão na pele foi limitado a <= 1.25 (não explodiu 5.6x de std 8 para std 45)
    src_lab_l = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)[70:120, 20:108, 0].astype(np.float32)
    out_lab_l = cv2.cvtColor(out_skin_mask, cv2.COLOR_BGR2LAB)[70:120, 20:108, 0].astype(np.float32)
    ratio_std = float(out_lab_l.std() / (src_lab_l.std() + 1e-6))
    assert 0.80 <= ratio_std <= 1.35

    # E garante que o modo opcional enable_border_lowfreq roda suavizando a faixa de transição na borda
    out_border = compositor.match_color_hist(
        src, tgt, skin_mask=skin_mask, std_gain_min=0.80, std_gain_max=1.25, enable_border_lowfreq=True
    )
    assert out_border.shape == src.shape
    assert float(out_border[72, 64].mean()) < float(out_skin_mask[95, 64].mean())


@pytest.mark.parametrize("blend_mode", ["alpha", "lowfreq_border", "laplacian"])
def test_compositor_blend_modes_and_timing(blend_mode: str) -> None:
    """Garante que os modos de blending ('alpha', 'lowfreq_border', 'laplacian') funcionem e meçam custo em ms."""
    compositor = FaceCompositor(blur_amount=15, erode_amount=3, blend_mode=blend_mode, pyramid_levels=3)
    # Tamanhos ímpares para garantir que pyrDown/pyrUp da pirâmide laplaciana lidem com dimensões não potências de 2
    frame = np.full((315, 333, 3), 80, dtype=np.uint8)
    landmarks = np.array(
        [[115.0, 125.0], [195.0, 125.0], [155.0, 160.0], [125.0, 200.0], [185.0, 200.0]],
        dtype=np.float32,
    )
    aligned_512, m_512 = FaceAligner.align_crop(frame, landmarks, crop_size=512)
    fake_crop = np.full((512, 512, 3), 210, dtype=np.uint8)

    result = compositor.composite(
        original_frame=frame,
        fake_crop=fake_crop,
        aligned_original_crop=aligned_512,
        affine_matrix=m_512,
        color_correct=False,
        blend_mode=blend_mode,
    )
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert result[160, 155, 0] > 180
    assert result[10, 10, 0] == 80
    assert compositor.last_blend_ms >= 0.0


@pytest.mark.parametrize("interp_name,expected_flag", [("cubic", cv2.INTER_CUBIC), ("lanczos4", cv2.INTER_LANCZOS4)])
def test_warp_back_interpolation_and_antialiasing_prefilter(interp_name: str, expected_flag: int) -> None:
    """Garante que o warp de volta use INTER_CUBIC/INTER_LANCZOS4 e filtre aliasing quando o rosto no frame é menor que o crop."""
    assert FaceCompositor.resolve_warp_flag(interp_name) == expected_flag

    # Cria padrão xadrez de altíssima frequência (1px alternado 0 e 255) no crop 512x512
    yy, xx = np.indices((512, 512))
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    fake_crop = np.stack([checker, checker, checker], axis=-1)

    # Rosto pequeno no frame (~80x80 px, logo downsample_ratio ≈ 512 / 80 = 6.4x)
    scaled_m_downscale = np.array([[6.4, 0.0, 0.0], [0.0, 6.4, 0.0]], dtype=np.float32)
    filtered = FaceCompositor.prefilter_for_downscale(fake_crop, scaled_m_downscale)
    # O pré-filtro anti-aliasing deve atenuar drasticamente a energia de Nyquist do xadrez 1px
    assert float(filtered.std()) < float(fake_crop.std()) * 0.25

    # Quando o rosto no frame é maior ou igual ao crop (downsample_ratio <= 1.15), não deve borrar nada
    scaled_m_upscale = np.array([[0.8, 0.0, 0.0], [0.0, 0.8, 0.0]], dtype=np.float32)
    untouched = FaceCompositor.prefilter_for_downscale(fake_crop, scaled_m_upscale)
    assert np.array_equal(untouched, fake_crop)

    # Verifica integração ponta a ponta no composite com warp_interpolation configurada
    comp = FaceCompositor(warp_interpolation=interp_name, prefilter_shrink=True, blend_mode="alpha")
    frame = np.full((240, 240, 3), 100, dtype=np.uint8)
    small_lmk = np.array(
        [[105.0, 105.0], [135.0, 105.0], [120.0, 120.0], [110.0, 135.0], [130.0, 135.0]],
        dtype=np.float32,
    )
    aligned_512, m_512 = FaceAligner.align_crop(frame, small_lmk, crop_size=512)
    out = comp.composite(frame, fake_crop, aligned_512, m_512, color_correct=False)
    assert out.shape == frame.shape


def test_swapper_model_specs_and_fallback_to_inswapper_128() -> None:
    """Garante que SWAPPER_MODEL_SPECS suporte inswapper_128 e hyperswap_256 (com fallback quando ausente)."""
    from src.face.face_swapper import SWAPPER_MODEL_SPECS

    spec_128 = SWAPPER_MODEL_SPECS["inswapper_128"]
    spec_256 = SWAPPER_MODEL_SPECS["hyperswap_256"]
    assert spec_128.input_size == (128, 128)
    assert spec_256.input_size == (256, 256)
    assert spec_256.norm_mean == (0.5, 0.5, 0.5)
    assert spec_256.norm_std == (0.5, 0.5, 0.5)

    # Verifica resolução de spec com fallback automático para inswapper_128 se hyperswap_1a_256.onnx não estiver baixado
    resolved_spec, resolved_path = INSwapperModel._resolve_spec_and_path("hyperswap_256", None)
    if not (MODELS_DIR / spec_256.filename).is_file():
        assert resolved_spec.name == "inswapper_128"
        assert resolved_path.name == "inswapper_128.onnx"

    # Testa preprocess_crop e postprocess_output na especificação de 256px ([-1, 1])
    dummy_swapper = INSwapperModel.__new__(INSwapperModel)
    dummy_swapper.spec = spec_256
    dummy_swapper.input_size = spec_256.input_size
    dummy_swapper.emap = None

    crop_512 = np.full((512, 512, 3), 255, dtype=np.uint8)
    blob_256 = dummy_swapper.preprocess_crop(crop_512)
    assert blob_256.shape == (1, 3, 256, 256)
    # Branco (255 -> 1.0) com mean=0.5, std=0.5 deve resultar em +1.0
    assert np.allclose(blob_256, 1.0, atol=1e-4)

    restored_bgr = dummy_swapper.postprocess_output(blob_256)
    assert restored_bgr.shape == (256, 256, 3)
    assert np.allclose(restored_bgr, 255)


def test_constant_caches_latent_emap_ellipse_and_kernels() -> None:
    """Garante que latent @ emap, máscara elíptica base, skin ROI e kernels morfológicos sejam cacheados."""
    from src.face.face_swapper import SWAPPER_MODEL_SPECS

    # 1. Cache de latent @ emap
    dummy_swapper = INSwapperModel.__new__(INSwapperModel)
    dummy_swapper.spec = SWAPPER_MODEL_SPECS["inswapper_128"]
    dummy_swapper.emap = np.eye(512, dtype=np.float32) * 2.0
    dummy_swapper._latent_cache_key = None
    dummy_swapper._latent_cache_val = None

    emb = np.ones(512, dtype=np.float32) / np.sqrt(512.0)
    lat1 = dummy_swapper.prepare_latent(emb)
    lat2 = dummy_swapper.prepare_latent(emb)
    assert lat1 is lat2  # Mesma instância retornada do cache

    # 2. Cache de elipse base, skin ROI e kernel morfológico
    e1 = FaceCompositor.get_cached_ellipse(256, 256)
    e2 = FaceCompositor.get_cached_ellipse(256, 256)
    assert e1 is e2

    s1 = FaceCompositor.get_cached_skin_roi(256, 256)
    s2 = FaceCompositor.get_cached_skin_roi(256, 256)
    assert s1 is s2

    k1 = FaceCompositor.get_cached_kernel(cv2.MORPH_ELLIPSE, 9)
    k2 = FaceCompositor.get_cached_kernel(cv2.MORPH_ELLIPSE, 9)
    assert k1 is k2






