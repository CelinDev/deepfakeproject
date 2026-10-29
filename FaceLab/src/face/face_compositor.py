"""Compositor Facial (Face Compositor).

Responsável por integrar o rosto sintetizado de volta ao quadro original de forma natural:
1. Projeta o rosto sintetizado (crop 128x128) de volta ao quadro usando a matriz afim inversa.
2. Gera máscara suave (feathering / Gaussian blur + erosão morfológica) para evitar bordas duras.
3. Permite transferência de cor/iluminação para evitar diferenças abruptas de tom de pele.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from config.settings import (
    ALIGNED_MASK_SIZE,
    COLOR_STD_GAIN_MAX,
    COLOR_STD_GAIN_MIN,
    COMPOSITOR_BLEND_MODE,
    COMPOSITOR_DEFAULT_BLUR,
    COMPOSITOR_DEFAULT_ERODE,
    DOWNSCALE_PREFILTER_RATIO_THRESH,
    ENABLE_BORDER_LOWFREQ_COLOR,
    LAPLACIAN_PYRAMID_LEVELS,
    MICROTEXTURE_STRENGTH,
    WARP_BACK_INTERPOLATION,
    WARP_PREFILTER_SHRINK,
)
from src.face.temporal_smoother import TemporalSmoother


class FaceCompositor:
    """Combina o rosto sintetizado com o quadro original."""

    _ellipse_cache: dict[tuple[int, int], np.ndarray] = {}
    _skin_roi_cache: dict[tuple[int, int], np.ndarray] = {}
    _kernel_cache: dict[tuple[int, int], np.ndarray] = {}

    def __init__(
        self,
        blur_amount: int = COMPOSITOR_DEFAULT_BLUR,
        erode_amount: int = COMPOSITOR_DEFAULT_ERODE,
        mask_size: int = ALIGNED_MASK_SIZE,
        blend_mode: str = COMPOSITOR_BLEND_MODE,
        pyramid_levels: int = LAPLACIAN_PYRAMID_LEVELS,
        warp_interpolation: str = WARP_BACK_INTERPOLATION,
        prefilter_shrink: bool = WARP_PREFILTER_SHRINK,
    ) -> None:
        self.blur_amount = blur_amount if blur_amount % 2 != 0 else blur_amount + 1
        self.erode_amount = erode_amount
        self.mask_size = int(mask_size)
        self.blend_mode = str(blend_mode).lower()
        self.pyramid_levels = max(2, int(pyramid_levels))
        self.warp_interpolation = str(warp_interpolation).lower()
        self.prefilter_shrink = bool(prefilter_shrink)
        self.last_blend_ms: float = 0.0

    @staticmethod
    def resolve_warp_flag(interpolation: str) -> int:
        """Converte o nome da interpolação ('cubic', 'lanczos4', 'linear') para a flag do OpenCV."""
        interp = interpolation.strip().lower()
        if interp == "lanczos4":
            return cv2.INTER_LANCZOS4
        if interp == "linear":
            return cv2.INTER_LINEAR
        return cv2.INTER_CUBIC

    @staticmethod
    def prefilter_for_downscale(fake_crop: np.ndarray, scaled_affine: np.ndarray) -> np.ndarray:
        """Aplica pré-filtro gaussiano proporcional (anti-aliasing) quando o rosto no frame é menor que o crop.

        ``scaled_affine`` mapeia ``frame -> fake_crop``; logo a norma da primeira linha mede o fator
        de ampliação ``crop_px / frame_px``. Quando ``crop_px / frame_px > 1.15`` (isto é, o warp de
        volta reduz a resolução da imagem), filtramos frequências acima do limite de Nyquist do destino
        para evitar serrilhado e cintilação (shimmering) nos poros e bordas.
        """
        downsample_ratio = float(np.hypot(scaled_affine[0, 0], scaled_affine[0, 1]))
        if downsample_ratio <= DOWNSCALE_PREFILTER_RATIO_THRESH:
            return fake_crop
        sigma = min(4.0, 0.45 * (downsample_ratio - 1.0))
        return cv2.GaussianBlur(fake_crop, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @classmethod
    def get_cached_ellipse(cls, height: int, width: int) -> np.ndarray:
        """Retorna máscara elíptica pré-calculada e cacheada para a resolução solicitada."""
        key = (int(height), int(width))
        cached = cls._ellipse_cache.get(key)
        if cached is not None:
            return cached

        ellipse = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, int(height * 0.50))
        axes = (int(width * 0.44), int(height * 0.48))
        cv2.ellipse(ellipse, center, axes, 0, 0, 360, 1.0, -1)
        cls._ellipse_cache[key] = ellipse
        return ellipse

    @classmethod
    def get_cached_kernel(cls, shape: int, ksize: int) -> np.ndarray:
        """Retorna elemento estruturante morfológico pré-calculado e cacheado."""
        key = (int(shape), int(ksize))
        cached = cls._kernel_cache.get(key)
        if cached is not None:
            return cached
        kernel = cv2.getStructuringElement(shape, (ksize, ksize))
        cls._kernel_cache[key] = kernel
        return kernel

    @classmethod
    def get_cached_skin_roi(cls, height: int, width: int) -> np.ndarray:
        """Retorna máscara booleana elíptica da zona central da pele cacheada."""
        key = (int(height), int(width))
        cached = cls._skin_roi_cache.get(key)
        if cached is not None:
            return cached
        skin_roi = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            skin_roi,
            (width // 2, int(height * 0.52)),
            (int(width * 0.32), int(height * 0.36)),
            0,
            0,
            360,
            255,
            -1,
        )
        roi_bool = skin_roi > 0
        cls._skin_roi_cache[key] = roi_bool
        return roi_bool

    def build_aligned_mask(
        self,
        crop_h: int,
        crop_w: int,
        orig_crop_w: int,
        parser_mask: np.ndarray | None = None,
        occluder_mask: np.ndarray | None = None,
        smoother: TemporalSmoother | None = None,
    ) -> np.ndarray:
        """Gera a máscara no espaço do crop alinhado (tamanho fixo ``self.mask_size``, ex. 256),
        combinando probabilidades softmax do parser, máscara opcional de oclusão (XSeg),
        elipse cacheada e EMA temporal.
        """
        ms = self.mask_size
        base_ellipse = self.get_cached_ellipse(ms, ms).copy()

        if parser_mask is not None:
            pm = (
                parser_mask
                if parser_mask.shape[:2] == (ms, ms)
                else cv2.resize(parser_mask.astype(np.float32), (ms, ms), interpolation=cv2.INTER_LINEAR)
            )
            mask_ms = base_ellipse * np.clip(pm, 0.0, 1.0)
        else:
            mask_ms = base_ellipse

        if occluder_mask is not None:
            om = (
                occluder_mask
                if occluder_mask.shape[:2] == (ms, ms)
                else cv2.resize(occluder_mask.astype(np.float32), (ms, ms), interpolation=cv2.INTER_LINEAR)
            )
            mask_ms = mask_ms * np.clip(om, 0.0, 1.0)

        if smoother is not None:
            mask_ms = smoother.smooth_mask(mask_ms)

        # Escala erode_amount e blur_amount para o espaço mask_size (referência base 128px)
        mask_scale = max(1.0, float(ms) / 128.0)
        scaled_erode = int(round(self.erode_amount * mask_scale))
        if scaled_erode > 0:
            erode_k = scaled_erode if scaled_erode % 2 != 0 else scaled_erode + 1
            kernel = self.get_cached_kernel(cv2.MORPH_ELLIPSE, erode_k)
            mask_ms = cv2.erode(mask_ms, kernel, iterations=1)

        scaled_blur = max(3, int(round(self.blur_amount * mask_scale)))
        blur_k = scaled_blur if scaled_blur % 2 != 0 else scaled_blur + 1
        mask_ms = cv2.GaussianBlur(mask_ms, (blur_k, blur_k), 0)

        # Supressão leve de cauda/halo (Anti-Aura)
        mask_ms = np.clip((mask_ms - 0.03) / (0.97 - 0.03), 0.0, 1.0)
        fine_k = max(3, int(round(5 * mask_scale))) | 1
        mask_ms = cv2.GaussianBlur(mask_ms, (fine_k, fine_k), 0)

        if (crop_h, crop_w) != (ms, ms):
            return cv2.resize(mask_ms, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        return mask_ms

    def match_color_hist(
        self,
        source: np.ndarray,
        target: np.ndarray,
        smoother: TemporalSmoother | None = None,
        skin_mask: np.ndarray | None = None,
        std_gain_min: float = COLOR_STD_GAIN_MIN,
        std_gain_max: float = COLOR_STD_GAIN_MAX,
        enable_border_lowfreq: bool = ENABLE_BORDER_LOWFREQ_COLOR,
    ) -> np.ndarray:
        """Transferência perceptual de cor no espaço CIE LAB restrita apenas à máscara de pele.

        Calcula média e desvio padrão exclusivamente dentro da máscara de pele (ou elipse central
        caso a máscara não seja fornecida), aplica ganho limitado na razão de desvios padrão
        (padrão ``[0.80, 1.25]``) e suaviza temporalmente as estatísticas. Opcionalmente aplica
        correção de iluminação de baixa frequência por razão de blur gaussiano grande na faixa de borda.
        """
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        if target.shape[:2] != source.shape[:2]:
            target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        h, w = source.shape[:2]
        roi_bool: np.ndarray | None = None
        sm_resized: np.ndarray | None = None
        if skin_mask is not None:
            sm_resized = (
                skin_mask.astype(np.float32)
                if skin_mask.shape[:2] == (h, w)
                else cv2.resize(skin_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            )
            candidate_bool = sm_resized > 0.25
            if int(np.count_nonzero(candidate_bool)) >= 64:
                roi_bool = candidate_bool

        if roi_bool is None:
            roi_bool = self.get_cached_skin_roi(h, w)
            if sm_resized is None:
                sm_resized = roi_bool.astype(np.float32)

        s_pixels = src_lab[roi_bool]
        t_pixels = tgt_lab[roi_bool]

        s_mean = s_pixels.mean(axis=0)
        s_std = s_pixels.std(axis=0) + 1e-6
        t_mean = t_pixels.mean(axis=0)
        t_std = t_pixels.std(axis=0) + 1e-6

        if smoother is not None:
            s_mean, s_std, t_mean, t_std = smoother.smooth_color_stats(s_mean, s_std, t_mean, t_std)

        matched_lab = np.zeros_like(src_lab)
        for c in range(3):
            # No canal L (luminância), transfere 90% do contraste real; nos canais A/B (cor), 85% para naturalidade
            scale = 0.90 if c == 0 else 0.85
            raw_ratio = 1.0 + scale * ((t_std[c] / s_std[c]) - 1.0)
            std_ratio = float(np.clip(raw_ratio, std_gain_min, std_gain_max))
            channel = (src_lab[..., c] - s_mean[c]) * std_ratio + (scale * t_mean[c] + (1.0 - scale) * s_mean[c])
            matched_lab[..., c] = np.clip(channel, 0.0, 255.0)

        if enable_border_lowfreq and sm_resized is not None:
            # Faixa de transição na borda da máscara de pele (4 * m * (1 - m) atinge pico 1.0 em m = 0.5)
            sigma = max(7.0, float(min(h, w)) * 0.12)
            soft_m = cv2.GaussianBlur(np.clip(sm_resized, 0.0, 1.0), (0, 0), sigmaX=sigma * 0.5)
            border_weight = np.clip(4.0 * soft_m * (1.0 - soft_m), 0.0, 1.0)[..., np.newaxis]

            low_src = cv2.GaussianBlur(matched_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
            low_tgt = cv2.GaussianBlur(tgt_lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
            ratio = np.clip((low_tgt + 1.0) / (low_src + 1.0), std_gain_min, std_gain_max)
            corrected_border = np.clip(matched_lab * ratio, 0.0, 255.0)
            matched_lab = matched_lab * (1.0 - border_weight) + corrected_border * border_weight

        return cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def inject_skin_microtexture(
        fake_crop: np.ndarray,
        original_crop: np.ndarray,
        strength: float = MICROTEXTURE_STRENGTH,
    ) -> np.ndarray:
        """Transfere a microtextura de poros e iluminação fina do rosto real para o crop."""
        if original_crop.shape[:2] != fake_crop.shape[:2]:
            original_crop = cv2.resize(
                original_crop, (fake_crop.shape[1], fake_crop.shape[0]), interpolation=cv2.INTER_CUBIC
            )
        orig_f = original_crop.astype(np.float32)
        low_freq = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=1.6, sigmaY=1.6)
        high_freq = orig_f - low_freq  # Detalhe puro de poros, pelos faciais finos e micro-sombras
        detailed = fake_crop.astype(np.float32) + high_freq * float(strength)
        return np.clip(detailed, 0, 255).astype(np.uint8)

    @staticmethod
    def laplacian_pyramid_blend(
        fake_roi: np.ndarray,
        orig_roi: np.ndarray,
        alpha_mask: np.ndarray,
        levels: int = LAPLACIAN_PYRAMID_LEVELS,
    ) -> np.ndarray:
        """Multi-band blending via pirâmide laplaciana em poucos níveis (padrão 3).

        Funde frequências baixas (iluminação global/gradiente de pele) em uma faixa larga
        e frequências altas (poros, fios de cabelo, rugas) em uma faixa estreita, eliminando
        costuras visíveis nas bordas.
        """
        f_img = fake_roi.astype(np.float32)
        o_img = orig_roi.astype(np.float32)
        m_img = alpha_mask.astype(np.float32)
        if m_img.ndim == 2:
            m_img = m_img[..., np.newaxis]

        h, w = f_img.shape[:2]
        eff_levels = max(1, min(int(levels), int(np.floor(np.log2(max(2, min(h, w))))) - 1))
        if eff_levels <= 1:
            return np.clip(o_img * (1.0 - m_img) + f_img * m_img, 0.0, 255.0).astype(np.uint8)

        gp_f = [f_img]
        gp_o = [o_img]
        gp_m = [m_img]
        for _ in range(eff_levels - 1):
            gp_f.append(cv2.pyrDown(gp_f[-1]))
            gp_o.append(cv2.pyrDown(gp_o[-1]))
            down_m = cv2.pyrDown(gp_m[-1])
            if down_m.ndim == 2:
                down_m = down_m[..., np.newaxis]
            gp_m.append(down_m)

        lp_f: list[np.ndarray] = []
        lp_o: list[np.ndarray] = []
        for i in range(eff_levels - 1):
            size_i = (gp_f[i].shape[1], gp_f[i].shape[0])
            up_f = cv2.pyrUp(gp_f[i + 1], dstsize=size_i)
            up_o = cv2.pyrUp(gp_o[i + 1], dstsize=size_i)
            lp_f.append(gp_f[i] - up_f)
            lp_o.append(gp_o[i] - up_o)
        lp_f.append(gp_f[-1])
        lp_o.append(gp_o[-1])

        blended_pyr: list[np.ndarray] = []
        for lf, lo, gm in zip(lp_f, lp_o, gp_m):
            blended_pyr.append(lo * (1.0 - gm) + lf * gm)

        reconstructed = blended_pyr[-1]
        for i in range(eff_levels - 2, -1, -1):
            size_i = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
            reconstructed = cv2.pyrUp(reconstructed, dstsize=size_i) + blended_pyr[i]

        return np.clip(reconstructed, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def lowfreq_border_blend(
        fake_roi: np.ndarray,
        orig_roi: np.ndarray,
        alpha_mask: np.ndarray,
    ) -> np.ndarray:
        """Blending rápido com correção de baixa frequência na faixa de borda da máscara.

        Iguala o gradiente de iluminação local entre ``fake_roi`` e ``orig_roi`` onde
        ``0 < alpha < 1`` (peso máximo em ``alpha = 0.5``) usando um blur gaussiano amplo.
        """
        f_img = fake_roi.astype(np.float32)
        o_img = orig_roi.astype(np.float32)
        m_img = alpha_mask.astype(np.float32)
        if m_img.ndim == 2:
            m_img = m_img[..., np.newaxis]

        h, w = f_img.shape[:2]
        small_w, small_h = max(16, w // 4), max(16, h // 4)
        low_f_small = cv2.GaussianBlur(
            cv2.resize(f_img, (small_w, small_h), interpolation=cv2.INTER_AREA), (0, 0), sigmaX=4.0, sigmaY=4.0
        )
        low_o_small = cv2.GaussianBlur(
            cv2.resize(o_img, (small_w, small_h), interpolation=cv2.INTER_AREA), (0, 0), sigmaX=4.0, sigmaY=4.0
        )
        diff_low = cv2.resize(low_o_small - low_f_small, (w, h), interpolation=cv2.INTER_LINEAR)

        border_band = np.clip(4.0 * m_img * (1.0 - m_img), 0.0, 1.0)
        f_corrected = np.clip(f_img + diff_low * border_band, 0.0, 255.0)
        blended = o_img * (1.0 - m_img) + f_corrected * m_img
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def composite(
        self,
        original_frame: np.ndarray,
        fake_crop: np.ndarray,
        aligned_original_crop: np.ndarray,
        affine_matrix: np.ndarray,
        color_correct: bool = True,
        smoother: TemporalSmoother | None = None,
        blend_intensity: float = 1.0,
        parser_mask: np.ndarray | None = None,
        occluder_mask: np.ndarray | None = None,
        skin_mask: np.ndarray | None = None,
        enable_border_lowfreq: bool = ENABLE_BORDER_LOWFREQ_COLOR,
        blend_mode: str | None = None,
    ) -> np.ndarray:
        """Projeta o fake_crop sobre o original_frame com blending anatômico e bordas super suaves.

        Suporta recortes de 128x128, 256x256 e 512x512, escalando automaticamente a matriz
        afim pelo fator ``s = fake_crop.shape[1] / aligned_original_crop.shape[1]`` antes de inverter.
        """
        h, w = original_frame.shape[:2]
        crop_h, crop_w = fake_crop.shape[:2]
        orig_crop_h, orig_crop_w = aligned_original_crop.shape[:2]

        effective_skin_mask = skin_mask if skin_mask is not None else parser_mask
        if color_correct:
            fake_crop = self.match_color_hist(
                fake_crop,
                aligned_original_crop,
                smoother=smoother,
                skin_mask=effective_skin_mask,
                enable_border_lowfreq=enable_border_lowfreq,
            )
            fake_crop = self.inject_skin_microtexture(
                fake_crop, aligned_original_crop, strength=MICROTEXTURE_STRENGTH
            )

        # Fator de escala entre o fake_crop (ex: 512px do GFPGAN) e o aligned_original_crop (ex: 128px)
        scale_factor = float(crop_w) / float(orig_crop_w)
        m3 = np.vstack([affine_matrix.astype(np.float64), [0.0, 0.0, 1.0]])
        s_mat = np.diag([scale_factor, scale_factor, 1.0])
        scaled_m = (s_mat @ m3)[:2, :].astype(np.float32)

        # Matriz inversa (espaço do fake_crop -> espaço do quadro original)
        inv_affine = cv2.invertAffineTransform(scaled_m)

        # Constrói máscara no espaço do crop alinhado (elipse cacheada * parser softmax * occluder + EMA temporal)
        mask = self.build_aligned_mask(
            crop_h=crop_h,
            crop_w=crop_w,
            orig_crop_w=orig_crop_w,
            parser_mask=parser_mask,
            occluder_mask=occluder_mask,
            smoother=smoother,
        )

        # OTIMIZAÇÃO (Fase 6): calcula a Região de Interesse (ROI) onde o rosto se projeta no frame.
        # Em vez de aplicar warpAffine e mistura alfa na imagem inteira de 1920x1080 (que custa ~100ms),
        # processa apenas na ROI local em torno do rosto (~1ms), reduzindo o gargalo em 98%.
        corners = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32).reshape(-1, 1, 2)
        proj_corners = cv2.transform(corners, inv_affine).squeeze()
        x1 = max(0, int(np.floor(proj_corners[:, 0].min())))
        y1 = max(0, int(np.floor(proj_corners[:, 1].min())))
        x2 = min(w, int(np.ceil(proj_corners[:, 0].max())))
        y2 = min(h, int(np.ceil(proj_corners[:, 1].max())))

        if x2 <= x1 or y2 <= y1:
            return original_frame

        local_inv = inv_affine.copy()
        local_inv[0, 2] -= x1
        local_inv[1, 2] -= y1
        roi_w, roi_h = x2 - x1, y2 - y1

        # Pré-filtro anti-aliasing quando o rosto no frame é menor que o fake_crop (ex: rosto de 140px vs crop 512px)
        crop_to_warp = (
            self.prefilter_for_downscale(fake_crop, scaled_m) if self.prefilter_shrink else fake_crop
        )
        warp_flag = self.resolve_warp_flag(self.warp_interpolation)

        warped_fake = cv2.warpAffine(crop_to_warp, local_inv, (roi_w, roi_h), borderValue=0.0, flags=warp_flag)
        warped_mask = cv2.warpAffine(mask, local_inv, (roi_w, roi_h), borderValue=0.0, flags=cv2.INTER_LINEAR)

        alpha = np.clip(warped_mask[..., np.newaxis] * float(blend_intensity), 0.0, 1.0)
        roi_orig = original_frame[y1:y2, x1:x2]
        mode = (blend_mode or self.blend_mode).lower()

        t_blend0 = time.perf_counter()
        if mode == "laplacian":
            blended_roi = self.laplacian_pyramid_blend(
                warped_fake, roi_orig, alpha, levels=self.pyramid_levels
            )
        elif mode == "lowfreq_border":
            blended_roi = self.lowfreq_border_blend(warped_fake, roi_orig, alpha)
        else:
            roi_orig_f = roi_orig.astype(np.float32)
            blended_roi = np.clip(
                roi_orig_f * (1.0 - alpha) + warped_fake.astype(np.float32) * alpha, 0, 255
            ).astype(np.uint8)
        self.last_blend_ms = (time.perf_counter() - t_blend0) * 1000.0

        out = original_frame.copy()
        out[y1:y2, x1:x2] = blended_roi
        return out

