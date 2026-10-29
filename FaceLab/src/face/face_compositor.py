"""Compositor Facial (Face Compositor).

Responsável por integrar o rosto sintetizado de volta ao quadro original de forma natural:
1. Projeta o rosto sintetizado (crop 128x128) de volta ao quadro usando a matriz afim inversa.
2. Gera máscara suave (feathering / Gaussian blur + erosão morfológica) para evitar bordas duras.
3. Permite transferência de cor/iluminação para evitar diferenças abruptas de tom de pele.
"""
from __future__ import annotations

import cv2
import numpy as np

from config.settings import ALIGNED_MASK_SIZE
from src.face.temporal_smoother import TemporalSmoother


class FaceCompositor:
    """Combina o rosto sintetizado com o quadro original."""

    _ellipse_cache: dict[tuple[int, int], np.ndarray] = {}

    def __init__(
        self,
        blur_amount: int = 15,
        erode_amount: int = 5,
        mask_size: int = ALIGNED_MASK_SIZE,
    ) -> None:
        self.blur_amount = blur_amount if blur_amount % 2 != 0 else blur_amount + 1
        self.erode_amount = erode_amount
        self.mask_size = int(mask_size)

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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
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
    ) -> np.ndarray:
        """Transferência perceptual de cor no espaço CIE LAB restrita à zona central da pele.

        Evita que sombras de fundo ou cabelos escuros nas bordas do crop 128x128
        contaminem o tom de pele ou a iluminação do rosto trocado.
        """
        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        if target.shape[:2] != source.shape[:2]:
            target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        h, w = source.shape[:2]
        skin_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(skin_roi, (w // 2, int(h * 0.52)), (int(w * 0.32), int(h * 0.36)), 0, 0, 360, 255, -1)
        roi_bool = skin_roi > 0

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
            std_ratio = 1.0 + scale * ((t_std[c] / s_std[c]) - 1.0)
            channel = (src_lab[..., c] - s_mean[c]) * std_ratio + (scale * t_mean[c] + (1.0 - scale) * s_mean[c])
            matched_lab[..., c] = np.clip(channel, 0, 255)

        return cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def inject_skin_microtexture(fake_crop: np.ndarray, original_crop: np.ndarray, strength: float = 0.22) -> np.ndarray:
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
    ) -> np.ndarray:
        """Projeta o fake_crop sobre o original_frame com blending anatômico e bordas super suaves.

        Suporta recortes de 128x128, 256x256 e 512x512, escalando automaticamente a matriz
        afim pelo fator ``s = fake_crop.shape[1] / aligned_original_crop.shape[1]`` antes de inverter.
        """
        h, w = original_frame.shape[:2]
        crop_h, crop_w = fake_crop.shape[:2]
        orig_crop_h, orig_crop_w = aligned_original_crop.shape[:2]

        if color_correct:
            fake_crop = self.match_color_hist(fake_crop, aligned_original_crop, smoother=smoother)
            fake_crop = self.inject_skin_microtexture(fake_crop, aligned_original_crop, strength=0.22)

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

        warped_fake = cv2.warpAffine(fake_crop, local_inv, (roi_w, roi_h), borderValue=0.0, flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(mask, local_inv, (roi_w, roi_h), borderValue=0.0, flags=cv2.INTER_LINEAR)

        alpha = np.clip(warped_mask[..., np.newaxis] * float(blend_intensity), 0.0, 1.0)
        roi_orig = original_frame[y1:y2, x1:x2].astype(np.float32)
        out = original_frame.copy()
        out[y1:y2, x1:x2] = np.clip(roi_orig * (1.0 - alpha) + warped_fake.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        return out

