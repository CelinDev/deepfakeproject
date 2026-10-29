"""Módulo de Alinhamento Facial (Face Aligner).

Responsável por:
1. Calcular a transformação afim (Similarity Transform: rotação, escala, translação)
   a partir dos 5 marcos faciais (olhos, nariz, cantos da boca) para alinhar o rosto
   em uma resolução canônica padrão (ex: 128x128).
2. Extrair o recorte facial normalizado e a matriz de transformação afim M (2x3).
3. Calcular a matriz inversa IM para permitir projetar o rosto gerado de volta
   às coordenadas originais do quadro.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage import transform as trans


# 5 pontos de referência canônicos no espaço 112x112 (ArcFace template)
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # olho esquerdo
        [73.5318, 51.5014],  # olho direito
        [56.0252, 71.7366],  # nariz
        [41.5493, 92.3655],  # canto esquerdo da boca
        [70.7299, 92.2041],  # canto direito da boca
    ],
    dtype=np.float32,
)


class FaceAligner:
    """Calcula matrizes afins e recortes alinhados de rostos."""

    @staticmethod
    def extract_5_from_106(landmarks_106: np.ndarray, fallback_5: np.ndarray | None = None) -> np.ndarray:
        """Extrai 5 pontos canônicos ultra-estáveis a partir da malha densa de 106 marcos do InsightFace.

        Em vez de depender de um único ponto ruidoso por região, calcula o centroide do contorno
        ocular esquerdo (33..42), contorno ocular direito (87..96), ponta/base do nariz (80, 85, 86)
        e cantos labiais (52/65 e 61/69). Se ``fallback_5`` for fornecido, funde 70% da estabilidade
        geométrica do contorno 106p com 30% da calibração ArcFace do detector SCRFD.
        """
        lmk = np.asarray(landmarks_106, dtype=np.float32)
        if lmk.shape != (106, 2):
            if fallback_5 is not None:
                return np.asarray(fallback_5, dtype=np.float32)
            raise ValueError(f"Esperado shape (106, 2), recebido {lmk.shape}")

        left_eye = lmk[33:43].mean(axis=0)
        right_eye = lmk[87:97].mean(axis=0)
        nose = lmk[[80, 85, 86]].mean(axis=0)
        mouth_left = lmk[[52, 64, 65]].mean(axis=0)
        mouth_right = lmk[[61, 68, 69]].mean(axis=0)
        pts5 = np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)

        if fallback_5 is not None and fallback_5.shape == (5, 2):
            pts5 = 0.70 * pts5 + 0.30 * np.asarray(fallback_5, dtype=np.float32)
        return pts5

    @staticmethod
    def decompose_similarity_matrix(matrix_m: np.ndarray) -> np.ndarray:
        """Decompõe uma matriz de similaridade 2x3 em ``[scale, cos_theta, sin_theta, tx, ty]``."""
        a = float(matrix_m[0, 0])
        b = float(matrix_m[1, 0])
        tx = float(matrix_m[0, 2])
        ty = float(matrix_m[1, 2])
        scale = float(np.hypot(a, b))
        if scale < 1e-8:
            return np.array([1.0, 1.0, 0.0, tx, ty], dtype=np.float32)
        return np.array([scale, a / scale, b / scale, tx, ty], dtype=np.float32)

    @staticmethod
    def compose_similarity_matrix(params: np.ndarray) -> np.ndarray:
        """Reconstrói a matriz afim de similaridade 2x3 a partir de ``[scale, cos_theta, sin_theta, tx, ty]``."""
        scale = float(params[0])
        cos_t = float(params[1])
        sin_t = float(params[2])
        norm = float(np.hypot(cos_t, sin_t))
        if norm > 1e-8:
            cos_t /= norm
            sin_t /= norm
        else:
            cos_t, sin_t = 1.0, 0.0
        a = scale * cos_t
        b = scale * sin_t
        tx = float(params[3])
        ty = float(params[4])
        return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)

    @classmethod
    def estimate_affine_matrix(
        cls,
        landmarks: np.ndarray,
        crop_size: int = 128,
        landmarks_106: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calcula a matriz afim 2x3 que alinha os landmarks para o crop canônico."""
        if landmarks_106 is not None and landmarks_106.shape == (106, 2):
            pts = cls.extract_5_from_106(landmarks_106, fallback_5=landmarks)
        else:
            pts = np.asarray(landmarks, dtype=np.float32)
        assert pts.shape == (5, 2), "Landmarks devem ter formato (5, 2)"

        ratio = float(crop_size) / 128.0 if crop_size % 128 == 0 else float(crop_size) / 112.0
        diff_x = 8.0 * ratio if crop_size % 128 == 0 else 0.0

        dst = ARCFACE_DST * ratio
        dst[:, 0] += diff_x

        # Usa SimilarityTransform.from_estimate do scikit-image sem deprecation warnings
        tform = trans.SimilarityTransform.from_estimate(pts, dst)
        return tform.params[0:2, :].astype(np.float32)

    @classmethod
    def align_crop(
        cls,
        image: np.ndarray,
        landmarks: np.ndarray,
        crop_size: int = 128,
        landmarks_106: np.ndarray | None = None,
        affine_matrix: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recorta e alinha o rosto na resolução crop_size x crop_size.

        Returns:
            aligned_face: imagem recortada (crop_size, crop_size, 3)
            matrix_m: matriz afim 2x3 usada para o recorte
        """
        matrix_m = (
            affine_matrix.astype(np.float32)
            if affine_matrix is not None
            else cls.estimate_affine_matrix(landmarks, crop_size, landmarks_106=landmarks_106)
        )
        aligned = cv2.warpAffine(
            image, matrix_m, (crop_size, crop_size), borderValue=0.0, flags=cv2.INTER_LINEAR
        )
        return aligned, matrix_m

