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
    def estimate_affine_matrix(landmarks: np.ndarray, crop_size: int = 128) -> np.ndarray:
        """Calcula a matriz afim 2x3 que alinha os landmarks para o crop canônico."""
        assert landmarks.shape == (5, 2), "Landmarks devem ter formato (5, 2)"
        
        ratio = float(crop_size) / 128.0 if crop_size % 128 == 0 else float(crop_size) / 112.0
        diff_x = 8.0 * ratio if crop_size % 128 == 0 else 0.0

        dst = ARCFACE_DST * ratio
        dst[:, 0] += diff_x

        # Usa SimilarityTransform do scikit-image sem deprecation warnings
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, dst)
        return tform.params[0:2, :].astype(np.float32)

    @classmethod
    def align_crop(cls, image: np.ndarray, landmarks: np.ndarray, crop_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Recorta e alinha o rosto na resolução crop_size x crop_size.

        Returns:
            aligned_face: imagem recortada (crop_size, crop_size, 3)
            matrix_m: matriz afim 2x3 usada para o recorte
        """
        matrix_m = cls.estimate_affine_matrix(landmarks, crop_size)
        aligned = cv2.warpAffine(
            image, matrix_m, (crop_size, crop_size), borderValue=0.0, flags=cv2.INTER_LINEAR
        )
        return aligned, matrix_m

