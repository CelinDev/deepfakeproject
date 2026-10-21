"""Módulo de Efeitos Especiais de Halloween — Transformador Zumbi Hiper-realista (Zombie Transformer).

Aplica maquiagem protética cinematográfica digital:
1. Palidez cadavérica com manchas de livor mortis e textura de pele necrosada.
2. Olheiras profundas e cavidades oculares necrosadas.
3. Olhos leitosos com catarata cadavérica e esclera vascularizada rompida.
4. Veias subcutâneas arroxeadas/negras ramificadas pelas têmporas e bochechas.
5. Feridas e lacerações tridimensionais (camadas de pele aberta, profundidade escura e specular highlights).
6. Lábios cianóticos rachados e sangue coagulado escorrido.

Otimizado em ROI facial com blending gaussiano para manter performance fluida em tempo real.
"""
from __future__ import annotations

import cv2
import numpy as np


class ZombieTransformer:
    """Aplica efeitos hiper-realistas de zumbi/Halloween sobre a face demarcada."""

    @staticmethod
    def _draw_vein_branch(
        canvas: np.ndarray,
        alpha_canvas: np.ndarray,
        start: tuple[int, int],
        branches: list[tuple[tuple[int, int], float]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Desenha uma rede de veias necróticas finas com ramificações irregulares."""
        curr = start
        for next_pt, alpha in branches:
            cv2.line(canvas, curr, next_pt, color, thickness, cv2.LINE_AA)
            cv2.line(alpha_canvas, curr, next_pt, alpha, thickness, cv2.LINE_AA)
            curr = next_pt

    @classmethod
    def transform(
        cls,
        frame: np.ndarray,
        landmarks: np.ndarray,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Aplica a transformação cinematográfica de zumbi sobre a face.

        Args:
            frame: Imagem de entrada em formato BGR (H, W, 3).
            landmarks: Marcos faciais (5 pontos: olho esq., olho dir., nariz, boca esq., boca dir.).
            intensity: Força do efeito de 0.0 (sem efeito) a 1.0 (zumbi total).

        Returns:
            Imagem processada com a textura hiper-realista aplicada.
        """
        if intensity <= 0.01 or landmarks is None or len(landmarks) < 5:
            return frame

        h, w = frame.shape[:2]
        eye_l, eye_r = landmarks[0], landmarks[1]
        nose = landmarks[2]
        mouth_l, mouth_r = landmarks[3], landmarks[4]

        face_center = ((eye_l + eye_r + mouth_l + mouth_r) / 4.0).astype(int)
        eye_dist = float(np.linalg.norm(eye_r - eye_l))
        if eye_dist < 6:
            return frame

        radius = int(eye_dist * 1.65)

        # Região de Interesse (ROI) para processamento rápido
        x1 = max(0, face_center[0] - radius)
        y1 = max(0, face_center[1] - radius)
        x2 = min(w, face_center[0] + radius)
        y2 = min(h, face_center[1] + radius)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2].copy()
        rh, rw = roi.shape[:2]

        # Coordenadas relativas à ROI
        rel_center = (face_center[0] - x1, face_center[1] - y1)
        rel_el = (int(eye_l[0] - x1), int(eye_l[1] - y1))
        rel_er = (int(eye_r[0] - x1), int(eye_r[1] - y1))
        rel_nose = (int(nose[0] - x1), int(nose[1] - y1))
        rel_ml = (int(mouth_l[0] - x1), int(mouth_l[1] - y1))
        rel_mr = (int(mouth_r[0] - x1), int(mouth_r[1] - y1))

        # -------------------------------------------------------------
        # 1. PELE CADAVÉRICA & LIVOR MORTIS (Decadência orgânica)
        # -------------------------------------------------------------
        mask_face = np.zeros((rh, rw), dtype=np.float32)
        cv2.circle(mask_face, rel_center, int(radius * 0.95), 1.0, -1)
        k_blur = max(3, int(radius * 0.45) * 2 + 1)
        mask_face = cv2.GaussianBlur(mask_face, (k_blur, k_blur), 0) * intensity

        # Desnaturação e tom esverdeado/cinza de decomposição
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        tint = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

        # Matiz cadavérica: suprime vermelhos, eleva azuis frígidos e tons oliva
        tint[..., 0] = np.clip(tint[..., 0] * 0.94 + 20, 0, 255)  # Azul cadavérico
        tint[..., 1] = np.clip(tint[..., 1] * 1.08 + 24, 0, 255)  # Verde decomposição
        tint[..., 2] = np.clip(tint[..., 2] * 0.70 - 20, 0, 255)  # Perda de circulação vital

        # Textura granulada orgânica de pele áspera/ressecada
        # Usa ruído determinístico de baixa amplitude para manter alta performance
        np.random.seed(42)
        noise = (np.random.randn(rh, rw) * 6.5).astype(np.float32)
        tint[..., 0] = np.clip(tint[..., 0] + noise, 0, 255)
        tint[..., 1] = np.clip(tint[..., 1] + noise, 0, 255)
        tint[..., 2] = np.clip(tint[..., 2] + noise, 0, 255)

        alpha_face = np.expand_dims(mask_face * 0.72, axis=-1)
        roi_mod = np.clip(
            roi.astype(np.float32) * (1.0 - alpha_face) + tint * alpha_face, 0, 255
        ).astype(np.uint8)

        # -------------------------------------------------------------
        # 2. OLHEIRAS NEGRAS PROFUNDAS (Necrotic Sunken Sockets)
        # -------------------------------------------------------------
        socket_r = int(eye_dist * 0.32)
        eye_mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.circle(eye_mask, rel_el, socket_r, 1.0, -1)
        cv2.circle(eye_mask, rel_er, socket_r, 1.0, -1)
        k_eye = max(3, socket_r * 2 + 1)
        eye_mask = cv2.GaussianBlur(eye_mask, (k_eye, k_eye), 0) * intensity

        roi_f = roi_mod.astype(np.float32)
        roi_f[..., 0] *= (1.0 - eye_mask * 0.55)  # B
        roi_f[..., 1] *= (1.0 - eye_mask * 0.78)  # G (escurece bastante)
        roi_f[..., 2] *= (1.0 - eye_mask * 0.65)  # R (deixa arroxeado profundo)
        roi_mod = np.clip(roi_f, 0, 255).astype(np.uint8)

        # -------------------------------------------------------------
        # 3. OLHOS LEITOSOS DE CATARATA CADAVÉRICA (Dead Milky Eyes)
        # -------------------------------------------------------------
        pupil_r = max(2, int(eye_dist * 0.11))
        iris_mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.circle(iris_mask, rel_el, pupil_r, 1.0, -1)
        cv2.circle(iris_mask, rel_er, pupil_r, 1.0, -1)
        k_iris = max(3, int(pupil_r * 0.8) * 2 + 1)
        iris_mask = cv2.GaussianBlur(iris_mask, (k_iris, k_iris), 0) * (0.85 * intensity)

        dead_eye_color = np.array([215, 225, 220], dtype=np.float32)  # Branco leitoso vítreo
        iris_3 = np.expand_dims(iris_mask, axis=-1)
        roi_mod = np.clip(
            roi_mod.astype(np.float32) * (1.0 - iris_3) + dead_eye_color * iris_3,
            0,
            255,
        ).astype(np.uint8)

        # Anel vascular sutil vermelho ao redor dos olhos
        sclera_ring = np.zeros((rh, rw), dtype=np.float32)
        cv2.circle(sclera_ring, rel_el, int(pupil_r * 1.4), 0.5, 2)
        cv2.circle(sclera_ring, rel_er, int(pupil_r * 1.4), 0.5, 2)
        sclera_ring = cv2.GaussianBlur(sclera_ring, (5, 5), 0) * intensity
        roi_f2 = roi_mod.astype(np.float32)
        roi_f2[..., 2] = np.clip(roi_f2[..., 2] + sclera_ring * 75, 0, 255)
        roi_mod = roi_f2.astype(np.uint8)

        # -------------------------------------------------------------
        # 4. VEIAS NECRÓTICAS SUBCUTÂNEAS (Subcutaneous Rotting Veins)
        # -------------------------------------------------------------
        vein_layer = np.zeros_like(roi_mod, dtype=np.uint8)
        vein_alpha = np.zeros((rh, rw), dtype=np.float32)
        vein_color = (40, 20, 50)  # Púrpura quase preto necrótico
        th_vein = max(1, int(eye_dist * 0.018))

        # Rede A: Têmpora esquerda descendo pela bochecha
        t_l = (rel_el[0] - int(eye_dist * 0.38), rel_el[1] - int(eye_dist * 0.15))
        branches_l = [
            ((t_l[0] - int(eye_dist * 0.15), t_l[1] + int(eye_dist * 0.18)), 0.65 * intensity),
            ((t_l[0] - int(eye_dist * 0.22), t_l[1] + int(eye_dist * 0.42)), 0.55 * intensity),
            ((t_l[0] - int(eye_dist * 0.10), t_l[1] + int(eye_dist * 0.65)), 0.45 * intensity),
        ]
        cls._draw_vein_branch(vein_layer, vein_alpha, t_l, branches_l, vein_color, th_vein)

        # Rede B: Têmpora direita subindo para a testa
        t_r = (rel_er[0] + int(eye_dist * 0.38), rel_er[1] - int(eye_dist * 0.15))
        branches_r = [
            ((t_r[0] + int(eye_dist * 0.12), t_r[1] - int(eye_dist * 0.22)), 0.60 * intensity),
            ((t_r[0] + int(eye_dist * 0.05), t_r[1] - int(eye_dist * 0.45)), 0.50 * intensity),
            ((t_r[0] - int(eye_dist * 0.08), t_r[1] - int(eye_dist * 0.60)), 0.40 * intensity),
        ]
        cls._draw_vein_branch(vein_layer, vein_alpha, t_r, branches_r, vein_color, th_vein)

        # Desfoque para manter as veias aparentando estar embaixo da epiderme
        vein_alpha = cv2.GaussianBlur(vein_alpha, (5, 5), 0)
        va_3 = np.expand_dims(vein_alpha, axis=-1)
        roi_mod = np.clip(
            roi_mod.astype(np.float32) * (1.0 - va_3) + vein_layer.astype(np.float32) * va_3,
            0,
            255,
        ).astype(np.uint8)

        # -------------------------------------------------------------
        # 5. LÁBIOS CIANÓTICOS / RESSECADOS (Cyanotic Lips)
        # -------------------------------------------------------------
        mouth_center = ((rel_ml[0] + rel_mr[0]) // 2, (rel_ml[1] + rel_mr[1]) // 2)
        mouth_w = int(np.linalg.norm(mouth_r - mouth_l))
        lip_mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(
            lip_mask,
            mouth_center,
            (int(mouth_w * 0.58), max(3, int(mouth_w * 0.22))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        lip_mask = cv2.GaussianBlur(lip_mask, (7, 7), 0) * (0.60 * intensity)
        lip_3 = np.expand_dims(lip_mask, axis=-1)
        lip_color = np.array([45, 25, 40], dtype=np.float32)  # Púrpura cianótico escuro
        roi_mod = np.clip(
            roi_mod.astype(np.float32) * (1.0 - lip_3) + lip_color * lip_3,
            0,
            255,
        ).astype(np.uint8)

        # -------------------------------------------------------------
        # 6. SANGUE COAGULADO, CORTES 3D & FERIDAS PROFUNDAS
        # -------------------------------------------------------------
        blood_layer = np.zeros_like(roi_mod, dtype=np.uint8)
        blood_alpha = np.zeros((rh, rw), dtype=np.float32)

        # A) Sangue escorrendo do canto esquerdo da boca
        pts_drip_left = np.array([
            [rel_ml[0], rel_ml[1]],
            [rel_ml[0] - int(mouth_w * 0.12), rel_ml[1] + int(mouth_w * 0.65)],
            [rel_ml[0] + int(mouth_w * 0.04), rel_ml[1] + int(mouth_w * 1.15)],
            [rel_ml[0] + int(mouth_w * 0.16), rel_ml[1] + int(mouth_w * 0.45)],
            [rel_ml[0] + int(mouth_w * 0.28), rel_ml[1]],
        ], dtype=np.int32)
        cv2.fillPoly(blood_layer, [pts_drip_left], (14, 8, 120))  # Sangue venoso escuro
        cv2.fillPoly(blood_alpha, [pts_drip_left], 0.90 * intensity)

        # Gota densa pendurada no queixo esquerdo
        cv2.circle(
            blood_layer,
            (rel_ml[0], rel_ml[1] + int(mouth_w * 1.25)),
            max(2, int(mouth_w * 0.08)),
            (8, 4, 90),
            -1,
        )
        cv2.circle(
            blood_alpha,
            (rel_ml[0], rel_ml[1] + int(mouth_w * 1.25)),
            max(2, int(mouth_w * 0.08)),
            0.94 * intensity,
            -1,
        )

        # B) Escorrido sutil do canto direito da boca
        pts_drip_right = np.array([
            [rel_mr[0], rel_mr[1]],
            [rel_mr[0] + int(mouth_w * 0.08), rel_mr[1] + int(mouth_w * 0.45)],
            [rel_mr[0] - int(mouth_w * 0.05), rel_mr[1] + int(mouth_w * 0.70)],
            [rel_mr[0] - int(mouth_w * 0.18), rel_mr[1]],
        ], dtype=np.int32)
        cv2.fillPoly(blood_layer, [pts_drip_right], (16, 10, 130))
        cv2.fillPoly(blood_alpha, [pts_drip_right], 0.85 * intensity)

        # C) Laceração 3D Profunda na Bochecha Direita (Gash com Especular)
        c_x = int(rel_er[0] + (rel_mr[0] - rel_er[0]) * 0.5 + eye_dist * 0.28)
        c_y = int(rel_er[1] + (rel_mr[1] - rel_er[1]) * 0.5)
        w_start = (c_x - int(eye_dist * 0.22), c_y - int(eye_dist * 0.32))
        w_end = (c_x + int(eye_dist * 0.22), c_y + int(eye_dist * 0.32))

        # 1. Borda externa inflamada (pele esfolada / carne viva avermelhada)
        thick_rim = max(4, int(eye_dist * 0.095))
        cv2.line(blood_layer, w_start, w_end, (20, 20, 175), thick_rim, cv2.LINE_AA)
        cv2.line(blood_alpha, w_start, w_end, 0.85 * intensity, thick_rim, cv2.LINE_AA)

        # 2. Fenda interna necrótica escura (sangue oxidado/coagulado no fundo do corte)
        thick_core = max(2, int(eye_dist * 0.038))
        cv2.line(blood_layer, w_start, w_end, (4, 2, 22), thick_core, cv2.LINE_AA)
        cv2.line(blood_alpha, w_start, w_end, 0.98 * intensity, thick_core, cv2.LINE_AA)

        # 3. Realce especular (highlight de borda úmida que cria a tridimensionalidade)
        h_offset = max(1, int(eye_dist * 0.03))
        w_h_start = (w_start[0] - h_offset, w_start[1] - h_offset)
        w_h_end = (w_end[0] - h_offset, w_end[1] - h_offset)
        cv2.line(blood_layer, w_h_start, w_h_end, (170, 180, 210), 1, cv2.LINE_AA)
        cv2.line(blood_alpha, w_h_start, w_h_end, 0.55 * intensity, 1, cv2.LINE_AA)

        # Escorrido derivado da ferida da bochecha
        wound_drip = (c_x + int(eye_dist * 0.15), c_y + int(eye_dist * 0.48))
        cv2.line(blood_layer, w_end, wound_drip, (18, 14, 125), max(1, int(eye_dist * 0.025)), cv2.LINE_AA)
        cv2.line(blood_alpha, w_end, wound_drip, 0.75 * intensity, max(1, int(eye_dist * 0.025)), cv2.LINE_AA)

        # D) Ferida Tridimensional na Testa
        forehead_pt = (rel_center[0] - int(eye_dist * 0.25), rel_center[1] - int(eye_dist * 0.65))
        f_start = (forehead_pt[0] - int(eye_dist * 0.18), forehead_pt[1] - int(eye_dist * 0.08))
        f_end = (forehead_pt[0] + int(eye_dist * 0.18), forehead_pt[1] + int(eye_dist * 0.08))

        cv2.line(blood_layer, f_start, f_end, (18, 16, 145), max(3, int(eye_dist * 0.055)), cv2.LINE_AA)
        cv2.line(blood_alpha, f_start, f_end, 0.80 * intensity, max(3, int(eye_dist * 0.055)), cv2.LINE_AA)
        cv2.line(blood_layer, f_start, f_end, (5, 3, 26), max(1, int(eye_dist * 0.022)), cv2.LINE_AA)
        cv2.line(blood_alpha, f_start, f_end, 0.98 * intensity, max(1, int(eye_dist * 0.022)), cv2.LINE_AA)

        # Suavização realista da máscara de sangue
        blood_alpha = cv2.GaussianBlur(blood_alpha, (5, 5), 0)
        ba_3 = np.expand_dims(blood_alpha, axis=-1)

        roi_final = np.clip(
            roi_mod.astype(np.float32) * (1.0 - ba_3) + blood_layer.astype(np.float32) * ba_3,
            0,
            255,
        ).astype(np.uint8)

        # Remonta sobre o quadro original
        out = frame.copy()
        out[y1:y2, x1:x2] = roi_final
        return out
