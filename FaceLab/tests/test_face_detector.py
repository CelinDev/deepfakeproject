"""Testes da Fase 3.

- Lógica do tracker: rostos falsos, sem nenhum modelo (rápido, roda sempre).
- Detector real: só roda se os pesos do buffalo_l já estiverem baixados.
"""
from __future__ import annotations

import numpy as np
import pytest

from config.settings import MODELS_DIR
from src.core.types import DetectedFace
from src.face.face_tracker import TargetFaceTracker, iou


def _emb(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _face(x: float, emb: np.ndarray, size: float = 100) -> DetectedFace:
    return DetectedFace(np.array([x, 100, x + size, 100 + size], np.float32), None, 0.9, emb)


PESSOA_A, PESSOA_B = _emb(1), _emb(2)


def test_iou() -> None:
    a = np.array([0, 0, 10, 10], np.float32)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, np.array([20, 20, 30, 30], np.float32)) == 0.0
    assert iou(a, np.array([5, 0, 15, 10], np.float32)) == pytest.approx(50 / 150)


def test_segue_a_pessoa_certa_entre_varias() -> None:
    tracker = TargetFaceTracker(PESSOA_A)
    for x in range(0, 200, 20):  # A anda para a direita; B fica parada
        alvo = tracker.update([_face(500, PESSOA_B, 150), _face(x, PESSOA_A)])
        assert alvo is not None and np.allclose(alvo.embedding, PESSOA_A)


def test_ignora_outra_pessoa_quando_alvo_some() -> None:
    tracker = TargetFaceTracker(PESSOA_A)
    tracker.update([_face(0, PESSOA_A)])
    # O alvo saiu de cena; B aparece exatamente onde A estava.
    assert tracker.update([_face(0, PESSOA_B)]) is None


def test_quadro_sem_rostos() -> None:
    assert TargetFaceTracker(PESSOA_A).update([]) is None


def test_select_largest() -> None:
    grande = _face(0, PESSOA_A, 200)
    assert TargetFaceTracker.select_largest([_face(300, PESSOA_B, 50), grande]) is grande
    assert TargetFaceTracker.select_largest([]) is None


def test_tracker_public_missing_and_velocity_prediction_without_embedding() -> None:
    """Garante propriedade pública tracker.missing, predição por velocidade no modo sem embedding e EMA de identidade."""
    tracker = TargetFaceTracker(PESSOA_A)
    assert tracker.missing == 0

    # Quadros 0 e 1 com embedding: rosto move-se rápido (+120px/frame), de modo que no quadro 2 o IoU com a caixa antiga seria 0
    tracker.update([_face(0, PESSOA_A, size=100)])
    tracker.update([_face(120, PESSOA_A, size=100)])

    # Quadro 2 SEM embedding (embedding=None) em x=240, com segunda pessoa parada em x=700:
    # Sem predição de velocidade, IoU([120,100,220,200], [240,100,340,200]) == 0.0 e descartaria!
    fast_face_no_emb = DetectedFace(np.array([240.0, 100.0, 340.0, 200.0], np.float32), None, 0.9, None)
    bystander_no_emb = DetectedFace(np.array([700.0, 100.0, 800.0, 200.0], np.float32), None, 0.9, None)
    matched = tracker.update([bystander_no_emb, fast_face_no_emb])
    assert matched is fast_face_no_emb
    assert tracker.missing == 0

    # Quando não há rostos, missing incrementa publicamente
    assert tracker.update([]) is None
    assert tracker.missing == 1

    # Atualização lenta do embedding quando a similaridade é alta (ex: colocou óculos / virou levemente)
    variacao_oculos = 0.85 * PESSOA_A + 0.15 * PESSOA_B
    variacao_oculos = (variacao_oculos / np.linalg.norm(variacao_oculos)).astype(np.float32)
    sim_before = float(np.dot(tracker.target_embedding, variacao_oculos))
    tracker.update([_face(360, variacao_oculos, size=100)])
    sim_after = float(np.dot(tracker.target_embedding, variacao_oculos))
    assert sim_after > sim_before


def test_select_target_manual_by_index_point_and_embedding() -> None:
    """Garante a seleção manual do rosto-alvo por clique (x, y), índice ordenado (esq->dir) ou embedding."""
    f_left = _face(20, PESSOA_A, size=80)    # centro (60, 140), menor
    f_right = _face(400, PESSOA_B, size=180) # centro (490, 190), maior

    faces = [f_right, f_left]

    # Por padrão sem argumentos, escolhe o maior (f_right)
    assert TargetFaceTracker.select_target(faces) is f_right
    # Por índice 0 (esquerda -> direita), escolhe f_left mesmo sendo menor
    assert TargetFaceTracker.select_target(faces, target_index=0) is f_left
    assert TargetFaceTracker.select_target(faces, target_index=1) is f_right
    # Por clique (x, y) dentro ou próximo de f_left
    assert TargetFaceTracker.select_target(faces, target_point=(55.0, 135.0)) is f_left
    assert TargetFaceTracker.select_target(faces, target_point=(480.0, 180.0)) is f_right
    # Por embedding específico
    assert TargetFaceTracker.select_target(faces, target_embedding=PESSOA_A) is f_left


def test_optical_flow_propagator_tracks_intermediate_and_falls_back_on_low_confidence() -> None:
    """Garante que OpticalFlowPropagator propague landmarks entre detecções SCRFD e retorne None quando a confiança cai."""
    import cv2
    from src.face.face_tracker import OpticalFlowPropagator

    propagator = OpticalFlowPropagator(min_confidence=0.65)
    # Cria quadro texturizado sintético e desloca +6 pixels em X no quadro seguinte
    rng = np.random.default_rng(99)
    base_patch = rng.integers(30, 220, size=(240, 320, 3), dtype=np.uint8)
    base_patch = cv2.GaussianBlur(base_patch, (5, 5), 1.0)

    M_shift = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    shifted_patch = cv2.warpAffine(base_patch, M_shift, (320, 240), borderMode=cv2.BORDER_REFLECT)

    lmk0 = np.array(
        [[110.0, 90.0], [170.0, 90.0], [140.0, 120.0], [120.0, 150.0], [160.0, 150.0]],
        dtype=np.float32,
    )
    face0 = DetectedFace(
        bbox=np.array([90.0, 60.0, 190.0, 180.0], dtype=np.float32),
        landmarks=lmk0,
        score=0.95,
        embedding=PESSOA_A,
    )

    propagator.update_reference(base_patch)
    prop_face, conf = propagator.propagate(shifted_patch, face0)
    assert prop_face is not None
    assert conf >= 0.65
    assert prop_face.landmarks is not None
    # Os landmarks propagados devem ter acompanhado o deslocamento (+6, -3)
    mean_shift = (prop_face.landmarks - lmk0).mean(axis=0)
    assert np.allclose(mean_shift, [6.0, -3.0], atol=1.0)

    # Se o próximo quadro for completamente preto (oclusão total / perda de textura), a confiança cai e retorna None
    black_frame = np.zeros_like(base_patch)
    lost_face, lost_conf = propagator.propagate(black_frame, prop_face)
    assert lost_face is None





MODEL_DIR = MODELS_DIR / "models" / "buffalo_l"


@pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="buffalo_l não baixado")
def test_detector_real_imagem_sem_rosto_e_com_rosto() -> None:
    from src.face.face_detector import InsightFaceDetector
    from src.inference.device_manager import DeviceManager

    det = InsightFaceDetector(DeviceManager())
    assert det.detect(np.zeros((480, 640, 3), np.uint8)) == []

    import insightface.data
    img = insightface.data.get_image("t1")  # foto de grupo de exemplo do InsightFace
    faces = det.detect(img)
    assert len(faces) >= 2
    for f in faces:
        assert f.landmarks.shape == (5, 2)
        assert f.embedding is not None and np.linalg.norm(f.embedding) == pytest.approx(1, abs=1e-3)

