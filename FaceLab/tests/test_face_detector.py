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

