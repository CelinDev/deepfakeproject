"""Testes automatizados do LivePipeline e WebcamSource."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.interfaces import FrameSource
from src.core.types import Frame
from src.pipeline.live_pipeline import LivePipeline
from src.video.webcam_source import WebcamSource


class DummyStreamSource(FrameSource):
    """Fonte sintética de quadros para testar o LivePipeline sem webcam física."""

    def __init__(self, frame_count: int = 5) -> None:
        self.frame_count = frame_count
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def frames(self):
        for i in range(self.frame_count):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            yield Frame(index=i, timestamp_s=i * 0.033, image=img)

    def close(self) -> None:
        self.opened = False


def test_dummy_stream_source() -> None:
    source = DummyStreamSource(3)
    with source:
        frames = list(source.frames())
    assert len(frames) == 3
    assert frames[0].image.shape == (480, 640, 3)
    assert not source.opened


def test_live_pipeline_with_dummy_source() -> None:
    received_frames = []

    def on_frame(img: np.ndarray, fps: float) -> None:
        received_frames.append(img)

    pipeline = LivePipeline(on_frame_rendered=on_frame)
    # Testa a execução com fonte sintética e foto do rosto
    pipeline.run(
        source_image_path="temp/rosto.png",
        camera_source=DummyStreamSource(4),
    )
    assert len(received_frames) == 4
    assert received_frames[0].shape == (480, 640, 3)


def test_enhancer_reuse_interval_in_crop() -> None:
    """Garante que o GFPGAN no modo live possa rodar a cada N quadros e interpolar nos intermediários sem tocar no frame inteiro."""
    from unittest.mock import MagicMock
    from src.face.face_enhancer import FaceEnhancer

    enh = FaceEnhancer.__new__(FaceEnhancer)
    enh.input_size = (512, 512)
    enh.input_name = "input"
    enh.output_name = "output"
    enh.blend_ratio = 1.0
    enh._frame_count = 0
    enh._cached_delta_512 = None

    # Simula saída NCHW (1, 3, 512, 512) do ONNX Runtime
    fake_out = np.full((1, 3, 512, 512), 0.5, dtype=np.float32)
    enh.session = MagicMock()
    enh.session.run.return_value = [fake_out]

    crop_in = np.full((128, 128, 3), 100, dtype=np.uint8)
    out1 = enh.enhance(crop_in, reuse_interval=3)
    out2 = enh.enhance(crop_in, reuse_interval=3)
    out3 = enh.enhance(crop_in, reuse_interval=3)
    out4 = enh.enhance(crop_in, reuse_interval=3)

    assert out1.shape == (512, 512, 3)
    assert out2.shape == (512, 512, 3)
    assert out3.shape == (512, 512, 3)
    assert out4.shape == (512, 512, 3)
    # Com reuse_interval=3 e 4 quadros (índices 0, 1, 2, 3), a sessão ONNX só deve rodar nos quadros 0 e 3 (2 vezes)
    assert enh.session.run.call_count == 2

