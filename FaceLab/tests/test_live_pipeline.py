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

