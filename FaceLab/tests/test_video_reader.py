"""Testes da Fase 2: leitura, gravação, áudio e validação.

Os vídeos de teste são gerados na hora pelo próprio FFmpeg (padrão de
cores + tom senoidal), então os testes não dependem de arquivos externos.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from config.settings import find_executable
from src.core.types import Frame
from src.utils.validators import ValidationError, validate_output_path, validate_video_path
from src.video.frame_source import VideoFileSource
from src.video.media_probe import probe_video
from src.video.video_writer import AI_DISCLOSURE, VideoFileSink

FFMPEG = find_executable("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="FFmpeg não encontrado")


def _make_video(path: Path, fps: str, seconds: float, audio: bool) -> Path:
    cmd = [str(FFMPEG), "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc=size=320x240:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def video_com_audio(tmp_path: Path) -> Path:
    # FPS "estranho" de propósito (NTSC 29,97) para testar a precisão da fração.
    return _make_video(tmp_path / "com_audio.mp4", "30000/1001", 2, audio=True)


@pytest.fixture
def video_sem_audio(tmp_path: Path) -> Path:
    return _make_video(tmp_path / "sem_audio.mp4", "25", 1, audio=False)


def test_probe_le_metadados(video_com_audio: Path) -> None:
    info = probe_video(video_com_audio)
    assert info.resolution == (320, 240)
    assert info.fps == Fraction(30000, 1001)
    assert info.has_audio and info.audio_codec == "aac"


def test_leitura_em_ordem(video_sem_audio: Path) -> None:
    with VideoFileSource(video_sem_audio) as src:
        frames = list(src.frames())
    assert len(frames) == 25
    assert [f.index for f in frames] == list(range(25))
    assert frames[0].image.shape == (240, 320, 3)
    assert frames[10].timestamp_s == pytest.approx(10 / 25)


def test_ida_e_volta_preserva_fps_audio_e_metadados(video_com_audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "saida.mp4"
    src = VideoFileSource(video_com_audio)
    info = src.info
    sink = VideoFileSink(out, info.width, info.height, info.fps, audio_source=video_com_audio)
    with src, sink:
        for frame in src.frames():
            sink.write(frame)

    res = probe_video(out)
    assert res.fps == info.fps
    assert res.resolution == info.resolution
    assert res.has_audio
    assert abs(res.duration_s - info.duration_s) < 1.5 / float(info.fps)

    tags = subprocess.run(
        [str(find_executable("ffprobe")), "-v", "error", "-show_entries",
         "format_tags=comment", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    assert tags == AI_DISCLOSURE


def test_quadro_com_resolucao_diferente_e_redimensionado(tmp_path: Path) -> None:
    out = tmp_path / "var.mp4"
    with VideoFileSink(out, 320, 240, Fraction(10)) as sink:
        sink.write(Frame(0, 0.0, np.zeros((240, 320, 3), np.uint8)))
        sink.write(Frame(1, 0.1, np.zeros((480, 640, 3), np.uint8)))  # tamanho diferente
    assert probe_video(out).resolution == (320, 240)


def test_abort_remove_arquivo_parcial(tmp_path: Path) -> None:
    out = tmp_path / "cancelado.mp4"
    with pytest.raises(KeyboardInterrupt):
        with VideoFileSink(out, 320, 240, Fraction(10)) as sink:
            sink.write(Frame(0, 0.0, np.zeros((240, 320, 3), np.uint8)))
            raise KeyboardInterrupt  # simula cancelamento
    assert not out.exists()


def test_validadores(video_sem_audio: Path, tmp_path: Path) -> None:
    assert validate_video_path(video_sem_audio) == video_sem_audio.resolve()
    with pytest.raises(ValidationError):
        validate_video_path(tmp_path / "nao_existe.mp4")
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(ValidationError):
        validate_video_path(txt)
    with pytest.raises(ValidationError):
        validate_output_path(video_sem_audio, video_sem_audio)  # sobrescrever original

