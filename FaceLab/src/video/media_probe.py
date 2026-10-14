"""Leitura de metadados de mídia com ffprobe.

Por que ffprobe e não só o OpenCV? O OpenCV informa o FPS como float
arredondado e não sabe se há áudio. O ffprobe devolve a taxa exata como
fração (ex.: ``19261/642``) e lista todas as trilhas.
"""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from config.settings import find_executable
from src.core.types import VideoInfo


class MediaProbeError(RuntimeError):
    """Falha ao ler os metadados do arquivo."""


def _ffprobe() -> Path:
    exe = find_executable("ffprobe")
    if exe is None:
        raise MediaProbeError("ffprobe não encontrado (tools/ffmpeg/bin ou PATH).")
    return exe


def _parse_rate(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "0"):
        return None
    try:
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def probe_video(path: str | Path) -> VideoInfo:
    """Lê resolução, FPS, duração e presença de áudio de um vídeo."""
    path = Path(path)
    cmd = [
        str(_ffprobe()), "-v", "error",
        "-show_streams", "-show_format", "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise MediaProbeError(f"ffprobe falhou em {path.name}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise MediaProbeError(f"Nenhuma trilha de vídeo em {path.name}.")

    # avg_frame_rate representa melhor vídeos com FPS variável; r_frame_rate é o fallback.
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    if fps is None:
        raise MediaProbeError(f"Não foi possível determinar o FPS de {path.name}.")

    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = video.get("nb_frames")

    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration_s=duration,
        frame_count=int(nb) if nb and str(nb).isdigit() else None,
        video_codec=video.get("codec_name", "?"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
    )

