"""Fase 2 — teste de ida e volta: lê um vídeo e o regrava sem alterações.

Serve para validar leitura, FPS, resolução, áudio e sincronização antes de
qualquer processamento facial.

Uso:
    python scripts/passthrough.py "temp\\meu video.mp4"
    python scripts/passthrough.py entrada.mp4 --saida output\\teste.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUT_DIR  # noqa: E402
from src.utils.validators import validate_output_path, validate_video_path  # noqa: E402
from src.video.frame_source import VideoFileSource  # noqa: E402
from src.video.media_probe import probe_video  # noqa: E402
from src.video.video_writer import VideoFileSink  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada")
    parser.add_argument("--saida", default=None)
    args = parser.parse_args()

    src = validate_video_path(args.entrada)
    out = validate_output_path(args.saida or OUTPUT_DIR / f"{src.stem}_passthrough.mp4", src)

    source = VideoFileSource(src)
    info = source.info
    print(f"Entrada : {src.name}")
    print(f"  {info.width}x{info.height} @ {info.fps} ({float(info.fps):.4f} fps), "
          f"{info.duration_s:.3f}s, {info.frame_count} quadros, "
          f"áudio={info.audio_codec or 'não'}")

    sink = VideoFileSink(out, info.width, info.height, info.fps,
                         audio_source=src if info.has_audio else None)
    t0 = time.perf_counter()
    with source, sink:
        for frame in source.frames():
            sink.write(frame)
    elapsed = time.perf_counter() - t0

    res = probe_video(out)
    print(f"Saída   : {out}")
    print(f"  {res.width}x{res.height} @ {res.fps}, {res.duration_s:.3f}s, "
          f"{res.frame_count} quadros, áudio={res.audio_codec or 'não'}")
    print(f"  {sink.frames_written} quadros em {elapsed:.1f}s "
          f"({sink.frames_written / elapsed:.1f} quadros/s)")

    problems = []
    if res.resolution != info.resolution:
        problems.append("resolução diferente")
    if res.fps != info.fps:
        problems.append("FPS diferente")
    if info.frame_count and sink.frames_written != info.frame_count:
        problems.append(f"quadros lidos {sink.frames_written} != {info.frame_count}")
    if abs(res.duration_s - info.duration_s) > 1.5 / float(info.fps):
        problems.append("duração do vídeo diverge mais de 1 quadro")
    if info.has_audio != res.has_audio:
        problems.append("trilha de áudio perdida")
    print("Resultado:", "OK ✔" if not problems else "PROBLEMAS: " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

