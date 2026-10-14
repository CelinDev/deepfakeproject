"""Gravação do vídeo final via FFmpeg, com áudio original e metadados de IA.

Como funciona: abrimos o FFmpeg como subprocesso e enviamos cada quadro
(bytes BGR crus) pelo stdin. O mesmo comando lê o vídeo original como
segunda entrada só para copiar a trilha de áudio, sem recodificá-la.
Assim vídeo e áudio já saem juntos e sincronizados num único passo.

Por que não ``cv2.VideoWriter``? Ele aceita FPS apenas como float, não
copia áudio e não grava metadados.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import cv2

from config.settings import TEMP_DIR, find_executable
from src.core.interfaces import FrameSink
from src.core.types import Frame

AI_DISCLOSURE = (
    "Conteudo sintetico: video alterado por IA (face swap) com o FaceLab, "
    "software educacional."
)


class VideoWriteError(RuntimeError):
    """Falha durante a gravação do vídeo."""


class VideoFileSink(FrameSink):
    """Grava quadros em MP4 (H.264) e copia o áudio de ``audio_source``."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: Fraction,
        audio_source: str | Path | None = None,
        crf: int = 18,
        disclosure: str = AI_DISCLOSURE,
    ) -> None:
        self.output_path = Path(output_path)
        self.width, self.height = width, height
        self.fps = fps
        self.audio_source = Path(audio_source) if audio_source else None
        self.crf = crf
        self.disclosure = disclosure
        self.frames_written = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self.log_path = TEMP_DIR / f"ffmpeg_{self.output_path.stem}.log"

    def _build_command(self) -> list[str]:
        ffmpeg = find_executable("ffmpeg")
        if ffmpeg is None:
            raise VideoWriteError("ffmpeg não encontrado (tools/ffmpeg/bin ou PATH).")
        cmd = [
            str(ffmpeg), "-y", "-hide_banner",
            # Entrada 0: quadros crus pelo stdin
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-framerate", f"{self.fps.numerator}/{self.fps.denominator}",
            "-i", "pipe:0",
        ]
        if self.audio_source:
            cmd += ["-i", str(self.audio_source)]  # Entrada 1: original (só áudio)
        cmd += ["-map", "0:v:0"]
        if self.audio_source:
            cmd += ["-map", "1:a:0?", "-c:a", "copy"]  # '?' = opcional se não houver áudio
        cmd += [
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", "medium",
            "-pix_fmt", "yuv420p",  # compatível com a maioria dos players
            "-metadata", f"comment={self.disclosure}",
            "-metadata", f"description={self.disclosure}",
            "-movflags", "+faststart",
            str(self.output_path),
        ]
        return cmd

    def open(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        # stderr vai para um arquivo: se fosse um PIPE não lido, o buffer
        # encheria e o FFmpeg travaria (deadlock).
        self._log_file = open(self.log_path, "wb")
        self._proc = subprocess.Popen(
            self._build_command(), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self._log_file,
        )

    def write(self, frame: Frame) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise VideoWriteError("Chame open() antes de write().")
        image = frame.image
        h, w = image.shape[:2]
        if (w, h) != (self.width, self.height):
            # Trata vídeos com resolução variável: todo quadro sai no tamanho de saída.
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        try:
            self._proc.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise VideoWriteError(f"FFmpeg encerrou inesperadamente. Veja {self.log_path}") from exc
        self.frames_written += 1

    def close(self) -> None:
        if self._proc is None:
            return
        assert self._proc.stdin is not None
        self._proc.stdin.close()  # sinaliza fim dos quadros
        code = self._proc.wait()
        self._cleanup_handles()
        if code != 0:
            raise VideoWriteError(f"FFmpeg terminou com código {code}. Veja {self.log_path}")
        self.log_path.unlink(missing_ok=True)  # limpeza de temporário em caso de sucesso

    def abort(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        self._proc.kill()
        self._proc.wait()
        self._cleanup_handles()
        self.output_path.unlink(missing_ok=True)  # não deixa arquivo parcial

    def _cleanup_handles(self) -> None:
        self._proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

