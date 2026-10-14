"""Validação de arquivos de entrada e saída."""
from __future__ import annotations

from pathlib import Path

from config.settings import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS


class ValidationError(ValueError):
    """Erro de validação com mensagem pensada para o usuário final."""


def _validate_file(path: str | Path, allowed: frozenset[str], kind: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValidationError(f"Arquivo de {kind} não encontrado: {p}")
    if not p.is_file():
        raise ValidationError(f"O caminho de {kind} não é um arquivo: {p}")
    if p.suffix.lower() not in allowed:
        formatos = ", ".join(sorted(allowed))
        raise ValidationError(f"Formato de {kind} não suportado ({p.suffix}). Use: {formatos}")
    if p.stat().st_size == 0:
        raise ValidationError(f"Arquivo de {kind} está vazio: {p}")
    return p


def validate_video_path(path: str | Path) -> Path:
    """Valida um vídeo de entrada e devolve o caminho absoluto."""
    return _validate_file(path, SUPPORTED_VIDEO_EXTENSIONS, "vídeo")


def validate_image_path(path: str | Path) -> Path:
    """Valida uma imagem de entrada e devolve o caminho absoluto."""
    return _validate_file(path, SUPPORTED_IMAGE_EXTENSIONS, "imagem")


def validate_output_path(output: str | Path, source: str | Path) -> Path:
    """Valida o arquivo de saída.

    Regras: extensão .mp4, pasta existente (é criada se faltar) e nunca
    sobrescrever o arquivo original — preservação dos originais.
    """
    out = Path(output).expanduser().resolve()
    if out.suffix.lower() != ".mp4":
        raise ValidationError("O arquivo de saída deve ter extensão .mp4")
    if out == Path(source).expanduser().resolve():
        raise ValidationError("A saída não pode sobrescrever o vídeo original.")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

