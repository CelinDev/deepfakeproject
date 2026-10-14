"""Contratos (interfaces) reutilizáveis do FaceLab.

O pipeline depende destas abstrações, e não das classes concretas.
Assim, um futuro ``WebcamSource`` ou uma ``PreviewWindowSink`` poderão
substituir os componentes de arquivo sem alterar o pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType

import numpy as np

from src.core.types import DetectedFace, Frame


class FaceDetector(ABC):
    """Detecta rostos numa imagem BGR."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Retorna todos os rostos encontrados (lista vazia se nenhum)."""


class FaceSwapper(ABC):
    """Executa a substituição facial."""

    @abstractmethod
    def swap(
        self,
        target_frame: np.ndarray,
        target_face: DetectedFace,
        source_face: DetectedFace,
    ) -> np.ndarray:
        """Substitui o target_face no target_frame pelo source_face."""


class FrameSource(ABC):
    """Fonte de quadros: arquivo de vídeo hoje, webcam no futuro."""

    @abstractmethod
    def open(self) -> None:
        """Abre o recurso (arquivo, câmera...)."""

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Itera sobre os quadros em ordem."""

    @abstractmethod
    def close(self) -> None:
        """Libera o recurso. Deve ser seguro chamar mais de uma vez."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class FrameSink(ABC):
    """Destino de quadros: arquivo de vídeo hoje, janela ao vivo no futuro."""

    @abstractmethod
    def open(self) -> None:
        """Prepara o destino."""

    @abstractmethod
    def write(self, frame: Frame) -> None:
        """Recebe um quadro processado."""

    @abstractmethod
    def close(self) -> None:
        """Finaliza o destino (ex.: fecha o arquivo e grava o áudio)."""

    @abstractmethod
    def abort(self) -> None:
        """Interrompe sem finalizar (ex.: cancelamento); descarta saída parcial."""

    def __enter__(self) -> FrameSink:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Se houve erro/cancelamento, não deixa um arquivo corrompido para trás.
        if exc_type is None:
            self.close()
        else:
            self.abort()

