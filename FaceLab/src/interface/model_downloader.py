"""Gerenciador e Diálogo de Download Obrigatório de Modelos (First-Run Setup).

Verifica na inicialização do FaceLab se os modelos essenciais estão presentes em `models/`:
1. `inswapper_128.onnx` (Troca facial)
2. `gfpgan_1.4.onnx` (Restauração e super-resolução)
3. `bisenet_resnet_34.onnx` (Segmentação semântica de cabeça/cabelo)
4. `models/buffalo_l/det_10g.onnx` e `w600k_r50.onnx` (Detecção SCRFD + Identidade ArcFace)

Se qualquer modelo estiver ausente, exibe um diálogo modal solicitando a confirmação
obrigatória do download com barra de progresso em tempo real.
"""
from __future__ import annotations

import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.settings import BISENET_MODEL_PATH, MODELS_DIR
from src.interface.theme import DARK_BLUE_THEME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    dest: Path
    size_mb_approx: int
    is_zip: bool = False
    extract_to: Path | None = None
    check_file: Path | None = None

    def is_present(self) -> bool:
        target = self.check_file if self.check_file is not None else self.dest
        return target.is_file() and target.stat().st_size > 100_000


REQUIRED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="INSwapper 128 (Troca Facial)",
        url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128.onnx",
        dest=MODELS_DIR / "inswapper_128.onnx",
        size_mb_approx=529,
    ),
    ModelSpec(
        name="GFPGAN 1.4 (Super-Resolução Facial)",
        url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        dest=MODELS_DIR / "gfpgan_1.4.onnx",
        size_mb_approx=325,
    ),
    ModelSpec(
        name="BiSeNet ResNet-34 (Segmentação de Cabeça/Cabelo)",
        url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/bisenet_resnet_34.onnx",
        dest=BISENET_MODEL_PATH,
        size_mb_approx=90,
    ),
    ModelSpec(
        name="InsightFace Buffalo_L (Detecção & ArcFace)",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        dest=MODELS_DIR / "models" / "buffalo_l.zip",
        size_mb_approx=275,
        is_zip=True,
        extract_to=MODELS_DIR / "models" / "buffalo_l",
        check_file=MODELS_DIR / "models" / "buffalo_l" / "det_10g.onnx",
    ),
)


def get_missing_models() -> list[ModelSpec]:
    """Retorna a lista de modelos obrigatórios que ainda não foram baixados."""
    return [m for m in REQUIRED_MODELS if not m.is_present()]


class _ModelDownloadWorker(QObject):
    """Faz o download dos modelos ausentes em uma QThread separada."""

    progress = Signal(str, int, int, int)  # (nome_modelo, percentual_item, mb_baixados, mb_total)
    finished = Signal()
    error = Signal(str)

    def __init__(self, missing: list[ModelSpec]) -> None:
        super().__init__()
        self.missing = missing
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            total_items = len(self.missing)

            for idx, spec in enumerate(self.missing, start=1):
                if self._cancelled:
                    return

                spec.dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = spec.dest.with_suffix(spec.dest.suffix + ".part")
                prefix = f"[{idx}/{total_items}] {spec.name}"

                req = urllib.request.Request(spec.url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out_file:
                    total_bytes = int(resp.headers.get("Content-Length", 0) or (spec.size_mb_approx * 1024 * 1024))
                    downloaded = 0
                    chunk_size = 512 * 1024
                    while True:
                        if self._cancelled:
                            out_file.close()
                            tmp_path.unlink(missing_ok=True)
                            return
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        pct = min(100, int(downloaded * 100 / total_bytes)) if total_bytes > 0 else 0
                        self.progress.emit(
                            prefix,
                            pct,
                            downloaded // (1024 * 1024),
                            total_bytes // (1024 * 1024),
                        )

                if tmp_path.is_file():
                    tmp_path.replace(spec.dest)

                if spec.is_zip and spec.extract_to is not None:
                    self.progress.emit(f"Extraindo {spec.name}...", 100, 0, 0)
                    spec.extract_to.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(spec.dest, "r") as zf:
                        zf.extractall(spec.extract_to)
                    spec.dest.unlink(missing_ok=True)

            self.finished.emit()
        except Exception as exc:
            logger.exception("Falha ao baixar modelos obrigatórios:")
            self.error.emit(str(exc))


class ModelDownloadDialog(QDialog):
    """Janela modal exibida na inicialização quando faltam modelos obrigatórios."""

    def __init__(self, missing: list[ModelSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.missing = missing
        self.setWindowTitle("⚡ FaceLab Studio — Download Obrigatório de Modelos")
        self.setMinimumWidth(540)
        self.setModal(True)
        self.setStyleSheet(DARK_BLUE_THEME)

        self._thread: QThread | None = None
        self._worker: _ModelDownloadWorker | None = None
        self._completed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        title = QLabel("📦 Modelos de IA Necessários")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(title)

        total_mb = sum(m.size_mb_approx for m in missing)
        names_html = "".join(f"<li><b>{m.name}</b> (~{m.size_mb_approx} MB)</li>" for m in missing)
        desc = QLabel(
            f"Para que o FaceLab funcione, é obrigatório baixar os seguintes modelos neurais "
            f"(tamanho total aproximado: <b>~{total_mb} MB</b>):"
            f"<ul style='margin-top:6px; margin-bottom:6px;'>{names_html}</ul>"
            f"Os arquivos serão salvos na pasta <code>models/</code> e este download só é necessário na primeira execução."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.lbl_status = QLabel("Aguardando confirmação para iniciar o download...")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_exit = QPushButton("Sair do Programa")
        self.btn_exit.setObjectName("cancelBtn")
        self.btn_exit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_exit.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.btn_exit)

        self.btn_download = QPushButton(f"⬇ Baixar Modelos (~{total_mb} MB)")
        self.btn_download.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_download.clicked.connect(self._start_download)
        btn_layout.addWidget(self.btn_download)

        layout.addLayout(btn_layout)

    def _start_download(self) -> None:
        self.btn_download.setEnabled(False)
        self.lbl_status.setText("Conectando aos repositórios oficiais de modelos...")

        self._thread = QThread(self)
        self._worker = _ModelDownloadWorker(self.missing)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_progress(self, label: str, pct: int, dl_mb: int, total_mb: int) -> None:
        self.progress_bar.setValue(pct)
        if total_mb > 0:
            self.lbl_status.setText(f"{label} — {pct}% ({dl_mb} MB / {total_mb} MB)")
        else:
            self.lbl_status.setText(label)

    def _on_finished(self) -> None:
        self._completed = True
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✔ Todos os modelos foram baixados com sucesso!")
        self.accept()

    def _on_error(self, msg: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_download.setText("🔄 Tentar Novamente")
        self.lbl_status.setText(f"Erro no download: {msg}")
        QMessageBox.critical(
            self,
            "Erro ao Baixar Modelos",
            f"Não foi possível concluir o download obrigatório dos modelos:\n\n{msg}\n\n"
            "Verifique sua conexão com a internet e clique em 'Tentar Novamente'.",
        )

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self.reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._completed:
            self._on_cancel()
        super().closeEvent(event)


def ensure_models_installed(parent: QWidget | None = None) -> bool:
    """Verifica se todos os modelos existem; se não, abre o diálogo obrigatório.

    Retorna True se todos os modelos estão prontos, ou False se o usuário recusou/fechou.
    """
    missing = get_missing_models()
    if not missing:
        return True

    dlg = ModelDownloadDialog(missing, parent=parent)
    result = dlg.exec()
    return result == QDialog.DialogCode.Accepted and not get_missing_models()

