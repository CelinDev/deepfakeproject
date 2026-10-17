"""Atalho inteligente para executar a aplicação FaceLab.

Se você rodar `python app.py` com o Python padrão do Windows (sem ter ativado o venv),
este script detecta automaticamente e re-executa a aplicação usando o interpretador
do ambiente virtual `.venv` onde o PySide6, PyTorch e CUDA estão instalados.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Localiza o diretório do FaceLab e o interpretador do venv
ROOT_DIR = Path(__file__).resolve().parent
FACELAB_DIR = ROOT_DIR if (ROOT_DIR / "src").is_dir() else (ROOT_DIR / "FaceLab")
VENV_PYTHON = FACELAB_DIR / ".venv" / "Scripts" / "python.exe"

# Verifica se o Python atual é o do venv
current_executable = Path(sys.executable).resolve()
is_venv = (
    VENV_PYTHON.is_file()
    and (current_executable == VENV_PYTHON.resolve() or sys.prefix == str(FACELAB_DIR / ".venv"))
)

if not is_venv and VENV_PYTHON.is_file():
    # Re-executa automaticamente com o interpretador do .venv!
    os.chdir(str(FACELAB_DIR))
    cmd = [str(VENV_PYTHON), str(FACELAB_DIR / "app.py")] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

# Se já está no venv, segue o fluxo normal
sys.path.insert(0, str(FACELAB_DIR))
os.chdir(str(FACELAB_DIR))

from src.interface.main_window import MainWindow
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
