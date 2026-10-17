"""Ponto de entrada da aplicação FaceLab (PySide6 GUI).

Se executado fora do .venv, redireciona automaticamente para o interpretador do .venv.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = CURRENT_DIR / ".venv" / "Scripts" / "python.exe"

# Se executado pelo Python global, relança automaticamente com o Python do .venv
if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    cmd = [str(VENV_PYTHON), str(CURRENT_DIR / "app.py")] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

# Garante que a pasta do projeto esteja no sys.path
sys.path.insert(0, str(CURRENT_DIR))

from PySide6.QtWidgets import QApplication
from src.interface.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
