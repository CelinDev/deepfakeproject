"""Ponto de entrada da aplicação FaceLab (PySide6 GUI).

Uso:
    python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

