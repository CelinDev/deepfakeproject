"""Script de compilação e empacotamento portátil do FaceLab (Windows x64).

Gera a distribuição completa em `FaceLab/dist/FaceLab/`:
1. Compila `FaceLab.exe` via PyInstaller (com PySide6, ONNX Runtime GPU, InsightFace, OpenCV e PyTorch/CUDA).
2. Copia automaticamente as pastas externas ao lado do `.exe`:
   - `assets/` (presets: zumbi, vampiro, ciborgue)
   - `tools/ffmpeg/bin/` (ffmpeg.exe e ffprobe.exe locais)
   - `models/` (todos os pesos .onnx já baixados para funcionar offline na outra máquina)
   - Cria as pastas vazias `output/`, `temp/` e `logs/`
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FACELAB_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = FACELAB_DIR / "dist" / "FaceLab"
BUILD_DIR = FACELAB_DIR / "build"


def main() -> int:
    print("=== 1/3 Compilando FaceLab.exe com PyInstaller ===")
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "FaceLab",
        "--paths",
        str(FACELAB_DIR),
        "--collect-all",
        "insightface",
        "--collect-all",
        "onnxruntime",
        "--hidden-import",
        "scipy._lib.messagestream",
        "--hidden-import",
        "skimage.transform",
        "--exclude-module",
        "pytest",
        str(FACELAB_DIR / "app.py"),
    ]
    res = subprocess.run(pyinstaller_cmd, cwd=str(FACELAB_DIR))
    if res.returncode != 0:
        print("[ERRO] Falha durante a compilação com PyInstaller.")
        return res.returncode

    print("\n=== 2/3 Copiando recursos externos (assets, tools) SEM a pasta pesada de modelos ===")
    for folder_name in ("assets", "tools"):
        src = FACELAB_DIR / folder_name
        dst = DIST_DIR / folder_name
        if src.is_dir():
            print(f"Copiando {folder_name}/ -> {dst}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Cria a pasta models/ vazia (apenas com o README.md) para que o programa peça o download obrigatório ao iniciar
    dst_models = DIST_DIR / "models"
    if dst_models.exists():
        shutil.rmtree(dst_models)
    dst_models.mkdir(parents=True, exist_ok=True)
    readme_models = FACELAB_DIR / "models" / "README.md"
    if readme_models.is_file():
        shutil.copy2(readme_models, dst_models / "README.md")

    for runtime_dir in ("output", "temp", "logs"):
        (DIST_DIR / runtime_dir).mkdir(parents=True, exist_ok=True)

    exe_path = DIST_DIR / "FaceLab.exe"
    print("\n=== 3/3 Compilação concluída com sucesso! ===")
    print(f"Executável gerado em: {exe_path}")
    print(f"Para levar para outra máquina, basta copiar a pasta inteira: {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

