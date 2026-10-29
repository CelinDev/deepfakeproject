"""Script de compilação e geração do Instalador Único do FaceLab (Windows x64).

Gera:
1. A distribuição enxuta em `FaceLab/dist/FaceLab/` (`FaceLab.exe` + ONNX Runtime GPU + CUDA 12 / cuDNN 9 + FFmpeg + Presets, sem a pasta pesada `models/` e sem binários C++ ociosos do PyTorch).
2. O instalador auto-extraível de arquivo único `FaceLab/dist/release_assets/FaceLab-Setup-v1.0.0.exe` (cria pasta de instalação, atalhos na Área de Trabalho e no Menu Iniciar, e inicia o `FaceLab.exe`).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

FACELAB_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = FACELAB_DIR / "dist" / "FaceLab"
BUILD_DIR = FACELAB_DIR / "build"

# DLLs do CUDA 12 e cuDNN 9 necessárias para o ONNX Runtime GPU (CUDAExecutionProvider)
REQUIRED_CUDA_DLLS = {
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cufft64_11.dll",
    "cufftw64_11.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
    "nvJitLink_120_0.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_126.dll",
    "zlibwapi.dll",
}


def prune_unused_binaries() -> None:
    """Remove binários pesados do PyTorch C++ (~1.5 GB) que não são usados pelo ONNX Runtime GPU."""
    internal_dir = DIST_DIR / "_internal"
    torch_dir = internal_dir / "torch"
    cuda_libs_dir = internal_dir / "cuda_libs"
    cuda_libs_dir.mkdir(parents=True, exist_ok=True)

    # Copia apenas as DLLs do CUDA 12 e cuDNN 9 usadas pelo ONNX Runtime GPU
    venv_torch_lib = FACELAB_DIR / ".venv" / "Lib" / "site-packages" / "torch" / "lib"

    copied_mb = 0.0
    for dll_name in REQUIRED_CUDA_DLLS:
        src_dll = venv_torch_lib / dll_name
        if not src_dll.is_file() and (torch_dir / "lib" / dll_name).is_file():
            src_dll = torch_dir / "lib" / dll_name
        if src_dll.is_file():
            dst_dll = cuda_libs_dir / dll_name
            if not dst_dll.exists():
                shutil.copy2(src_dll, dst_dll)
            copied_mb += dst_dll.stat().st_size / (1024**2)

    # Remove duplicatas dessas mesmas DLLs se o PyInstaller tiver copiado para a raiz de _internal
    for dll_name in REQUIRED_CUDA_DLLS:
        dup = internal_dir / dll_name
        if dup.is_file() and (cuda_libs_dir / dll_name).is_file():
            dup.unlink()

    # Remove o pacote pesado torch / torchvision de _internal se tiver sido coletado
    for pkg in ("torch", "torchvision"):
        p = internal_dir / pkg
        if p.is_dir():
            shutil.rmtree(p)
        for dist_info in internal_dir.glob(f"{pkg}-*.dist-info"):
            shutil.rmtree(dist_info)

    # Remove ffplay.exe (102 MB) que não é usado pelo pipeline (apenas ffmpeg.exe e ffprobe.exe são usados)
    ffplay = DIST_DIR / "tools" / "ffmpeg" / "bin" / "ffplay.exe"
    if ffplay.is_file():
        ffplay.unlink()

    print(f"Bibliotecas CUDA 12 / cuDNN 9 mantidas em _internal/cuda_libs: {copied_mb:.1f} MB")


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
        "--exclude-module",
        "torch",
        "--exclude-module",
        "torchvision",
        "--exclude-module",
        "torchaudio",
        str(FACELAB_DIR / "app.py"),
    ]
    res = subprocess.run(pyinstaller_cmd, cwd=str(FACELAB_DIR))
    if res.returncode != 0:
        print("[ERRO] Falha durante a compilação com PyInstaller.")
        return res.returncode

    print("\n=== 2/3 Copiando recursos externos (assets, tools, CUDA libs) SEM a pasta pesada de modelos ===")
    for folder_name in ("assets", "tools"):
        src = FACELAB_DIR / folder_name
        dst = DIST_DIR / folder_name
        if src.is_dir():
            print(f"Copiando {folder_name}/ -> {dst}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    prune_unused_binaries()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
