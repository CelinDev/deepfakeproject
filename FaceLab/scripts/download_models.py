"""Script utilitário para download automático dos modelos pré-treinados do FaceLab.

Baixa para a pasta models/:
1. inswapper_128.onnx (modelo base de face swap)
2. gfpgan_1.4.onnx (restauração e super-resolução facial)
3. buffalo_l (SCRFD + ArcFace do InsightFace)
"""
from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

# Adiciona raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128.onnx",
        "dest": MODELS_DIR / "inswapper_128.onnx",
        "is_zip": False,
    },
    "gfpgan_1.4.onnx": {
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        "dest": MODELS_DIR / "gfpgan_1.4.onnx",
        "is_zip": False,
    },
    "buffalo_l.zip": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "dest": MODELS_DIR / "models" / "buffalo_l.zip",
        "is_zip": True,
        "extract_to": MODELS_DIR / "models" / "buffalo_l",
    },
}


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file() and dest_path.stat().st_size > 1000:
        logger.info("✔ Já existe: %s (%d MB)", dest_path.name, dest_path.stat().st_size // (1024 * 1024))
        return

    logger.info("⏳ Baixando %s de: %s", dest_path.name, url)

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rProgresso: {percent}% [{count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB]")
        sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, str(dest_path), reporthook=_progress)
    print()
    logger.info("✔ Concluído: %s", dest_path.name)


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Iniciando verificação/download dos modelos para: %s", MODELS_DIR)

    for name, info in MODELS.items():
        dest = info["dest"]
        download_file(info["url"], dest)
        if info.get("is_zip"):
            extract_to = info["extract_to"]
            if not (extract_to / "det_10g.onnx").is_file():
                logger.info("Extraindo %s para %s ...", dest.name, extract_to)
                with zipfile.ZipFile(dest, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info("✔ Extração concluída.")

    logger.info("🎉 Todos os modelos estão prontos para uso no FaceLab!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

