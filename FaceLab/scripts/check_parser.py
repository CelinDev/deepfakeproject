"""Script de verificação visual da segmentação facial (BiSeNet FaceParser).

Executa o ``FaceParser`` sobre uma imagem (por padrão ``assets/vampire_reference.png``)
e salva um painel comparativo colorido em ``output/parser_check.png`` contendo:
1. Imagem original
2. Mapa colorido das 19 classes anatômicas (CelebAMask-HQ)
3. Sobreposição semitransparente e máscara de oclusão (cabelo/orelhas/dentes)

Uso:
    python scripts/check_parser.py [caminho_da_imagem] [--saida output/parser_check.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import OUTPUT_DIR, VAMPIRE_REFERENCE_IMAGE  # noqa: E402
from src.face.face_parser import FACE_PARSER_CLASSES, FaceParser  # noqa: E402
from src.inference.device_manager import DeviceManager  # noqa: E402

# Paleta BGR distinta para as 19 classes do CelebAMask-HQ
PALETTE_BGR = np.array(
    [
        [20, 20, 20],      # 0: background
        [180, 200, 255],   # 1: skin
        [0, 165, 255],     # 2: l_brow
        [0, 140, 255],     # 3: r_brow
        [255, 255, 0],     # 4: l_eye
        [255, 200, 0],     # 5: r_eye
        [255, 0, 255],     # 6: eye_g (óculos)
        [147, 20, 255],    # 7: l_ear
        [180, 50, 255],    # 8: r_ear
        [0, 255, 255],     # 9: ear_r
        [120, 180, 255],   # 10: nose
        [255, 255, 255],   # 11: mouth (interior/dentes)
        [80, 80, 255],     # 12: u_lip
        [50, 50, 220],     # 13: l_lip
        [150, 180, 210],   # 14: neck
        [0, 215, 255],     # 15: neck_l
        [200, 100, 50],    # 16: cloth
        [60, 220, 60],     # 17: hair (verde vibrante)
        [255, 100, 100],   # 18: hat
    ],
    dtype=np.uint8,
)


def colorize_parsing(parsed_map: np.ndarray) -> np.ndarray:
    """Converte o mapa 2D de classes (0..18) em uma imagem BGR colorida."""
    idx = np.clip(parsed_map, 0, len(PALETTE_BGR) - 1)
    return PALETTE_BGR[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "imagem",
        nargs="?",
        default=str(VAMPIRE_REFERENCE_IMAGE),
        help="Caminho da imagem de entrada para testar o parser",
    )
    ap.add_argument(
        "--saida",
        default=str(OUTPUT_DIR / "parser_check.png"),
        help="Caminho onde salvar a visualização colorida",
    )
    args = ap.parse_args()

    img_path = Path(args.imagem).expanduser().resolve()
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERRO] Não foi possível abrir a imagem: {img_path}")
        return 1

    dm = DeviceManager()
    parser = FaceParser(dm)
    parsed = parser.parse(img)
    occ_mask = parser.extract_mask(img, feather=5)

    color_map = colorize_parsing(parsed)
    overlay = cv2.addWeighted(img, 0.45, color_map, 0.55, 0)
    occ_vis = cv2.cvtColor((np.clip(occ_mask, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    target_h = 512
    target_w = int(round(w * (target_h / h)))
    panels = [
        cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for p in (img, color_map, overlay, occ_vis)
    ]
    grid = np.hstack(panels)

    out_path = Path(args.saida).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)

    unique_ids, counts = np.unique(parsed, return_counts=True)
    total_px = parsed.size
    print(f"[OK] Visualização salva em: {out_path}")
    print("Classes detectadas:")
    for cid, cnt in zip(unique_ids, counts):
        cname = FACE_PARSER_CLASSES.get(int(cid), f"class_{cid}")
        print(f"  - [{int(cid):2d}] {cname:<12}: {100.0 * cnt / total_px:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

