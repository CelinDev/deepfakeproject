"""Script de Execução de Face Swap via Linha de Comando (CLI).

Uso:
    python scripts/run_swap.py "temp/video baixado cortado.mp4" "temp/rosto.png"
    python scripts/run_swap.py video.mp4 foto.jpg --device gpu --saida output/resultado.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.device_manager import DevicePreference  # noqa: E402
from src.pipeline.video_pipeline import PipelineProgress, VideoPipeline  # noqa: E402


def on_progress(p: PipelineProgress) -> None:
    tot_str = f"/{p.total_frames}" if p.total_frames else ""
    target_str = "ALVO OK" if p.target_found else "SEM ALVO"
    print(
        f"\r[Quadro {p.current_frame}{tot_str}] "
        f"{p.fps:.1f} fps | {p.elapsed_seconds:.1f}s decorridos | "
        f"Rostos: {p.faces_detected} | {target_str}  ",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Caminho do vídeo de entrada")
    parser.add_argument("rosto", help="Caminho da foto do rosto de referência")
    parser.add_argument("--saida", default=None, help="Caminho do vídeo de saída")
    parser.add_argument(
        "--device",
        choices=[d.value for d in DevicePreference],
        default="auto",
        help="Dispositivo de execução",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Ativa restauração e super-resolução facial (GFPGAN)",
    )
    args = parser.parse_args()

    print(f"=== FaceLab — Execução de Face Swap ===")
    print(f"Vídeo de entrada: {args.video}")
    print(f"Rosto de origem : {args.rosto}")
    print(f"Dispositivo     : {args.device}")
    print(f"Restauração GFPGAN: {'Ativada' if args.enhance else 'Desativada'}")

    try:
        pipeline = VideoPipeline(
            device_preference=DevicePreference(args.device),
            on_progress=on_progress,
            enable_enhancer=args.enhance,
        )
        print(f"Dispositivo selecionado: {pipeline.device_manager.info.name}")
        out_file = pipeline.process(
            video_path=args.video,
            source_image_path=args.rosto,
            output_path=args.saida,
        )
        print(f"\n\n[OK] Video processado com sucesso!")
        print(f"Arquivo gerado: {out_file}")
        return 0
    except Exception as exc:
        print(f"\n\n[ERRO] Erro durante o processamento: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

