"""Script de Execução e Teste de Face Swap ao Vivo pela Webcam.

Uso:
    python scripts/run_webcam.py "temp/rosto.png"
    python scripts/run_webcam.py "temp/rosto.png" --camera 0 --width 1280 --height 720
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.device_manager import DevicePreference  # noqa: E402
from src.pipeline.live_pipeline import LivePipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rosto", help="Caminho da foto do rosto de referência")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Largura da captura (padrão: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Altura da captura (padrão: 720)")
    parser.add_argument(
        "--device",
        choices=[d.value for d in DevicePreference],
        default="auto",
        help="Dispositivo de execução",
    )
    args = parser.parse_args()

    print("=== FaceLab — Live Face Swap (Webcam) ===")
    print(f"Rosto de referência: {args.rosto}")
    print(f"Câmera: {args.camera} ({args.width}x{args.height})")
    print("Pressione 'Q' ou 'ESC' na janela de vídeo para sair.")

    window_name = "FaceLab — Live Webcam (Pressione Q para sair)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_frame(rendered_frame: np.ndarray, fps: float) -> None:
        # Desenha indicador de status ao vivo
        h, w = rendered_frame.shape[:2]
        hud_text = f"FaceLab Live | {fps:.1f} FPS | [Q] Sair"
        cv2.putText(
            rendered_frame,
            hud_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, rendered_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q ou ESC
            pipeline.stop()

    try:
        pipeline = LivePipeline(
            device_preference=DevicePreference(args.device),
            on_frame_rendered=on_frame,
        )
        pipeline.run(
            source_image_path=args.rosto,
            camera_index=args.camera,
            width=args.width,
            height=args.height,
        )
        cv2.destroyAllWindows()
        print("\nTransmissão ao vivo encerrada.")
        return 0
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"\n❌ Erro na execução da webcam: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

