"""Fase 3 — detecção facial em imagem ou vídeo, com caixas desenhadas.

Uso:
    python scripts/detect_faces.py "temp\\video.mp4"          # vídeo anotado
    python scripts/detect_faces.py foto.jpg                   # imagem anotada
    python scripts/detect_faces.py video.mp4 --device cpu     # comparar CPU
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config.settings import OUTPUT_DIR, SUPPORTED_IMAGE_EXTENSIONS  # noqa: E402
from src.core.types import DetectedFace  # noqa: E402
from src.face.face_detector import InsightFaceDetector  # noqa: E402
from src.inference.device_manager import DeviceManager, DevicePreference  # noqa: E402
from src.utils.validators import validate_image_path, validate_video_path  # noqa: E402
from src.video.frame_source import VideoFileSource  # noqa: E402
from src.video.video_writer import VideoFileSink  # noqa: E402

ANNOTATION_NOTE = "Video anotado pelo FaceLab (caixas de deteccao facial)."


def draw(image: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
    out = image.copy()
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"#{i} {f.score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if f.landmarks is not None:
            for x, y in f.landmarks.astype(int):
                cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada")
    parser.add_argument("--device", choices=[p.value for p in DevicePreference], default="auto")
    args = parser.parse_args()

    dm = DeviceManager(DevicePreference(args.device))
    print(f"Dispositivo: {dm.info.name}")
    t_load = time.perf_counter()
    detector = InsightFaceDetector(dm)
    print(f"Modelo carregado em {time.perf_counter() - t_load:.1f}s. "
          f"Providers ativos: {detector.active_providers}")

    path = Path(args.entrada)
    if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        src = validate_image_path(path)
        image = cv2.imread(str(src))
        if image is None:
            print("Não foi possível ler a imagem.")
            return 1
        faces = detector.detect(image)
        out = OUTPUT_DIR / f"{src.stem}_faces.png"
        cv2.imwrite(str(out), draw(image, faces))
        print(f"{len(faces)} rosto(s). Resultado: {out}")
        return 0

    src = validate_video_path(path)
    source = VideoFileSource(src)
    info = source.info
    out = OUTPUT_DIR / f"{src.stem}_faces.mp4"
    sink = VideoFileSink(out, info.width, info.height, info.fps,
                         audio_source=src if info.has_audio else None,
                         disclosure=ANNOTATION_NOTE)
    counts: list[int] = []
    det_time = 0.0
    with source, sink:
        for frame in source.frames():
            t0 = time.perf_counter()
            faces = detector.detect(frame.image)
            det_time += time.perf_counter() - t0
            counts.append(len(faces))
            sink.write(frame.__class__(frame.index, frame.timestamp_s, draw(frame.image, faces)))
            if frame.index % 30 == 0:
                print(f"  quadro {frame.index}: {len(faces)} rosto(s)")

    arr = np.array(counts)
    print("\n=== Relatório ===")
    print(f"Quadros: {len(arr)} | sem rosto: {(arr == 0).sum()} | "
          f"1 rosto: {(arr == 1).sum()} | 2+ rostos: {(arr >= 2).sum()}")
    print(f"Detecção: {len(arr) / det_time:.1f} quadros/s "
          f"({1000 * det_time / len(arr):.1f} ms/quadro)")
    if dm.info.is_cuda:
        import torch
        print(f"VRAM (PyTorch): {torch.cuda.memory_allocated() / 1024**2:.0f} MB "
              "(o ONNX Runtime usa memória própria; veja nvidia-smi)")
    print(f"Resultado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

