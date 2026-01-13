# scripts/inspect_wav2lip_bbox.py

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _add_wav2lip_to_syspath(wav2lip_dir: Path) -> None:
    wav2lip_dir = wav2lip_dir.resolve()
    if not wav2lip_dir.exists():
        raise FileNotFoundError(f"Wav2Lip dir not found: {wav2lip_dir}")

    # Permite imports como: import face_detection
    if str(wav2lip_dir) not in sys.path:
        sys.path.insert(0, str(wav2lip_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", required=True, help="MP4 crop (ex.: 512x512)")
    ap.add_argument("--out", default="output/debug_bbox.png")
    ap.add_argument(
        "--pads",
        nargs=4,
        type=int,
        default=[0, 32, 0, 0],
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Padding aplicado sobre o bbox do detector",
    )
    ap.add_argument(
        "--wav2lip_dir",
        default="third_party/Wav2Lip",
        help="Path para o repo Wav2Lip (para importar face_detection)",
    )
    args = ap.parse_args()

    wav2lip_dir = Path(args.wav2lip_dir)
    _add_wav2lip_to_syspath(wav2lip_dir)

    # Importa depois de adicionar ao sys.path
    import face_detection  # type: ignore

    face_path = Path(args.face)
    if not face_path.exists():
        raise FileNotFoundError(face_path)

    cap = cv2.VideoCapture(str(face_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Não consegui ler o 1º frame do vídeo.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device
    )

    rect = detector.get_detections_for_batch(np.array([frame]))[0]
    if rect is None:
        raise RuntimeError("Face não detectada no 1º frame.")

    # rect = [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, rect)
    pad_top, pad_bottom, pad_left, pad_right = args.pads

    y1p = max(0, y1 - pad_top)
    y2p = min(frame.shape[0], y2 + pad_bottom)
    x1p = max(0, x1 - pad_left)
    x2p = min(frame.shape[1], x2 + pad_right)

    dbg = frame.copy()
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)  # bbox detector (amarelo)
    cv2.rectangle(dbg, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)  # bbox + pads (verde)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), dbg)

    print("Detector rect (x1,y1,x2,y2):", (x1, y1, x2, y2))
    print("With pads rect (x1,y1,x2,y2):", (x1p, y1p, x2p, y2p))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
