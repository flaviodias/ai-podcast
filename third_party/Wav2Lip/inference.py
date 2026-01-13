import argparse
import os
import platform
import subprocess
from typing import Any

import audio
import cv2
import face_detection
import numpy as np
import torch
from models import Wav2Lip
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)

parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)

parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)

parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right). Adjust to include chin at least.",
)

parser.add_argument(
    "--face_det_batch_size",
    type=int,
    help="Batch size for face detection",
    default=16,
)

parser.add_argument(
    "--wav2lip_batch_size",
    type=int,
    help="Batch size for Wav2Lip model(s)",
    default=128,
)

parser.add_argument(
    "--resize_factor",
    default=1,
    type=int,
    help="Reduce resolution by this factor. Often best at 480p/720p.",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help=(
        "Crop video to smaller region (top, bottom, left, right). "
        "Applied after resize_factor and rotate. -1 auto-infers."
    ),
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help=(
        "Specify a constant bounding box for the face (top, bottom, left, right). "
        "Use only as last resort if detection fails."
    ),
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Rotate video 90deg clockwise (use if input is flipped).",
)

parser.add_argument(
    "--nosmooth",
    default=False,
    action="store_true",
    help="Prevent smoothing face detections over temporal window.",
)

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split(".")[-1].lower() in {"jpg", "png", "jpeg"}:
    args.static = True

mel_step_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for inference.")


def get_smoothened_boxes(boxes: np.ndarray, t: int) -> np.ndarray:
    for i in range(len(boxes)):
        if i + t > len(boxes):
            window = boxes[len(boxes) - t :]
        else:
            window = boxes[i : i + t]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images: list[np.ndarray]) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D,
        flip_input=False,
        device=device,
    )

    batch_size = args.face_det_batch_size

    while True:
        predictions: list[Any] = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                batch = np.array(images[i : i + batch_size])
                predictions.extend(detector.get_detections_for_batch(batch))
        except RuntimeError as err:
            if batch_size == 1:
                raise RuntimeError(
                    "Image too big to run face detection on GPU. Use --resize_factor."
                ) from err
            batch_size //= 2
            print(f"Recovering from OOM error; New batch size: {batch_size}")
            continue
        break

    results: list[list[int]] = []
    pady1, pady2, padx1, padx2 = args.pads

    for rect, image in zip(predictions, images, strict=False):
        if rect is None:
            os.makedirs("temp", exist_ok=True)
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError("Face not detected in one or more frames.")

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, t=5)

    out: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for image, (x1, y1, x2, y2) in zip(images, boxes, strict=False):
        face = image[y1:y2, x1:x2]
        out.append((face, (y1, y2, x1, x2)))

    del detector
    return out


def datagen(
    frames: list[np.ndarray],
    mels: list[np.ndarray],
) -> Any:
    img_batch: list[np.ndarray] = []
    mel_batch: list[np.ndarray] = []
    frame_batch: list[np.ndarray] = []
    coords_batch: list[tuple[int, int, int, int]] = []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = args.box
        face_det_results = [(f[y1:y2, x1:x2], (y1, y2, x1, x2)) for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()

        face, coords = face_det_results[idx]
        face = face.copy()

        if face.size == 0:
            raise ValueError(f"Empty face crop. Check --pads / --box / input crop. coords={coords}")

        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            yield _make_batch(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if img_batch:
        yield _make_batch(img_batch, mel_batch, frame_batch, coords_batch)


def _make_batch(
    img_batch: list[np.ndarray],
    mel_batch: list[np.ndarray],
    frame_batch: list[np.ndarray],
    coords_batch: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[int, int, int, int]]]:
    img_arr = np.asarray(img_batch)
    mel_arr = np.asarray(mel_batch)

    img_masked = img_arr.copy()
    img_masked[:, args.img_size // 2 :] = 0

    img_out = np.concatenate((img_masked, img_arr), axis=3) / 255.0
    mel_out = np.reshape(mel_arr, [len(mel_arr), mel_arr.shape[1], mel_arr.shape[2], 1])

    return img_out, mel_out, frame_batch, coords_batch


def _load(checkpoint_path: str) -> Any:
    # Try TorchScript (Drive .pt)
    try:
        return torch.jit.load(checkpoint_path, map_location=device)
    except Exception:
        # Classic checkpoints (.pth) with state_dict
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_model(path: str) -> Any:
    print(f"Load checkpoint from: {path}")
    checkpoint = _load(path)

    # TorchScript: already a model
    if not isinstance(checkpoint, dict):
        model = checkpoint
        try:
            model = model.to(device)
        except Exception:
            pass
        model.eval()
        return model

    # Classic: dict with state_dict
    model = Wav2Lip()
    s = checkpoint["state_dict"]

    new_s: dict[str, Any] = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v

    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()


def main() -> None:
    if not os.path.isfile(args.face):
        raise ValueError("--face must be a valid path to video/image file")

    face_ext = args.face.split(".")[-1].lower()
    if face_ext in {"jpg", "png", "jpeg"}:
        img = cv2.imread(args.face)
        if img is None:
            raise ValueError("Failed to read input image with cv2.imread")
        full_frames = [img]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print("Reading video frames...")

        full_frames: list[np.ndarray] = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor),
                )

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # type: ignore[attr-defined]

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    if not full_frames:
        raise ValueError("No frames found in input --face.")

    print(f"Number of frames available for inference: {len(full_frames)}")

    if not args.audio.endswith(".wav"):
        print("Extracting raw audio...")
        os.makedirs("temp", exist_ok=True)
        command = f"ffmpeg -y -i {args.audio} -strict -2 temp/temp.wav"
        subprocess.call(command, shell=True)
        args.audio = "temp/temp.wav"

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError("Mel contains NaN. If using TTS, add small epsilon noise and retry.")

    mel_chunks: list[np.ndarray] = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Length of mel chunks: {len(mel_chunks)}")
    full_frames = full_frames[: len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    out = None
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
    ):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:2]
            os.makedirs("temp", exist_ok=True)
            out = cv2.VideoWriter(
                "temp/result.avi",
                cv2.VideoWriter_fourcc(*"DIVX"),  # type: ignore[attr-defined]
                fps,
                (frame_w, frame_h),
            )

        assert out is not None

        img_batch_t = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch_t = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch_t, img_batch_t)

        pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred_np, frames, coords, strict=False):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    if out is None:
        raise RuntimeError("VideoWriter was not created; no output generated.")
    out.release()

    command = f"ffmpeg -y -i {args.audio} -i temp/result.avi -strict -2 -q:v 1 {args.outfile}"
    subprocess.call(command, shell=platform.system() != "Windows")


if __name__ == "__main__":
    main()
