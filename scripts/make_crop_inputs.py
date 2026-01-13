# scripts/make_crop_inputs.py

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

# BBoxes SAFE já validadas no master_scene 2816x1536
# format: (x1, y1, x2, y2)
BBOXES: dict[str, tuple[int, int, int, int]] = {
    "left_face": (751, 238, 1343, 950),
    "right_face": (1471, 331, 2028, 964),
}


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout


def ffprobe_duration_seconds(media_path: Path) -> float:
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(media_path),
        ]
    ).strip()
    return float(out)


def load_segments(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        segs = raw
    elif isinstance(raw, dict) and isinstance(raw.get("segments"), list):
        segs = raw["segments"]
    else:
        raise ValueError(
            f"Unsupported segments format: {path}. Expected list or dict with 'segments' list."
        )

    if not segs:
        raise ValueError(f"No segments found in {path}.")
    if not isinstance(segs[0], dict):
        raise ValueError(f"Segment items must be dicts; got {type(segs[0]).__name__}.")
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="output/diarization/segments.json")
    ap.add_argument("--wavdir", default="output/segments")
    ap.add_argument("--outdir", default="output/segments_crop")
    ap.add_argument("--image", default="assets/master_scene.jpg")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--target", type=int, default=512, help="Crop output size (NxN)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seg_path = Path(args.segments)
    wavdir = Path(args.wavdir)
    outdir = Path(args.outdir)
    image_path = Path(args.image)

    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json not found: {seg_path}")
    if not wavdir.exists():
        raise FileNotFoundError(f"wavdir not found: {wavdir}")
    if not image_path.exists():
        raise FileNotFoundError(f"master image not found: {image_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    segments = load_segments(seg_path)

    # Deterministic schema checks
    for k in ("logical_speaker", "face_id"):
        if k not in segments[0]:
            keys = sorted(segments[0].keys())
            raise KeyError(f"Missing key '{k}' in segments items. First keys: {keys}")

    wavs = sorted(wavdir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAVs found in {wavdir}")

    # Assumption (consistent with your current pipeline):
    # wav filenames are in the same order as segments.json, using 0001_, 0002_, ...
    max_n = len(segments) if args.limit == 0 else min(args.limit, len(segments))

    if len(wavs) < max_n:
        raise ValueError(
            f"wav count ({len(wavs)}) is less than segments to process ({max_n}). "
            "Check naming/order in output/segments."
        )

    for i in range(max_n):
        seg = segments[i]

        face_id = str(seg["face_id"])
        if face_id not in BBOXES:
            known = ", ".join(sorted(BBOXES.keys()))
            raise KeyError(f"Unknown face_id '{face_id}' at segment index {i + 1}. Known: {known}")

        x1, y1, x2, y2 = BBOXES[face_id]
        w = x2 - x1
        h = y2 - y1

        wav_path = wavs[i]
        dur = ffprobe_duration_seconds(wav_path)

        base = wav_path.stem
        out_mp4 = outdir / f"{base}_crop.mp4"

        if out_mp4.exists() and not args.overwrite:
            print(f"[{i + 1:04d}/{max_n:04d}] Skip (exists): {out_mp4}")
            continue

        vf = f"crop={w}:{h}:{x1}:{y1},scale={args.target}:{args.target}"

        cmd = [
            "ffmpeg",
            "-y" if args.overwrite else "-n",
            "-loop",
            "1",
            "-framerate",
            str(args.fps),
            "-i",
            str(image_path),
            "-i",
            str(wav_path),
            "-t",
            f"{dur:.3f}",
            "-r",
            str(args.fps),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-shortest",
            str(out_mp4),
        ]

        if args.dry_run:
            print(f"[{i + 1:04d}/{max_n:04d}] DRY: {' '.join(cmd)}")
        else:
            run(cmd)
            speaker = seg.get("logical_speaker")
            print(
                f"[{i + 1:04d}/{max_n:04d}] Wrote: {out_mp4} "
                f"(face_id={face_id}, dur={dur:.3f}s, speaker={speaker})"
            )

    print(f"Done. Output: {outdir}")


if __name__ == "__main__":
    main()
