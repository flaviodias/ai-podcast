# scripts/cut_segments.py

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stdout}")


def _load_segments(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        segments = raw
    elif isinstance(raw, dict) and isinstance(raw.get("segments"), list):
        segments = raw["segments"]
    else:
        raise ValueError(
            f"Unsupported segments.json format in {path}. "
            "Expected a list, or a dict with key 'segments' containing a list."
        )

    if not segments:
        raise ValueError(f"No segments found in {path} (empty list).")

    if not isinstance(segments[0], dict):
        raise ValueError(
            f"Unsupported segment item type: {type(segments[0]).__name__}. Expected dict objects."
        )

    return segments


def _safe_name(s: str) -> str:
    # Keep filenames stable across platforms
    return "".join(c if (c.isalnum() or c in ("_", "-", ".")) else "_" for c in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="output/diarization/segments.json")
    ap.add_argument("--audio", default="assets/audio_full.wav")
    ap.add_argument("--outdir", default="output/segments")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--use-id",
        action="store_true",
        help="Use segment 'id' (if present) for numbering instead of sequential index.",
    )
    args = ap.parse_args()

    seg_path = Path(args.segments)
    audio_path = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json not found: {seg_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio master not found: {audio_path}")

    segments = _load_segments(seg_path)

    n = len(segments)
    max_n = n if args.limit == 0 else min(args.limit, n)

    # Deterministic schema validation (fail fast with actionable info)
    required = ("start_sec", "end_sec")
    for k in required:
        if k not in segments[0]:
            raise KeyError(
                f"Missing required key '{k}' in segments.json items. "
                f"First item keys: {sorted(segments[0].keys())}"
            )

    for i in range(max_n):
        s = segments[i]

        start = float(s["start_sec"])
        end = float(s["end_sec"])

        # Prefer explicit duration when present, else derive
        dur_val = s.get("duration_sec")
        if dur_val is not None:
            dur = float(dur_val)
        else:
            dur = end - start
        dur = max(0.0, dur)

        speaker = str(
            s.get("logical_speaker") or s.get("speaker") or s.get("speaker_label") or "Speaker_?"
        )
        speaker = _safe_name(speaker)

        if args.use_id and s.get("id") is not None:
            idx_num = int(s["id"])
            idx = f"{idx_num:04d}"
        else:
            idx = f"{i + 1:04d}"

        out_wav = outdir / f"{idx}_{speaker}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-ac",
            str(args.channels),
            "-ar",
            str(args.sr),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
        run(cmd)
        print(
            f"Wrote: {out_wav}  start={start:.3f} dur={dur:.3f} "
            f"speaker={speaker} face_id={s.get('face_id')}"
        )

    print(f"Done. Generated {max_n}/{n} segments into {outdir}")


if __name__ == "__main__":
    main()
