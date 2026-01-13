# scripts/concat_final.py

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout


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
    return segs


def resolve_composited(
    composited_dir: Path,
    seg_id: int,
    logical_speaker: str,
    suffix: str,
) -> Path:
    """
    Primary: <id:04d>_<logical_speaker><suffix>
    Fallback: unique match by prefix <id:04d>_ and suffix.
    """
    sid = f"{seg_id:04d}"
    preferred = composited_dir / f"{sid}_{logical_speaker}{suffix}"
    if preferred.exists():
        return preferred

    # Fallback: try to find exactly one file matching the segment id
    candidates = sorted(composited_dir.glob(f"{sid}_*{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Missing composited file for segment id={sid}. "
            f"Tried: {preferred} and glob {sid}_*{suffix}"
        )
    raise RuntimeError(
        f"Ambiguous composited files for segment id={sid}: "
        + ", ".join(str(c.name) for c in candidates[:10])
        + (f" ... (+{len(candidates) - 10} more)" if len(candidates) > 10 else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="output/diarization/segments.json")
    ap.add_argument("--composited-dir", default="output/segments_composited")
    ap.add_argument("--suffix", default="_composited.mp4")
    ap.add_argument("--out", default="output/final/podcast.mp4")
    ap.add_argument(
        "--reencode",
        action="store_true",
        help="If set, re-encode (slower) instead of stream copy. Use if ffmpeg concat -c copy fails.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all, otherwise concat only first N segments (in segments.json order).",
    )
    args = ap.parse_args()

    seg_path = Path(args.segments)
    composited_dir = Path(args.composited_dir)
    out_path = Path(args.out)

    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json not found: {seg_path}")
    if not composited_dir.exists():
        raise FileNotFoundError(f"composited dir not found: {composited_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    segments = load_segments(seg_path)

    # Schema expectations (your current pipeline)
    for k in ("id", "logical_speaker"):
        if k not in segments[0]:
            keys = sorted(segments[0].keys())
            raise KeyError(f"Missing key '{k}' in segments items. First keys: {keys}")

    max_n = len(segments) if args.limit == 0 else min(args.limit, len(segments))

    # Build concat list in deterministic order (segments.json order)
    concat_list = out_path.parent / "concat_list.txt"
    lines: list[str] = []

    for i in range(max_n):
        seg = segments[i]
        seg_id = int(seg["id"])
        logical_speaker = str(seg["logical_speaker"])
        mp4 = resolve_composited(composited_dir, seg_id, logical_speaker, args.suffix)

        # concat demuxer expects: file '<path>'
        # Use absolute paths to avoid cwd surprises
        lines.append(f"file '{mp4.resolve()}'")

    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote concat list: {concat_list} ({max_n} entries)")

    # Concat with ffmpeg concat demuxer
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
    ]

    if args.reencode:
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    else:
        cmd += ["-c", "copy"]

    cmd += [str(out_path)]

    run(cmd)
    print(f"OK: wrote final video: {out_path}")

    # Optional sanity check
    dur = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(out_path),
        ]
    ).strip()
    print(f"Final duration: {float(dur):.3f}s")


if __name__ == "__main__":
    main()
