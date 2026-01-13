# scripts/composite_segments.py

from __future__ import annotations

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


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "assets").exists() and (p / "output").exists():
            return p
    return start


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stdout}")


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
    ap.add_argument("--segments_mp4_dir", default="output/segments_mp4")
    ap.add_argument("--lipsync_dir", default="output/segments_lipsync_crop")
    ap.add_argument("--outdir", default="output/segments_composited")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument(
        "--ids",
        default="",
        help="Processar apenas alguns IDs (ex.: '1,2,5-8'). IDs são os do segments.json",
    )
    args = ap.parse_args()

    root = find_project_root(Path.cwd())
    seg_path = (root / args.segments).resolve()
    mp4_dir = (root / args.segments_mp4_dir).resolve()
    lipsync_dir = (root / args.lipsync_dir).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json not found: {seg_path}")
    if not mp4_dir.exists():
        raise FileNotFoundError(f"segments_mp4_dir not found: {mp4_dir}")
    if not lipsync_dir.exists():
        raise FileNotFoundError(f"lipsync_dir not found: {lipsync_dir}")

    segs = load_segments(seg_path)

    wanted_ids: set[int] = set()
    if args.ids.strip():
        for part in args.ids.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                wanted_ids.update(range(int(a), int(b) + 1))
            else:
                wanted_ids.add(int(part))

    total = len(segs)
    max_n = total if args.limit == 0 else min(args.limit, total)

    ok = 0
    skipped = 0
    failed = 0
    missing = 0

    for i in range(max_n):
        s = segs[i]
        sid_int = int(s["id"])
        if wanted_ids and sid_int not in wanted_ids:
            continue

        sid = f"{sid_int:04d}"
        speaker = str(s["logical_speaker"])
        face_id = str(s["face_id"])

        if face_id not in BBOXES:
            known = ", ".join(sorted(BBOXES.keys()))
            raise KeyError(f"Unknown face_id '{face_id}' for segment id={sid_int}. Known: {known}")

        x1, y1, x2, y2 = BBOXES[face_id]
        w = x2 - x1
        h = y2 - y1

        # Inputs esperados
        base_mp4 = mp4_dir / f"{sid}_{speaker}_static_with_audio.mp4"
        if not base_mp4.exists():
            # fallback para naming antigo (se existir)
            base_mp4_alt = mp4_dir / f"{sid}_static_with_audio.mp4"
            base_mp4 = base_mp4_alt

        lipsync_mp4 = lipsync_dir / f"{sid}_{speaker}_lipsync_crop.mp4"
        out_mp4 = outdir / f"{sid}_{speaker}_composited.mp4"

        if args.skip_existing and out_mp4.exists() and not args.overwrite:
            print(f"[SKIP] {out_mp4.name}")
            skipped += 1
            continue

        if not base_mp4.exists():
            print(f"[MISS] base mp4 not found: {base_mp4}")
            missing += 1
            continue
        if not lipsync_mp4.exists():
            print(f"[MISS] lipsync crop not found: {lipsync_mp4}")
            missing += 1
            continue

        # Overlay: scale 512->(w,h) e cola em (x1,y1).
        # -map 0:a? preserva áudio do base (já é o áudio do segmento).
        # -shortest evita estourar duração.
        cmd = [
            "ffmpeg",
            "-y" if args.overwrite else "-n",
            "-i",
            str(base_mp4),
            "-i",
            str(lipsync_mp4),
            "-filter_complex",
            f"[1:v]scale={w}:{h}[ov];[0:v][ov]overlay={x1}:{y1}:shortest=1[v]",
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(int(args.fps)),
            "-c:a",
            "copy",
            str(out_mp4),
        ]

        try:
            run(cmd)
            ok += 1
            print(f"[OK] {out_mp4.name} (face_id={face_id} box=({x1},{y1},{x2},{y2}))")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {sid}_{speaker}: {e}")

    print("\n=== Summary ===")
    print(f"Processed OK: {ok}")
    print(f"Skipped:      {skipped}")
    print(f"Missing:      {missing}")
    print(f"Failed:       {failed}")
    print(f"Segments read: {total} (limit={args.limit})")


if __name__ == "__main__":
    main()
