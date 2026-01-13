# scripts/run_wav2lip_batch.py

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for p in candidates:
        if (p / "assets").exists() and (p / "output").exists() and (p / "third_party").exists():
            return p
    # fallback: se ao menos existir third_party/Wav2Lip
    for p in candidates:
        if (p / "third_party" / "Wav2Lip" / "inference.py").exists():
            return p
    raise FileNotFoundError(
        "Não consegui localizar a raiz do projeto. Rode a partir de ~/workspace/projects/ai-podcast "
        "ou passe --project_root."
    )


def run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}):\n" + " ".join(cmd))


def parse_ids(ids_str: str) -> set[int]:
    # Aceita: "1,2,5-8,10"
    out: set[int] = set()
    for part in ids_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def pads_from_string(s: str) -> list[int]:
    parts = [int(x) for x in s.strip().split()]
    if len(parts) != 4:
        raise ValueError("Pads devem ter 4 ints: 'TOP BOTTOM LEFT RIGHT'")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="", help="Raiz do projeto (auto se vazio)")
    ap.add_argument(
        "--segments",
        default="output/diarization/segments.json",
        help="Relativo à raiz do projeto",
    )
    ap.add_argument(
        "--wav2lip_dir",
        default="third_party/Wav2Lip",
        help="Relativo à raiz do projeto",
    )
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/Wav2Lip-SD-NOGAN.pt",
        help="Relativo a wav2lip_dir",
    )
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--static", action="store_true", default=True)
    ap.add_argument(
        "--pads_right",
        default="0 16 0 0",
        help="Pads para right_face: 'TOP BOTTOM LEFT RIGHT'",
    )
    ap.add_argument(
        "--pads_left",
        default="0 16 0 0",
        help="Pads para left_face: 'TOP BOTTOM LEFT RIGHT'",
    )
    ap.add_argument(
        "--segments_crop_dir",
        default="output/segments_crop",
        help="Relativo à raiz do projeto",
    )
    ap.add_argument(
        "--segments_audio_dir",
        default="output/segments",
        help="Relativo à raiz do projeto",
    )
    ap.add_argument(
        "--outdir",
        default="output/segments_lipsync_crop",
        help="Relativo à raiz do projeto",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = todos")
    ap.add_argument(
        "--ids",
        default="",
        help="Processar apenas alguns IDs (ex.: '1,2,5-8'). IDs são os do segments.json",
    )
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path.cwd())

    seg_path = (root / args.segments).resolve()
    wav2lip_dir = (root / args.wav2lip_dir).resolve()
    infer_py = wav2lip_dir / "inference.py"
    ckpt = (wav2lip_dir / args.checkpoint).resolve()

    crop_dir = (root / args.segments_crop_dir).resolve()
    audio_dir = (root / args.segments_audio_dir).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json não encontrado: {seg_path}")
    if not infer_py.exists():
        raise FileNotFoundError(f"inference.py não encontrado: {infer_py}")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint não encontrado: {ckpt}")
    if not crop_dir.exists():
        raise FileNotFoundError(f"segments_crop_dir não existe: {crop_dir}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"segments_audio_dir não existe: {audio_dir}")

    pads_right = pads_from_string(args.pads_right)
    pads_left = pads_from_string(args.pads_left)

    segs: list[dict[str, Any]] = json.loads(seg_path.read_text(encoding="utf-8"))

    only_ids: set[int] = parse_ids(args.ids) if args.ids.strip() else set()

    total = len(segs)
    max_n = total if args.limit == 0 else min(args.limit, total)

    ok = 0
    skipped = 0
    failed = 0

    print(f"Project root: {root}")
    print(f"Segments: {seg_path} (total={total})")
    print(f"Wav2Lip: {wav2lip_dir}")
    print(f"Checkpoint: {ckpt}")
    print(f"Outdir: {outdir}")
    print(f"Pads right_face: {pads_right}")
    print(f"Pads left_face:  {pads_left}")

    for i in range(max_n):
        s = segs[i]
        sid_int = int(s["id"])
        if only_ids and sid_int not in only_ids:
            continue

        sid = f"{sid_int:04d}"
        speaker = str(s.get("logical_speaker", "Speaker_?"))
        face_id = str(s.get("face_id", ""))

        crop_mp4 = crop_dir / f"{sid}_{speaker}_crop.mp4"
        wav = audio_dir / f"{sid}_{speaker}.wav"
        out_mp4 = outdir / f"{sid}_{speaker}_lipsync_crop.mp4"

        if args.skip_existing and out_mp4.exists():
            print(f"[SKIP] {sid} {speaker} (já existe) -> {out_mp4.name}")
            skipped += 1
            continue

        if not crop_mp4.exists():
            print(f"[MISS] {sid} {speaker} crop não encontrado: {crop_mp4}")
            failed += 1
            continue
        if not wav.exists():
            print(f"[MISS] {sid} {speaker} wav não encontrado: {wav}")
            failed += 1
            continue

        pads = pads_right if face_id == "right_face" else pads_left

        cmd = [
            sys.executable,
            str(infer_py),
            "--checkpoint_path",
            str(ckpt),
            "--face",
            str(crop_mp4),
            "--audio",
            str(wav),
            "--outfile",
            str(out_mp4),
            "--fps",
            str(args.fps),
            "--pads",
            str(pads[0]),
            str(pads[1]),
            str(pads[2]),
            str(pads[3]),
        ]
        if args.static:
            cmd += ["--static", "True"]

        print(f"[RUN ] {sid} {speaker} {face_id} pads={pads} -> {out_mp4.name}")
        try:
            run(cmd, cwd=wav2lip_dir)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {sid} {speaker}: {e}")

    print("\n=== Summary ===")
    print(f"Processed OK: {ok}")
    print(f"Skipped:      {skipped}")
    print(f"Failed:       {failed}")
    print(f"Segments read: {total} (limit={args.limit})")


if __name__ == "__main__":
    main()
