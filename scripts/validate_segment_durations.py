# scripts/validate_segment_durations.py

import argparse
import subprocess
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavdir", default="output/segments", help="Directory containing segment WAVs")
    ap.add_argument(
        "--mp4dir",
        default="output/segments_mp4",
        help="Directory containing segment MP4s (*_static_with_audio.mp4)",
    )
    ap.add_argument(
        "--mp4-suffix",
        default="_static_with_audio.mp4",
        help="Suffix appended to WAV stem to form MP4 filename",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.050,
        help="Max acceptable absolute duration delta in seconds (default 0.050)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all, otherwise validate N files (in sorted order)",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first unacceptable delta or missing file",
    )
    args = ap.parse_args()

    wavdir = Path(args.wavdir)
    mp4dir = Path(args.mp4dir)

    if not wavdir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wavdir}")
    if not mp4dir.exists():
        raise FileNotFoundError(f"MP4 directory not found: {mp4dir}")

    wavs = sorted(wavdir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in: {wavdir}")

    max_n = len(wavs) if args.limit == 0 else min(args.limit, len(wavs))

    bad: list[tuple[str, float, float, float]] = []
    missing: list[str] = []

    for i, wav_path in enumerate(wavs[:max_n], start=1):
        mp4_path = mp4dir / f"{wav_path.stem}{args.mp4_suffix}"

        if not mp4_path.exists():
            missing.append(str(mp4_path))
            msg = f"[{i}/{max_n}] MISSING: {mp4_path}"
            print(msg)
            if args.fail_fast:
                break
            continue

        wav_dur = ffprobe_duration_seconds(wav_path)
        mp4_dur = ffprobe_duration_seconds(mp4_path)
        delta = abs(mp4_dur - wav_dur)

        if delta > args.tolerance:
            bad.append((wav_path.stem, wav_dur, mp4_dur, delta))
            msg = (
                f"[{i}/{max_n}] BAD: {wav_path.stem} "
                f"wav={wav_dur:.3f}s mp4={mp4_dur:.3f}s delta={delta:.3f}s"
            )
            print(msg)
            if args.fail_fast:
                break
        else:
            print(f"[{i}/{max_n}] OK : {wav_path.stem} delta={delta:.3f}s")

    print("\n=== Summary ===")
    print(f"Validated: {min(i, max_n) if 'i' in locals() else 0}/{max_n}")
    print(f"Missing MP4: {len(missing)}")
    print(f"Out of tolerance: {len(bad)} (tolerance={args.tolerance:.3f}s)")

    if missing:
        print("\nMissing files:")
        for p in missing[:50]:
            print(f" - {p}")
        if len(missing) > 50:
            print(f" ... +{len(missing) - 50} more")

    if bad:
        print("\nOut-of-tolerance files:")
        for stem, wav_dur, mp4_dur, delta in bad[:50]:
            print(f" - {stem}: wav={wav_dur:.3f}s mp4={mp4_dur:.3f}s delta={delta:.3f}s")
        if len(bad) > 50:
            print(f" ... +{len(bad) - 50} more")

    if not missing and not bad:
        print("\nLOTE VALIDADO: todas as durações estão dentro da tolerância.")
    else:
        # Exit code non-zero is useful for CI/scripts, but won't break interactive usage
        raise SystemExit(1)


if __name__ == "__main__":
    main()
