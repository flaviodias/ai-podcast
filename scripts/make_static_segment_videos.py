# scripts/make_static_segment_videos.py

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
    ap.add_argument("--outdir", default="output/segments_mp4", help="Output directory for MP4s")
    ap.add_argument("--image", default="assets/master_scene.jpg", help="Master scene image path")
    ap.add_argument("--fps", type=int, default=25, help="Output FPS (must match project config)")
    ap.add_argument("--audio_codec", default="aac", help="Audio codec for MP4 (default: aac)")
    ap.add_argument("--vcodec", default="libx264", help="Video codec (default: libx264)")
    ap.add_argument("--pix_fmt", default="yuv420p", help="Pixel format (default: yuv420p)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all, otherwise process N files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    wavdir = Path(args.wavdir)
    outdir = Path(args.outdir)
    image_path = Path(args.image)

    if not wavdir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wavdir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Master image not found: {image_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(wavdir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in: {wavdir}")

    max_n = len(wavs) if args.limit == 0 else min(args.limit, len(wavs))

    for i, wav_path in enumerate(wavs[:max_n], start=1):
        dur = ffprobe_duration_seconds(wav_path)
        base = wav_path.stem
        out_mp4 = outdir / f"{base}_static_with_audio.mp4"

        if out_mp4.exists() and not args.overwrite:
            print(f"[{i}/{max_n}] Skip (exists): {out_mp4}")
            continue

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
            "-c:v",
            args.vcodec,
            "-pix_fmt",
            args.pix_fmt,
            "-tune",
            "stillimage",
            "-c:a",
            args.audio_codec,
            "-shortest",
            str(out_mp4),
        ]

        if args.dry_run:
            print(f"[{i}/{max_n}] DRY: {' '.join(cmd)}")
        else:
            run(cmd)
            print(f"[{i}/{max_n}] Wrote: {out_mp4} (dur={dur:.3f}s)")

    print(f"Done. Processed {max_n} file(s). Output: {outdir}")


if __name__ == "__main__":
    main()
