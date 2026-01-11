# src/aipodcast/video.py

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class VideoError(RuntimeError):
    """Erro ao executar operações de vídeo/ffmpeg."""


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoError(
            "ffmpeg não encontrado no PATH. Instale com: sudo apt install ffmpeg "
            "ou ajuste seu PATH para incluir o binário."
        )
    return ffmpeg


def render_static_with_audio(
    *,
    image_path: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
    fps: int = 25,
    overwrite: bool = True,
    show_progress: bool = True,
) -> Path:
    """
    Renderiza um vídeo MP4 com imagem estática + áudio, usando ffmpeg.

    Equivalente ao comando manual:
      ffmpeg -y -loop 1 -i master_scene.jpg -i audio_full.wav
             -c:v libx264 -tune stillimage -pix_fmt yuv420p
             -c:a aac -b:a 192k -shortest output.mp4

    Notas:
    - Define fps fixo no stream de vídeo.
    - Força yuv420p para compatibilidade ampla.
    - Garante dimensões pares via filtro scale (H.264/yuv420p costuma exigir isso).
    - Se show_progress=True, deixa o ffmpeg escrever no terminal em tempo real.
    """
    if fps <= 0 or fps > 120:
        raise VideoError(f"fps inválido: {fps}. Use um valor entre 1 e 120.")

    image = Path(image_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not image.exists() or image.stat().st_size == 0:
        raise VideoError(f"Imagem não encontrada ou vazia: {image}")
    if not audio.exists() or audio.stat().st_size == 0:
        raise VideoError(f"Áudio não encontrado ou vazio: {audio}")

    ffmpeg = _require_ffmpeg()

    # Garante dimensões pares (evita erros com H.264/yuv420p)
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    args = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-stats",
        "-loop",
        "1",
        "-i",
        str(image),
        "-i",
        str(audio),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out),
    ]

    try:
        if show_progress:
            # Deixa o ffmpeg imprimir no terminal (frames/time/speed em tempo real).
            subprocess.run(args, check=True)
        else:
            # Captura saída para debug/erro sem "poluir" o terminal.
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise VideoError(
                    "Falha ao gerar vídeo com ffmpeg.\n"
                    f"Comando: {' '.join(args)}\n"
                    f"Saída:\n{proc.stdout}"
                )
    except subprocess.CalledProcessError as e:
        raise VideoError(
            "Falha ao gerar vídeo com ffmpeg.\n"
            f"Comando: {' '.join(args)}\n"
            f"Exit code: {e.returncode}"
        ) from e

    if not out.exists() or out.stat().st_size == 0:
        raise VideoError(f"Saída não foi gerada corretamente: {out}")

    return out
