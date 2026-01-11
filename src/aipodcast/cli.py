# src/aipodcast/cli.py

from __future__ import annotations

import typer

from aipodcast.config import ConfigError, load_config
from aipodcast.video import VideoError, render_static_with_audio

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("run")
def run_cmd(
    config: str = typer.Option(
        "assets/config.json",
        "--config",
        "-c",
        help="Caminho para o config.json (default: assets/config.json).",
    ),
) -> None:
    """
    Milestone 0: gera um vídeo final com imagem estática + áudio (ffmpeg),
    lendo paths e fps do assets/config.json.
    """
    try:
        cfg = load_config(config)
        out = render_static_with_audio(
            image_path=cfg.paths.master_scene,
            audio_path=cfg.paths.audio_full,
            out_path=cfg.paths.static_with_audio,
            fps=cfg.fps,
            overwrite=True,
        )
    except (ConfigError, VideoError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1) from e

    typer.echo(f"OK: gerado {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
