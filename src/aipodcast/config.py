# src/aipodcast/config.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Erro de configuração (config.json inválido ou arquivos ausentes)."""


@dataclass(frozen=True)
class Paths:
    assets_dir: Path
    output_dir: Path

    audio_full: Path
    master_scene: Path

    diarization_dir: Path
    face_map_dir: Path
    segments_dir: Path
    final_dir: Path
    tmp_dir: Path

    static_with_audio: Path
    final_video: Path


@dataclass(frozen=True)
class AppConfig:
    fps: int
    paths: Paths

    speakers: dict[str, str]
    faces: dict[str, dict[str, Any]]
    diarization: dict[str, Any]
    crop: dict[str, Any]
    compositing: dict[str, Any]
    quality: dict[str, Any]


def _as_path(root: Path, value: str) -> Path:
    """Resolve paths relativos em relação ao diretório do projeto (root)."""
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(config_path: str | Path = "assets/config.json") -> AppConfig:
    """
    Carrega e valida o config.json.

    Regras:
    - Espera a chave "fps" (int).
    - Espera a chave "paths" com os caminhos usados no projeto.
    - Valida existência de audio_full e master_scene.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise ConfigError(f"Config não encontrado: {cfg_path}")

    project_root = cfg_path.parent.parent if cfg_path.parent.name == "assets" else cfg_path.parent

    raw: dict[str, Any]
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"JSON inválido em {cfg_path}: {e}") from e

    fps = int(raw.get("fps", 25))
    if fps <= 0 or fps > 120:
        raise ConfigError(f"fps inválido: {fps}. Use um valor entre 1 e 120.")

    paths_raw = raw.get("paths")
    if not isinstance(paths_raw, dict):
        raise ConfigError('Chave "paths" ausente ou inválida no config.json.')

    # Obrigatórios
    audio_full = _as_path(
        project_root,
        str(paths_raw.get("audio_full", "assets/audio_full.wav")),
    )
    master_scene = _as_path(
        project_root,
        str(paths_raw.get("master_scene", "assets/master_scene.jpg")),
    )

    # Diretórios (defaults coerentes com seu layout)
    assets_dir = _as_path(project_root, str(paths_raw.get("assets_dir", "assets")))
    output_dir = _as_path(project_root, str(paths_raw.get("output_dir", "output")))

    diarization_dir = _as_path(
        project_root,
        str(paths_raw.get("diarization_dir", "output/diarization")),
    )
    face_map_dir = _as_path(
        project_root,
        str(paths_raw.get("face_map_dir", "output/face_map")),
    )
    segments_dir = _as_path(
        project_root,
        str(paths_raw.get("segments_dir", "output/segments")),
    )
    final_dir = _as_path(
        project_root,
        str(paths_raw.get("final_dir", "output/final")),
    )
    tmp_dir = _as_path(
        project_root,
        str(paths_raw.get("tmp_dir", "output/tmp")),
    )

    static_with_audio = _as_path(
        project_root,
        str(paths_raw.get("static_with_audio", "output/final/static_with_audio.mp4")),
    )
    final_video = _as_path(
        project_root,
        str(paths_raw.get("final_video", "output/final/podcast.mp4")),
    )

    # Valida inputs
    if not audio_full.exists() or audio_full.stat().st_size == 0:
        raise ConfigError(f"Áudio não encontrado ou vazio: {audio_full}")

    if not master_scene.exists() or master_scene.stat().st_size == 0:
        raise ConfigError(f"Imagem master_scene não encontrada ou vazia: {master_scene}")

    paths = Paths(
        assets_dir=assets_dir,
        output_dir=output_dir,
        audio_full=audio_full,
        master_scene=master_scene,
        diarization_dir=diarization_dir,
        face_map_dir=face_map_dir,
        segments_dir=segments_dir,
        final_dir=final_dir,
        tmp_dir=tmp_dir,
        static_with_audio=static_with_audio,
        final_video=final_video,
    )

    # Demais seções (podem existir agora ou depois; default = dict vazio)
    speakers = raw.get("speakers") or {}
    faces = raw.get("faces") or {}
    diarization = raw.get("diarization") or {}
    crop = raw.get("crop") or {}
    compositing = raw.get("compositing") or {}
    quality = raw.get("quality") or {}

    # Tipos mínimos
    if not isinstance(speakers, dict):
        raise ConfigError('"speakers" deve ser um objeto JSON (dict).')
    if not isinstance(faces, dict):
        raise ConfigError('"faces" deve ser um objeto JSON (dict).')

    return AppConfig(
        fps=fps,
        paths=paths,
        speakers=speakers,
        faces=faces,
        diarization=diarization,
        crop=crop,
        compositing=compositing,
        quality=quality,
    )
