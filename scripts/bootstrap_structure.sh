#!/usr/bin/env bash
set -euo pipefail

# bootstrap_ai_podcast_structure.sh
# Cria a estrutura base do projeto ai-podcast sem sobrescrever o que já existe.

PROJECT_ROOT="${1:-$PWD}"

mkdir -p "$PROJECT_ROOT"

# Diretórios principais
mkdir -p \
  "$PROJECT_ROOT/assets" \
  "$PROJECT_ROOT/output/diarization" \
  "$PROJECT_ROOT/output/face_map" \
  "$PROJECT_ROOT/output/segments" \
  "$PROJECT_ROOT/output/final" \
  "$PROJECT_ROOT/output/tmp" \
  "$PROJECT_ROOT/src/aipodcast" \
  "$PROJECT_ROOT/scripts" \
  "$PROJECT_ROOT/.vscode"

# Arquivos no root
[ -f "$PROJECT_ROOT/README.md" ] || printf "# ai-podcast\n" > "$PROJECT_ROOT/README.md"

[ -f "$PROJECT_ROOT/pyproject.toml" ] || cat > "$PROJECT_ROOT/pyproject.toml" <<'EOF'
[project]
name = "ai-podcast"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
EOF

# Assets (placeholders)
[ -f "$PROJECT_ROOT/assets/audio_full.wav" ] || : > "$PROJECT_ROOT/assets/audio_full.wav"
[ -f "$PROJECT_ROOT/assets/master_scene.jpg" ] || : > "$PROJECT_ROOT/assets/master_scene.jpg"

[ -f "$PROJECT_ROOT/assets/config.json" ] || cat > "$PROJECT_ROOT/assets/config.json" <<'EOF'
{
  "fps": 25,
  "speakers": {
    "Speaker_0": "left_face",
    "Speaker_1": "right_face"
  },
  "faces": {
    "left_face":  { "bbox": null },
    "right_face": { "bbox": null }
  },
  "diarization": {
    "min_segment_sec": 0.40,
    "merge_gap_sec": 0.20
  },
  "crop": {
    "pad_px": 24,
    "target_size": 512
  },
  "compositing": {
    "method": "alpha_feather",
    "feather_px": 18
  },
  "quality": {
    "use_gfpgan": true
  }
}
EOF

# Package src/aipodcast
[ -f "$PROJECT_ROOT/src/aipodcast/__init__.py" ] || : > "$PROJECT_ROOT/src/aipodcast/__init__.py"

for f in cli.py config.py diarization.py faces.py audio.py video.py lipsync.py restore.py compositing.py pipeline.py; do
  [ -f "$PROJECT_ROOT/src/aipodcast/$f" ] || : > "$PROJECT_ROOT/src/aipodcast/$f"
done

# scripts/run.sh
[ -f "$PROJECT_ROOT/scripts/run.sh" ] || cat > "$PROJECT_ROOT/scripts/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Atalho opcional para rodar a CLI (ajuste conforme a implementação evoluir)
python -m aipodcast.cli run --assets assets --out output
EOF
chmod +x "$PROJECT_ROOT/scripts/run.sh"

# Projeto menu markers
[ -f "$PROJECT_ROOT/.project-env" ] || printf "ai-podcast\n" > "$PROJECT_ROOT/.project-env"

[ -f "$PROJECT_ROOT/.project-tasks" ] || cat > "$PROJECT_ROOT/.project-tasks" <<'EOF'
# Tasks - AI Podcast
Abrir VS Code | code .
EOF

# VS Code settings (cria somente se não existir)
[ -f "$PROJECT_ROOT/.vscode/settings.json" ] || cat > "$PROJECT_ROOT/.vscode/settings.json" <<'EOF'
{
  "python.defaultInterpreterPath": "/home/criton/miniconda3/envs/ai-podcast/bin/python",
  "python.terminal.activateEnvironment": true,

  "editor.formatOnSave": true,
  "editor.formatOnSaveMode": "file",

  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/*.pyo": true,
    "**/*.pyd": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true,

    "**/output": true,
    "**/tmp": true,

    "**/*.mp4": true,
    "**/*.mkv": true,
    "**/*.mov": true,
    "**/*.avi": true,
    "**/*.wav": true,
    "**/*.mp3": true
  },

  "search.exclude": {
    "**/.git": true,
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true,
    "**/output": true,
    "**/tmp": true,

    "**/*.mp4": true,
    "**/*.mkv": true,
    "**/*.mov": true,
    "**/*.avi": true,
    "**/*.wav": true,
    "**/*.mp3": true
  },

  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    }
  },

  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,

  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.autoTestDiscoverOnSaveEnabled": false,

  "terminal.integrated.scrollback": 10000,

  "terminal.integrated.env.linux": {
    "SKIP_PROJ_MENU": "1"
  }
}
EOF

echo "OK: Estrutura do projeto garantida em: $PROJECT_ROOT"
echo "Dica: rode 'tree -a -L 4' para inspecionar."
