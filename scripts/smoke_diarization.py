# scripts/smoke_diarization.py

from __future__ import annotations

import os
from pathlib import Path

import torch

AUDIO = Path("output/tmp/audio_16k_mono.wav")


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda:", torch.version.cuda)

    if not AUDIO.exists():
        raise SystemExit(f"Arquivo não encontrado: {AUDIO.resolve()}")

    # Teste 1: leitura com soundfile (camada mais simples)
    print("\n[1/3] Teste de leitura com soundfile...")
    import soundfile as sf

    info = sf.info(str(AUDIO))
    print("soundfile:", info)

    # Teste 2: leitura com torchaudio (backends/decoders próprios)
    print("\n[2/3] Teste de leitura com torchaudio...")
    import torchaudio

    wav, sr = torchaudio.load(str(AUDIO))
    print("torchaudio:", "shape=", tuple(wav.shape), "sr=", sr, "dtype=", wav.dtype)

    # Teste 3: pyannote diarization
    print("\n[3/3] Teste pyannote diarization...")

    requested = os.environ.get("DEVICE", "cpu").lower().strip()
    device = "cuda" if (requested == "cuda" and torch.cuda.is_available()) else "cpu"
    print("device:", device)

    from pyannote.audio import Pipeline as DiarizationPipeline

    pipeline = DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if pipeline is None:
        raise RuntimeError(
            "Pipeline.from_pretrained retornou None. Verifique autenticação/termos no Hugging Face."
        )

    pipeline.to(torch.device(device))
    diarization = pipeline(str(AUDIO))

    print("\nPrimeiros 10 segmentos:")
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), start=1):
        print(f"{turn.start:8.2f}  {turn.end:8.2f}  {speaker}")
        if i >= 10:
            break


if __name__ == "__main__":
    main()
