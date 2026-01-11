# scripts/smoke_diarization_sf.py

from __future__ import annotations

import os

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


class SmokeError(RuntimeError):
    pass


def _device() -> torch.device:
    forced = os.getenv("DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_waveform_soundfile(path: str) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # data: (num_samples, num_channels)
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)  # mono

    # pyannote espera (channels, num_samples)
    waveform = torch.from_numpy(data.T)
    return waveform, int(sr)


def main() -> None:
    audio_path = "output/tmp/audio_16k_mono.wav"

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda:", torch.version.cuda)

    dev = _device()
    print("device:", dev)

    print("[1/3] Lendo WAV com soundfile...")
    waveform, sr = load_waveform_soundfile(audio_path)
    print("waveform:", tuple(waveform.shape), "sr:", sr)

    print("[2/3] Carregando pipeline do pyannote...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if pipeline is None:
        raise SmokeError(
            "Pipeline.from_pretrained retornou None. Verifique login/token do Hugging Face "
            "e acesso ao modelo."
        )

    # Alguns pipelines expõem .to(); outros não. Guard explícito evita alertas e erros.
    to_fn = getattr(pipeline, "to", None)
    if callable(to_fn):
        to_fn(dev)

    print("[3/3] Rodando diarização...")
    diarization = pipeline({"waveform": waveform.to(dev), "sample_rate": sr})

    print("\nPrimeiros segmentos (até 10):")
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), start=1):
        print(f"{turn.start:8.2f}  {turn.end:8.2f}  {speaker}")
        if i >= 10:
            break

    print("\nOK: diarização executou sem torchaudio.load.")


if __name__ == "__main__":
    main()
