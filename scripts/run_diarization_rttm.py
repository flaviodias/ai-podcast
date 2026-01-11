# scripts/run_diarization_rttm.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TextIO

import torch
from pyannote.audio import Pipeline


def load_pipeline(model_id: str, hf_token: str | None) -> Pipeline:
    """
    Carrega o pipeline de diarização do pyannote.

    Nota sobre token:
    - Preferimos `token=...` quando disponível.
    - Se a assinatura não aceitar `token`, fazemos fallback:
        - exporta `HUGGINGFACE_HUB_TOKEN` (usado pelo huggingface_hub)
        - chama `from_pretrained(model_id)` sem kwargs.
    """
    if hf_token:
        try:
            pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
        except TypeError:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            pipeline = Pipeline.from_pretrained(model_id)
    else:
        pipeline = Pipeline.from_pretrained(model_id)

    if pipeline is None:
        raise RuntimeError("Pipeline.from_pretrained retornou None (inesperado).")

    return pipeline


def extract_annotation(diarize_output: Any) -> Any:
    """
    Em pyannote.audio 4.x, pipeline(audio) retorna um 'DiarizeOutput' (wrapper),
    e o Annotation costuma estar em `output.speaker_diarization`.
    """
    # Caso já seja um Annotation compatível (tem itertracks), devolve direto.
    if hasattr(diarize_output, "itertracks"):
        return diarize_output

    # Caso principal (pyannote.audio 4.x)
    if hasattr(diarize_output, "speaker_diarization"):
        ann = diarize_output.speaker_diarization
        if ann is not None:
            return ann

    # Outros nomes possíveis (fallbacks)
    if hasattr(diarize_output, "annotation"):
        ann = diarize_output.annotation
        if ann is not None:
            return ann

    if hasattr(diarize_output, "diarization"):
        ann = diarize_output.diarization
        if ann is not None:
            return ann

    raise TypeError(
        "Não consegui extrair um Annotation da saída do pipeline. "
        "Esperado: output.speaker_diarization (pyannote.audio 4.x) "
        "ou um objeto com itertracks."
    )


def write_rttm_from_itertracks(annotation: Any, f: TextIO, *, uri: str) -> None:
    """
    Escreve RTTM usando itertracks(yield_label=True), que é o caminho mais estável.

    Formato RTTM:
      SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    if not hasattr(annotation, "itertracks"):
        raise TypeError("Annotation não possui itertracks; não consigo exportar RTTM.")

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = float(turn.start)
        dur = float(turn.end - turn.start)
        f.write(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


def main() -> None:
    model_id = "pyannote/speaker-diarization-3.1"
    audio_path = Path("output/tmp/audio_16k_mono.wav").resolve()
    out_path = Path("output/diarization/audio.rttm").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise SystemExit(f"Áudio não encontrado: {audio_path}")

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ai-podcast] device={device}  torch={torch.__version__}")

    print(f"[ai-podcast] carregando pipeline: {model_id}")
    pipeline = load_pipeline(model_id, hf_token)
    pipeline.to(device)

    print(f"[ai-podcast] rodando diarização em: {audio_path}")
    diarize_output = pipeline(str(audio_path))

    annotation = extract_annotation(diarize_output)

    uri = audio_path.stem  # usado no RTTM
    print(f"[ai-podcast] salvando RTTM em: {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        write_rttm_from_itertracks(annotation, f, uri=uri)

    print("[ai-podcast] amostra de segmentos:")
    for i, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True), start=1):
        print(f"  {turn.start:8.2f}  {turn.end:8.2f}  {speaker}")
        if i >= 10:
            break


if __name__ == "__main__":
    main()
