# scripts/rttm_to_segments.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    speaker_label: str
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def parse_rttm(path: Path) -> list[Segment]:
    """
    Parse mínimo de RTTM:
      SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    segments: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue

        start = float(parts[3])
        dur = float(parts[4])
        speaker = parts[7]
        segments.append(Segment(speaker_label=speaker, start=start, end=start + dur))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def merge_close_segments(segments: list[Segment], merge_gap_sec: float) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = []
    cur = segments[0]

    for seg in segments[1:]:
        # Só une se for o mesmo speaker e gap pequeno o bastante
        if seg.speaker_label == cur.speaker_label and (seg.start - cur.end) <= merge_gap_sec:
            cur = Segment(cur.speaker_label, cur.start, max(cur.end, seg.end))
        else:
            merged.append(cur)
            cur = seg

    merged.append(cur)
    return merged


def filter_min_duration(segments: list[Segment], min_segment_sec: float) -> list[Segment]:
    return [s for s in segments if s.dur >= min_segment_sec]


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_speaker_map(labels: list[str], swap: bool) -> dict[str, str]:
    """
    Mapeia labels do pyannote (ex.: SPEAKER_00, SPEAKER_01) para Speaker_0/1.
    Por padrão:
      SPEAKER_00 -> Speaker_0
      SPEAKER_01 -> Speaker_1
    Use --swap para inverter.
    """
    labels_sorted = sorted(labels)
    mapping: dict[str, str] = {}

    if len(labels_sorted) >= 1:
        mapping[labels_sorted[0]] = "Speaker_0" if not swap else "Speaker_1"
    if len(labels_sorted) >= 2:
        mapping[labels_sorted[1]] = "Speaker_1" if not swap else "Speaker_0"

    # Se houver mais labels, mantém como Speaker_N (raramente útil, mas evita crash)
    for i, lab in enumerate(labels_sorted[2:], start=2):
        mapping[lab] = f"Speaker_{i}"

    return mapping


def main() -> None:
    description = (
        "Converte RTTM em segments.json com merge/min_duration e mapeamento "
        "(labels do pyannote -> Speaker_0/1)."
    )
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--config",
        default="assets/config.json",
        help="Caminho do config.json.",
    )
    ap.add_argument(
        "--rttm",
        default="output/diarization/audio.rttm",
        help="Caminho do RTTM.",
    )
    ap.add_argument(
        "--out",
        default="output/diarization/segments.json",
        help="Saída segments.json.",
    )
    ap.add_argument(
        "--swap",
        action="store_true",
        help="Inverte SPEAKER_00<->SPEAKER_01 no mapeamento lógico.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    rttm_path = Path(args.rttm).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        raise SystemExit(f"Config não encontrado: {cfg_path}")
    if not rttm_path.exists():
        raise SystemExit(f"RTTM não encontrado: {rttm_path}")

    cfg = load_config(cfg_path)

    diar_cfg = cfg.get("diarization", {}) or {}
    min_segment_sec = float(diar_cfg.get("min_segment_sec", 0.4))
    merge_gap_sec = float(diar_cfg.get("merge_gap_sec", 0.2))

    # Ex.: Speaker_0 -> left_face
    speakers_cfg = cfg.get("speakers", {}) or {}
    if not isinstance(speakers_cfg, dict):
        raise SystemExit('"speakers" no config.json deve ser um objeto (dict).')

    segments = parse_rttm(rttm_path)
    if not segments:
        raise SystemExit("Nenhum segmento encontrado no RTTM.")

    labels = sorted({s.speaker_label for s in segments})
    spk_map = default_speaker_map(labels, swap=args.swap)

    print(f"[ai-podcast] lidos {len(segments)} segmentos do RTTM")
    print(f"[ai-podcast] speakers encontrados: {labels}")
    print(f"[ai-podcast] merge_gap_sec={merge_gap_sec}  min_segment_sec={min_segment_sec}")
    print(f"[ai-podcast] mapeamento label->lógico: {spk_map}")

    merged = merge_close_segments(segments, merge_gap_sec=merge_gap_sec)
    filtered = filter_min_duration(merged, min_segment_sec=min_segment_sec)

    print(f"[ai-podcast] após merge: {len(merged)} segmentos")
    print(f"[ai-podcast] após filtro min duração: {len(filtered)} segmentos")

    out: list[dict[str, Any]] = []
    for i, s in enumerate(filtered, start=1):
        logical = spk_map.get(s.speaker_label, s.speaker_label)
        face_id = speakers_cfg.get(logical)  # ex.: Speaker_0 -> left_face
        out.append(
            {
                "id": i,
                "speaker_label": s.speaker_label,
                "logical_speaker": logical,
                "face_id": face_id,
                "start_sec": round(s.start, 3),
                "end_sec": round(s.end, 3),
                "duration_sec": round(s.dur, 3),
            }
        )

    payload = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    print(f"[ai-podcast] segments.json salvo em: {out_path}")

    print("[ai-podcast] amostra:")
    for row in out[:10]:
        row_id = int(row["id"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        spk = str(row["speaker_label"])
        logical = str(row["logical_speaker"])
        face = row["face_id"]
        face_txt = str(face) if face is not None else "None"

        line = f"  #{row_id:>4}  {start:>8.2f}–{end:<8.2f}  {spk} -> {logical} -> {face_txt}"
        print(line)


if __name__ == "__main__":
    main()
