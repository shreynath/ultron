from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.film_identity import (
    covering_gold,
    ensure_gold_clips,
    load_profile_gold,
    typical_pause_ms,
)
from app.services.ultron_tone import LTAS_BANDS, apply_ltas_match, measure_ltas_db


def reference_pause_ms(data_dir: str | Path | None, profile_id: str | None) -> int:
    """Median pause between phrases measured from this profile's uploads."""
    if not data_dir or not profile_id:
        return 0
    clips = load_profile_gold(Path(data_dir), profile_id)
    return typical_pause_ms(clips) if clips else 0


def _median_ltas_from_clips(clips) -> list[float]:
    rows: list[list[float]] = []
    for clip in clips:
        ltas = measure_ltas_db(clip.audio, clip.sample_rate)
        if len(ltas) == len(LTAS_BANDS) and max(abs(x) for x in ltas) > 1.0:
            rows.append(ltas)
    if not rows:
        return []
    return [float(np.median([row[i] for row in rows])) for i in range(len(LTAS_BANDS))]


def apply_reference_tone(
    audio: np.ndarray,
    sample_rate: int,
    data_dir: str | Path | None,
    profile_id: str | None,
) -> np.ndarray:
    """Move every take toward the median tone of this profile's reference clips."""
    if audio.size < sample_rate // 16 or not data_dir or not profile_id:
        return audio
    clips = load_profile_gold(Path(data_dir), profile_id)
    target = _median_ltas_from_clips(clips)
    if not target:
        return audio
    return apply_ltas_match(audio, sample_rate, target, loose=True)


def film_median_ltas() -> list[float]:
    return _median_ltas_from_clips(ensure_gold_clips())


def film_pause_ms() -> int:
    pause = typical_pause_ms(ensure_gold_clips())
    return pause if pause >= 180 else 380


def delivery_pause_ms(data_dir: str | Path | None, profile_id: str | None) -> int:
    pause = reference_pause_ms(data_dir, profile_id)
    return pause if pause >= 180 else film_pause_ms()


def film_pacing_speed(duration_sec: float, text: str) -> float | None:
    """Suggest atempo speed so clone pacing nears film references without hard-trimming."""
    if duration_sec <= 0.2:
        return None
    ordered = covering_gold(text, ensure_gold_clips())
    if not ordered:
        return None
    film_sec = sum(len(clip.audio) / max(1, clip.sample_rate) for clip in ordered)
    if film_sec < 0.35:
        return None
    # Leave ~8% headroom so we never chop words to fit film length.
    target = film_sec * 1.08
    return float(np.clip(duration_sec / target, 0.92, 1.12))


def apply_delivery_tone(
    audio: np.ndarray,
    sample_rate: int,
    data_dir: str | Path | None,
    profile_id: str | None,
) -> np.ndarray:
    """Profile tone when available; otherwise match built-in film Ultron spectrum."""
    if audio.size < sample_rate // 16:
        return audio
    target: list[float] = []
    if data_dir and profile_id:
        clips = load_profile_gold(Path(data_dir), profile_id)
        target = _median_ltas_from_clips(clips)
    if not target:
        target = film_median_ltas()
    if not target:
        return audio
    return apply_ltas_match(audio, sample_rate, target, loose=True)
