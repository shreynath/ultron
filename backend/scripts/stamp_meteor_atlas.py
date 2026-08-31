"""Rebuild sentence-level Ultron atlas from known film windows (no Whisper)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.film_identity import FILM_WINDOWS, extract_gold_from_source
from app.services.ultron_atlas import load_atlas, phrases_from_paragraph, save_atlas
from app.services.wav_io import wav_bytes_to_mono_float

SOURCE = Path(__file__).resolve().parents[1] / "data" / "ultron_ref" / "source.wav"
OUT = Path(__file__).resolve().parents[1] / "data" / "ultron_ref" / "ultron_prosody_atlas.json"
SHIP = Path(__file__).resolve().parents[1] / "app" / "data" / "ultron_prosody_atlas.json"

WINDOWS = FILM_WINDOWS

_DROP_MARKERS = (
    "meant to be new",
    "think a lot about meteor",
    "wasn't sure you'd wake",
    "look up in horror",
    "seen mercy",
)


def _is_stale(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DROP_MARKERS)


def main() -> None:
    audio, sample_rate = wav_bytes_to_mono_float(SOURCE.read_bytes())
    gold = []
    for start_sec, end_sec, text in WINDOWS:
        clip = audio[int(start_sec * sample_rate) : int(end_sec * sample_rate)]
        built = phrases_from_paragraph(text, clip.astype(np.float32), sample_rate)
        print(f"{start_sec:7.2f}-{end_sec:7.2f} -> {len(built)} sentences")
        for phrase in built:
            filled = sum(1 for contour in phrase.word_f0 if len(contour) >= 2)
            medians = [float(np.median(c)) for c in phrase.word_f0 if c]
            print(
                f"  {phrase.duration_sec:5.2f}s filled={filled}/{len(phrase.words)} "
                f"f0~{np.median(medians) if medians else 0:5.1f}  {phrase.text}"
            )
            for word, contour in zip(phrase.words, phrase.word_f0):
                voiced = [hz for hz in contour if hz > 1]
                med = float(np.median(voiced)) if voiced else 0.0
                print(f"     {word:12s} n={len(voiced):3d}  {med:6.1f}")
        gold.extend(built)

    kept = []
    gold_texts = {phrase.text.strip().lower() for phrase in gold}
    for phrase in load_atlas():
        if phrase.text.strip().lower() in gold_texts or _is_stale(phrase.text):
            continue
        kept.append(phrase)
    phrases = gold + kept
    save_atlas(phrases, OUT)
    save_atlas(phrases, SHIP)
    gold_clips = extract_gold_from_source(SOURCE)
    print(f"saved {len(gold)} gold + {len(kept)} lexicon -> {len(phrases)}  {OUT}")
    print(f"wrote {len(gold_clips)} gold wavs")


if __name__ == "__main__":
    main()
