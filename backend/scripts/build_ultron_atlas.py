"""Transcribe Ultron compilation utterances and store per-word F0 contours."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ultron_atlas import phrases_from_paragraph, save_atlas
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import wav_bytes_to_mono_float, mono_float_to_wav_bytes

SOURCE = Path(__file__).resolve().parents[1] / "data" / "ultron_ref" / "source.wav"
OUT = Path(__file__).resolve().parents[1] / "data" / "ultron_ref" / "ultron_prosody_atlas.json"
SHIP = Path(__file__).resolve().parents[1] / "app" / "data" / "ultron_prosody_atlas.json"


def _utterances(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frame = int(sample_rate * 0.02)
    n = len(audio) // frame
    rms = np.array(
        [np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2)) for i in range(n)]
    )
    speech = rms > 10 ** (-32 / 20)
    gap_max = int(0.28 / 0.02)
    min_frames = int(0.80 / 0.02)
    utts: list[tuple[int, int]] = []
    start = None
    silence = 0
    for index, voiced in enumerate(speech):
        if voiced:
            if start is None:
                start = index
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_max:
                end = index - silence + 1
                if end - start >= min_frames:
                    utts.append((start * frame, end * frame))
                start = None
                silence = 0
    if start is not None and n - start >= min_frames:
        utts.append((start * frame, n * frame))
    return utts


async def main() -> None:
    audio, sample_rate = wav_bytes_to_mono_float(SOURCE.read_bytes())
    client = VoiceboxClient()
    phrases = []
    raw_utts = _utterances(audio, sample_rate)
    utts: list[tuple[int, int]] = []
    max_samples = int(18 * sample_rate)
    min_samples = int(0.8 * sample_rate)
    for start, end in raw_utts:
        if end - start <= max_samples:
            utts.append((start, end))
            continue
        cursor = start
        while cursor < end:
            stop = min(cursor + max_samples, end)
            if stop - cursor >= min_samples:
                utts.append((cursor, stop))
            cursor = stop

    for index, (start, end) in enumerate(utts):
        clip = audio[start:end]
        wav_path = SOURCE.parent / f"_atlas_probe_{index}.wav"
        wav_path.write_bytes(mono_float_to_wav_bytes(clip.astype(np.float32), sample_rate))
        try:
            text = (await client.transcribe(wav_path)).strip()
        except Exception as exc:
            print(f"skip {index}: {exc}")
            wav_path.unlink(missing_ok=True)
            continue
        wav_path.unlink(missing_ok=True)
        if len(text.split()) < 2:
            print(f"{index} too short: {text!r}")
            continue
        built = phrases_from_paragraph(text, clip.astype(np.float32), sample_rate)
        if not built:
            print(f"{index} no f0: {text[:80]!r}")
            continue
        for phrase in built:
            medians = [
                float(np.median(contour)) if contour else 0.0 for contour in phrase.word_f0
            ]
            filled = sum(1 for contour in phrase.word_f0 if len(contour) >= 2)
            print(
                f"{index:02d} t={start/sample_rate:6.1f}s {phrase.duration_sec:5.2f}s "
                f"words={len(phrase.words):2d} filled={filled} "
                f"f0~{np.median([m for m in medians if m>0] or [0]):5.1f}  {phrase.text[:90]}"
            )
            phrases.append(phrase)

    save_atlas(phrases, OUT)
    save_atlas(phrases, SHIP)
    print(f"saved {len(phrases)} phrases -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
