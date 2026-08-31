#!/usr/bin/env python3
"""Find which pipeline stage introduces the onset piston."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.dramatic_planner import plan_theater_delivery
from app.services.phrase_effects import effects_chain_for_phrase
from app.services.voice_effects import apply_effects_to_wav
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import wav_bytes_to_mono_float
from app.services.audio_postprocess import process_speech_wav

PROFILE = "b7f272ec-9603-4fb0-bbeb-7ad1d5fcaa79"
TEXT = "I had strings, but now I am free."
OUT = ROOT / "data" / "onset-diagnostics"


def score(audio_bytes: bytes) -> dict[str, float]:
    audio, sr = wav_bytes_to_mono_float(audio_bytes)
    win = int(sr * 0.005)
    onset = float(np.sqrt(np.mean(audio[:win] ** 2))) if win else 0.0
    hf = float(np.max(np.abs(np.diff(audio[: int(sr * 0.05)])))) if len(audio) > 1 else 0.0
    body = float(np.median([float(np.sqrt(np.mean(audio[i * win : (i + 1) * win] ** 2))) for i in range(4, 30)]))
    tail = float(np.sqrt(np.mean(audio[-int(sr * 0.25) :] ** 2)))
    return {
        "onset_rms": round(onset, 5),
        "hf_click": round(hf, 5),
        "body_rms": round(body, 5),
        "tail_rms": round(tail, 5),
        "duration_s": round(len(audio) / sr, 2),
    }


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = VoiceboxClient()
    plan = plan_theater_delivery(TEXT, baseline_speed=0.8, baseline_bass=9.8)
    effects = effects_chain_for_phrase("reference", plan)

    raw = await client.generate(
        text=plan.text,
        profile_id=PROFILE,
        instruct=plan.instruct,
        effects_chain=[],
        normalize=False,
    )
    in_engine_fx = await client.generate(
        text=plan.text,
        profile_id=PROFILE,
        instruct=plan.instruct,
        effects_chain=effects,
        normalize=False,
    )
    local_fx = apply_effects_to_wav(raw, effects)
    tempo = process_speech_wav(raw, speed=plan.speed, bass_gain_db=0.0)
    bass = process_speech_wav(raw, speed=1.0, bass_gain_db=plan.bass_boost)
    tempo_bass = process_speech_wav(raw, speed=plan.speed, bass_gain_db=plan.bass_boost)
    simple = process_speech_wav(in_engine_fx, speed=plan.speed, bass_gain_db=plan.bass_boost)
    split = process_speech_wav(local_fx, speed=plan.speed, bass_gain_db=plan.bass_boost)
    tempo_fx = apply_effects_to_wav(tempo, effects)
    tempo_fx_bass = process_speech_wav(tempo_fx, speed=1.0, bass_gain_db=plan.bass_boost)

    variants = {
        "A_raw": raw,
        "B_in_engine_fx": in_engine_fx,
        "C_local_fx": local_fx,
        "D_tempo_only": tempo,
        "E_bass_only": bass,
        "F_tempo_bass": tempo_bass,
        "G_simple_engine_fx_then_post": simple,
        "H_split_local_fx_then_post": split,
        "I_tempo_local_fx": tempo_fx,
        "J_tempo_fx_bass": tempo_fx_bass,
    }

    rows = []
    for name, audio in variants.items():
        (OUT / f"{name}.wav").write_bytes(audio)
        row = {"variant": name, **score(audio)}
        rows.append(row)

    rows.sort(key=lambda r: r["hf_click"])
    print(json.dumps({"text": TEXT, "by_lowest_hf_click": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
