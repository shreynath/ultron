#!/usr/bin/env python3
"""Sweep attack-fix strategies and rank by HF click + noise floor."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.audio_postprocess import process_speech_wav
from app.services.dramatic_planner import plan_theater_delivery
from app.services.phrase_effects import effects_chain_for_phrase
from app.services.tts_cleanup import cleanup_mlx_wav
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import wav_bytes_to_mono_float

PROFILE = "b7f272ec-9603-4fb0-bbeb-7ad1d5fcaa79"
TEXT = "I had strings, but now I am free."


def score(audio_bytes: bytes) -> dict[str, float]:
    audio, sr = wav_bytes_to_mono_float(audio_bytes)
    win = int(sr * 0.005)
    hf = float(np.max(np.abs(np.diff(audio[: int(sr * 0.06)]))))
    onset = float(np.sqrt(np.mean(audio[:win] ** 2)))
    body = float(np.median([float(np.sqrt(np.mean(audio[i * win : (i + 1) * win] ** 2))) for i in range(4, 24)]))
    floor_bins = sorted(
        float(np.sqrt(np.mean(audio[i : i + int(sr * 0.02)] ** 2)))
        for i in range(0, len(audio) - int(sr * 0.02), int(sr * 0.02))
    )
    floor = float(np.mean(floor_bins[: max(1, len(floor_bins) // 10)]))
    return {"hf": round(hf, 4), "onset": round(onset, 4), "body": round(body, 4), "floor": round(floor, 4)}


async def main() -> None:
    from app.services import attack_processor as ap

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
    cleaned = cleanup_mlx_wav(raw)

    variants = {
        "raw": cleaned,
        "current": ap.process_theater_audio(cleaned, effects, plan.speed, plan.bass_boost),
        "tempo_first": ap.process_theater_audio(
            cleaned, effects, plan.speed, plan.bass_boost, tempo_before_effects=True
        ),
        "long_predelay": ap.process_theater_audio(
            cleaned,
            effects,
            plan.speed,
            plan.bass_boost,
            reverb_predelay_ms=120,
            reverb_xfade_ms=60,
        ),
        "no_bass": ap.process_theater_audio(cleaned, effects, plan.speed, 0.0),
    }

    rows = [{"variant": k, **score(v)} for k, v in variants.items()]
    rows.sort(key=lambda r: (r["hf"], r["onset"]))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
