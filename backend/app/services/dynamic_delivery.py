from __future__ import annotations

import numpy as np

from app.services.attack_processor import process_theater_audio
from app.services.dramatic_planner import plan_phrases, plan_theater_delivery
from app.services.phrase_effects import effects_chain_for_phrase
from app.services.reference_delivery import film_pacing_speed
from app.services.tts_cleanup import cleanup_mlx_wav
from app.services.voice_presets import ULTRON_REFERENCE_INSTRUCT
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import wav_bytes_to_mono_float


async def deliver_theater_speech(
    client: VoiceboxClient,
    *,
    text: str,
    profile_id: str,
    language: str = "en",
    engine: str | None = None,
    voice_style: str = "theater_auto",
    baseline_speed: float = 0.84,
    baseline_bass: float = 5.0,
    data_dir: str | None = None,
) -> bytes:
    """Always generate from the clone, then shape delivery toward Ultron references."""
    plan = plan_theater_delivery(text, baseline_speed=baseline_speed, baseline_bass=baseline_bass)
    plans = plan_phrases(text, baseline_speed=baseline_speed, baseline_bass=baseline_bass)
    style = "reference" if voice_style == "theater_auto" else voice_style
    effects = effects_chain_for_phrase(style, plan)

    raw_audio = await client.generate(
        text=plan.text,
        profile_id=profile_id,
        language=language,
        engine=engine,
        instruct=ULTRON_REFERENCE_INSTRUCT,
        effects_chain=[],
        normalize=False,
    )
    cleaned = cleanup_mlx_wav(raw_audio)
    audio, sample_rate = wav_bytes_to_mono_float(cleaned)
    duration = len(audio) / max(1, sample_rate)
    words = max(1, len(plan.text.split()))
    wps = words / max(0.4, duration)

    speed = plan.speed
    film_speed = film_pacing_speed(duration, text)
    if film_speed is not None:
        speed = min(speed, film_speed)
    elif words <= 16 and wps > 1.7:
        speed = max(0.88, min(speed, 1.48 / wps))
    elif wps > 2.1:
        speed = max(0.90, min(speed, 1.95 / wps))
    else:
        speed = min(speed, 0.92)
    if wps < 1.5:
        speed = max(speed, 0.90)

    return process_theater_audio(
        cleaned,
        effects,
        speed,
        plan.bass_boost,
        text=text,
        plans=plans,
        baseline_bass=baseline_bass,
        bass_freq=plan.bass_freq,
        intensity=plan.intensity,
        profile_id=profile_id,
        data_dir=data_dir,
    )
