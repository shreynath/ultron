from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.services.audio_postprocess import AudioPostprocessError, process_speech_wav
from app.services.dynamic_delivery import deliver_theater_speech
from app.services.voice_presets import resolve_voice_delivery
from app.services.voicebox_client import VoiceboxClient


class VoiceDeliverySettings(BaseModel):
    voice_style: str = Field(
        default="theater_auto",
        pattern="^(theater_auto|ultron_cinematic|reference|natural)$",
    )
    speed: float = Field(default=0.84, ge=0.7, le=1.0)
    bass_boost: float = Field(default=5.0, ge=0.0, le=12.0)
    profile_id: str | None = None


async def deliver_speech(
    client: VoiceboxClient,
    *,
    text: str,
    profile_id: str,
    language: str = "en",
    engine: str | None = None,
    delivery: VoiceDeliverySettings | None = None,
    data_dir: str | None = None,
) -> bytes:
    delivery = delivery or VoiceDeliverySettings(profile_id=profile_id)

    if delivery.profile_id is None:
        delivery.profile_id = profile_id

    baseline_speed = delivery.speed
    baseline_bass = delivery.bass_boost
    if data_dir and delivery.voice_style != "natural":
        from app.services.reference_extractor import load_profile_delivery

        profile_delivery = load_profile_delivery(Path(data_dir), profile_id)
        if profile_delivery:
            baseline_speed = profile_delivery.get("speed", baseline_speed)
            baseline_bass = profile_delivery.get("bass_boost", baseline_bass)

    if delivery.voice_style == "natural":
        return await client.generate(
            text=text,
            profile_id=profile_id,
            language=language,
            engine=engine,
        )

    if delivery.voice_style in {"theater_auto", "reference", "ultron_cinematic"}:
        try:
            return await deliver_theater_speech(
                client,
                text=text,
                profile_id=profile_id,
                language=language,
                engine=engine,
                voice_style=delivery.voice_style,
                baseline_speed=baseline_speed,
                baseline_bass=baseline_bass,
                data_dir=data_dir,
            )
        except AudioPostprocessError:
            pass

    instruct, effects_chain = resolve_voice_delivery(delivery.voice_style)
    audio_bytes = await client.generate(
        text=text,
        profile_id=profile_id,
        language=language,
        engine=engine,
        instruct=instruct,
        effects_chain=effects_chain,
    )
    return process_speech_wav(audio_bytes, speed=baseline_speed, bass_gain_db=baseline_bass)
