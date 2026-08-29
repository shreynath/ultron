from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.audio_postprocess import AudioPostprocessError, process_speech_wav
from app.services.voice_presets import resolve_voice_delivery
from app.services.voicebox_client import VoiceboxClient


class VoiceDeliverySettings(BaseModel):
    voice_style: str = Field(default="ultron_cinematic", pattern="^(ultron_cinematic|natural)$")
    speed: float = Field(default=0.86, ge=0.7, le=1.0)
    bass_boost: float = Field(default=6.0, ge=0.0, le=12.0)


async def deliver_speech(
    client: VoiceboxClient,
    *,
    text: str,
    profile_id: str,
    language: str = "en",
    engine: str | None = None,
    delivery: VoiceDeliverySettings | None = None,
) -> bytes:
    delivery = delivery or VoiceDeliverySettings()
    instruct, effects_chain = resolve_voice_delivery(delivery.voice_style)

    audio_bytes = await client.generate(
        text=text,
        profile_id=profile_id,
        language=language,
        engine=engine,
        instruct=instruct,
        effects_chain=effects_chain,
    )

    if delivery.voice_style == "ultron_cinematic":
        try:
            audio_bytes = process_speech_wav(
                audio_bytes,
                speed=delivery.speed,
                bass_gain_db=delivery.bass_boost,
            )
        except AudioPostprocessError:
            if delivery.speed < 0.99:
                audio_bytes = process_speech_wav(audio_bytes, speed=delivery.speed)
    elif delivery.speed < 0.99:
        audio_bytes = process_speech_wav(audio_bytes, speed=delivery.speed)

    return audio_bytes
