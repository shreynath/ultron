from __future__ import annotations

import base64
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import settings
from app.services.engines import normalize_engine
from app.services.voice_delivery import VoiceDeliverySettings, deliver_speech
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/speak", tags=["speak"])
client = VoiceboxClient()


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    profile_id: str
    language: str = "en"
    engine: str | None = None
    voice_style: str = Field(default="theater_auto", pattern="^(theater_auto|ultron_cinematic|reference|natural)$")
    speed: float = Field(default=0.84, ge=0.7, le=1.0)
    bass_boost: float = Field(default=5.0, ge=0.0, le=12.0)


class SpeakResponse(BaseModel):
    audio_base64: str
    content_type: str = "audio/wav"


def _delivery_from_request(body: SpeakRequest) -> VoiceDeliverySettings:
    return VoiceDeliverySettings(
        voice_style=body.voice_style,
        speed=body.speed,
        bass_boost=body.bass_boost,
    )


@router.post("")
async def speak(body: SpeakRequest):
    try:
        audio_bytes = await deliver_speech(
            client,
            text=body.text,
            profile_id=body.profile_id,
            language=body.language,
            engine=normalize_engine(body.engine),
            delivery=_delivery_from_request(body),
            data_dir=settings.data_dir,
        )
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Speech generation failed: {exc}") from exc

    output_dir = Path(settings.data_dir) / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4()}.wav"
    output_path.write_bytes(audio_bytes)

    return SpeakResponse(audio_base64=base64.b64encode(audio_bytes).decode("ascii"))


@router.post("/download")
async def speak_download(body: SpeakRequest):
    try:
        audio_bytes = await deliver_speech(
            client,
            text=body.text,
            profile_id=body.profile_id,
            language=body.language,
            engine=normalize_engine(body.engine),
            delivery=_delivery_from_request(body),
            data_dir=settings.data_dir,
        )
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="ultron-speech.wav"'},
    )
