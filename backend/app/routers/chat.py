from __future__ import annotations

import base64
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.ollama_client import OllamaError, chat_response
from app.services.voice_delivery import VoiceDeliverySettings, deliver_speech
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/chat", tags=["chat"])
voicebox = VoiceboxClient()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    profile_id: str
    engine: str | None = "qwen"
    system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational, 1-3 sentences."
    voice_style: str = Field(default="theater_auto", pattern="^(theater_auto|ultron_cinematic|reference|natural)$")
    speed: float = Field(default=0.84, ge=0.7, le=1.0)
    bass_boost: float = Field(default=5.0, ge=0.0, le=12.0)


class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str
    audio_base64: str


def _delivery_from_request(body: ChatRequest) -> VoiceDeliverySettings:
    return VoiceDeliverySettings(
        voice_style=body.voice_style,
        speed=body.speed,
        bass_boost=body.bass_boost,
    )


@router.post("")
async def chat_and_speak(body: ChatRequest):
    try:
        assistant_message = await chat_response(
            user_message=body.message,
            system_prompt=body.system_prompt,
        )
        audio_bytes = await deliver_speech(
            voicebox,
            text=assistant_message,
            profile_id=body.profile_id,
            engine=body.engine,
            delivery=_delivery_from_request(body),
            data_dir=settings.data_dir,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat failed: {exc}") from exc

    return ChatResponse(
        user_message=body.message,
        assistant_message=assistant_message,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
    )


@router.post("/voice")
async def voice_chat(
    file: UploadFile = File(...),
    profile_id: str = Form(...),
    engine: str = Form("qwen"),
    voice_style: str = Form("theater_auto"),
):
    work_dir = Path(settings.data_dir) / "chat" / str(uuid.uuid4())
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    upload_path = work_dir / f"recording{suffix}"
    upload_path.write_bytes(await file.read())

    wav_path = work_dir / "recording.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(upload_path),
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail="Could not process audio recording")

    delivery = VoiceDeliverySettings(
        voice_style=voice_style,
        profile_id=profile_id,
    )

    try:
        user_message = await voicebox.transcribe(wav_path)
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="No speech detected in recording")

        assistant_message = await chat_response(user_message=user_message.strip())
        audio_bytes = await deliver_speech(
            voicebox,
            text=assistant_message,
            profile_id=profile_id,
            engine=engine,
            delivery=delivery,
            data_dir=settings.data_dir,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc

    return ChatResponse(
        user_message=user_message.strip(),
        assistant_message=assistant_message,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
    )
