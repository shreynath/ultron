from __future__ import annotations

import asyncio
import base64
import json
import logging
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.brain import CognitiveRuntime, Executive, bound_history
from app.config import settings
from app.services.voice_delivery import VoiceDeliverySettings, deliver_speech
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)
voicebox = VoiceboxClient()
executive = Executive()
runtime = CognitiveRuntime()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=32)
    # Voice is an optional delivery modality. Text conversation never requires
    # a cloned profile to be available.
    profile_id: str | None = None
    voice_enabled: bool = True
    engine: str | None = "qwen"
    system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational, 1-3 sentences."
    voice_style: str = Field(default="theater_auto", pattern="^(theater_auto|ultron_cinematic|reference|natural)$")
    speed: float = Field(default=0.84, ge=0.7, le=1.0)
    bass_boost: float = Field(default=5.0, ge=0.0, le=12.0)


class ConversationMetadata(BaseModel):
    """Safe presentation metadata; never raw cognitive internals."""

    task_type: str
    route: str
    status: str
    request_id: str | None = None
    execution_events: tuple[str, ...] | None = None


class ChatResponse(BaseModel):
    user_message: str
    assistant_message: str
    audio_base64: str | None = None
    voice_error: str | None = None
    metadata: ConversationMetadata


def _delivery_from_request(body: ChatRequest) -> VoiceDeliverySettings:
    return VoiceDeliverySettings(
        voice_style=body.voice_style,
        speed=body.speed,
        bass_boost=body.bass_boost,
    )


async def _run_conversation(
    *,
    message: str,
    history: list[dict[str, str]],
    system_prompt: str,
) -> tuple[str, ConversationMetadata]:
    """The single text conversation path: Executive then CognitiveRuntime."""

    try:
        bounded_history = bound_history(history)
        assessment = await executive.assess(message, history=bounded_history)
        execution = await runtime.handle_conversation(
            user_input=message,
            history=bounded_history,
            assessment=assessment,
            system_prompt=system_prompt,
        )
        return execution.answer, ConversationMetadata(
            task_type=execution.assessment.task_type.value,
            route=execution.route.value,
            status=execution.status.value,
            request_id=str(execution.request_id) if execution.request_id else None,
            execution_events=tuple(event.event for event in execution.trace) or None,
        )
    except Exception as exc:
        # Executive and runtime failures are not implementation details for the
        # conversation surface. Voice failures are handled separately below.
        logger.warning("conversation processing failed kind=%s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="I couldn't process that request. Please try again.",
        ) from exc


async def _voice_delivery(
    *,
    message: str,
    profile_id: str | None,
    engine: str | None,
    delivery: VoiceDeliverySettings,
) -> tuple[str | None, str | None]:
    if not profile_id:
        return None, None
    try:
        audio_bytes = await deliver_speech(
            voicebox,
            text=message,
            profile_id=profile_id,
            engine=engine,
            delivery=delivery,
            data_dir=settings.data_dir,
        )
    except Exception as exc:  # noqa: BLE001 - delivery cannot invalidate text
        logger.warning("conversation voice delivery failed kind=%s", type(exc).__name__)
        return None, "Audio is unavailable for this response."
    return base64.b64encode(audio_bytes).decode("ascii"), None


@router.post("", response_model=ChatResponse)
async def chat_and_speak(body: ChatRequest):
    assistant_message, metadata = await _run_conversation(
        message=body.message,
        history=body.history,
        system_prompt=body.system_prompt,
    )
    audio_base64 = None
    voice_error = None
    if body.voice_enabled:
        audio_base64, voice_error = await _voice_delivery(
            message=assistant_message,
            profile_id=body.profile_id,
            engine=body.engine,
            delivery=_delivery_from_request(body),
        )
    return ChatResponse(
        user_message=body.message,
        assistant_message=assistant_message,
        audio_base64=audio_base64,
        voice_error=voice_error,
        metadata=metadata,
    )


@router.post("/voice")
async def voice_chat(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI declares multipart input here
    profile_id: str | None = Form(None),
    engine: str = Form("qwen"),
    voice_style: str = Form("theater_auto"),
    voice_enabled: bool = Form(True),
    history_json: str = Form("[]"),
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
    result = await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail="Could not process audio recording")

    try:
        user_message = await voicebox.transcribe(wav_path)
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="No speech detected in recording")
    except VoiceboxError as exc:
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail="I couldn't understand the recording. Please try again.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("conversation transcription failed kind=%s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="I couldn't understand the recording. Please try again.",
        ) from exc

    try:
        history = json.loads(history_json)
        if not isinstance(history, list):
            history = []
    except json.JSONDecodeError:
        history = []
    assistant_message, metadata = await _run_conversation(
        message=user_message.strip(),
        history=history,
        system_prompt="You are a helpful assistant. Keep replies concise and conversational, 1-3 sentences.",
    )
    audio_base64 = None
    voice_error = None
    if voice_enabled:
        audio_base64, voice_error = await _voice_delivery(
            message=assistant_message,
            profile_id=profile_id,
            engine=engine,
            delivery=VoiceDeliverySettings(voice_style=voice_style, profile_id=profile_id),
        )

    return ChatResponse(
        user_message=user_message.strip(),
        assistant_message=assistant_message,
        audio_base64=audio_base64,
        voice_error=voice_error,
        metadata=metadata,
    )
