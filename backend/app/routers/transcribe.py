from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.services.ollama_client import OllamaError, refine_transcript
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])
client = VoiceboxClient()


class RefineRequest(BaseModel):
    text: str


def _to_wav(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".wav":
        destination.write_bytes(source.read_bytes())
        return
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(destination),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail="Could not process audio recording")


@router.post("")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("turbo"),
):
    work_dir = Path(settings.data_dir) / "transcribe" / str(uuid.uuid4())
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    upload_path = work_dir / f"recording{suffix}"
    upload_path.write_bytes(await file.read())
    wav_path = work_dir / "recording.wav"

    try:
        _to_wav(upload_path, wav_path)
        text = await client.transcribe(wav_path, model=model)
        return {"text": text}
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc


@router.post("/refine")
async def refine_text(body: RefineRequest):
    try:
        refined = await refine_transcript(body.text)
        return {"text": refined}
    except OllamaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
