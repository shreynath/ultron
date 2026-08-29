from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.clip_extractor import extract_best_clips
from app.services.media_ingest import ensure_ffmpeg, ensure_yt_dlp, ingest_media
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/voices", tags=["voices"])
client = VoiceboxClient()


class CloneFromUrlRequest(BaseModel):
    url: str
    name: str = Field(min_length=1, max_length=120)
    engine: str = "qwen3-tts"
    language: str = "en"


class CloneResponse(BaseModel):
    profile_id: str
    name: str
    samples_added: int
    engine: str


@router.get("")
async def list_voices():
    try:
        profiles = await client.list_profiles()
        return {"profiles": profiles}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Voicebox unavailable: {exc}") from exc


@router.post("/clone")
async def clone_from_url(body: CloneFromUrlRequest):
    work_dir = Path(settings.data_dir) / "ingest" / str(uuid.uuid4())
    try:
        audio_path = ingest_media(url=body.url, work_dir=work_dir)
        return await _create_profile_from_audio(
            audio_path=audio_path,
            name=body.name,
            engine=body.engine,
            language=body.language,
            work_dir=work_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/clone/upload")
async def clone_from_upload(
    name: str = Form(...),
    engine: str = Form("qwen3-tts"),
    language: str = Form("en"),
    file: UploadFile = File(...),
):
    work_dir = Path(settings.data_dir) / "ingest" / str(uuid.uuid4())
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    upload_path = work_dir / f"upload{suffix}"
    upload_path.write_bytes(await file.read())

    try:
        audio_path = ingest_media(upload_path=upload_path, work_dir=work_dir)
        return await _create_profile_from_audio(
            audio_path=audio_path,
            name=name,
            engine=engine,
            language=language,
            work_dir=work_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{profile_id}")
async def delete_voice(profile_id: str):
    try:
        await client.delete_profile(profile_id)
        return {"deleted": True, "profile_id": profile_id}
    except VoiceboxError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc


async def _create_profile_from_audio(
    *,
    audio_path: Path,
    name: str,
    engine: str,
    language: str,
    work_dir: Path,
) -> CloneResponse:
    clips_dir = work_dir / "clips"
    clips = extract_best_clips(audio_path, clips_dir)

    profile = await client.create_profile(
        name=name,
        engine=engine,
        language=language,
        description="Created by Ultron voice ingestion pipeline",
    )
    profile_id = profile.get("id") or profile.get("profile_id")
    if not profile_id:
        raise HTTPException(status_code=502, detail=f"Unexpected profile response: {profile}")

    samples_added = 0
    for clip in clips:
        transcript = await client.transcribe(clip.path)
        if not transcript.strip():
            continue
        await client.add_profile_sample(profile_id, clip.path, transcript.strip())
        samples_added += 1

    if samples_added == 0:
        await client.delete_profile(profile_id)
        raise HTTPException(status_code=400, detail="No usable speech clips found in the source audio")

    return CloneResponse(
        profile_id=profile_id,
        name=name,
        samples_added=samples_added,
        engine=engine,
    )


@router.get("/tools/status")
async def tools_status():
    return {
        "ffmpeg": ensure_ffmpeg(),
        "yt_dlp": ensure_yt_dlp(),
        "supports_youtube": ensure_yt_dlp(),
    }
