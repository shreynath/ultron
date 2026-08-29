from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.clip_extractor import extract_best_clips
from app.services.media_ingest import ensure_ffmpeg, ensure_yt_dlp, ingest_media
from app.services.voice_presets import ultron_cinematic_effects_chain
from app.services.voicebox_client import VoiceboxClient, VoiceboxError

router = APIRouter(prefix="/api/voices", tags=["voices"])
client = VoiceboxClient()


class CloneFromUrlRequest(BaseModel):
    url: str
    name: str = Field(min_length=1, max_length=120)
    engine: str = "qwen"
    language: str = "en"


class CloneResponse(BaseModel):
    profile_id: str
    name: str
    samples_added: int
    engine: str
    quality_warning: str | None = None


@router.get("")
async def list_voices():
    try:
        profiles = await client.list_profiles()
        return {"profiles": profiles}
    except Exception as exc:
        message = str(exc)
        if "connection attempts failed" in message.lower():
            message = (
                "Voicebox is not running. Install from https://voicebox.sh/download/mac-arm "
                "and launch the app."
            )
        raise HTTPException(status_code=502, detail=message) from exc


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
    engine: str = Form("qwen"),
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


def _looks_like_multi_speaker(transcript: str) -> bool:
    lowered = transcript.lower()
    dialogue_markers = [
        " i am ",
        " you are ",
        " he said",
        " she said",
        "hello,",
        "?",
    ]
    hits = sum(1 for marker in dialogue_markers if marker in lowered)
    return hits >= 3 or ("?" in transcript and len(transcript.split()) > 35)


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

    prepared_samples: list[tuple[Path, str]] = []
    errors: list[str] = []
    quality_warnings: list[str] = []
    for clip in clips:
        try:
            transcript = await client.transcribe(clip.path)
            if transcript.strip():
                if _looks_like_multi_speaker(transcript):
                    quality_warnings.append(
                        "One clip sounds like multiple speakers or dialogue. "
                        "For best results, use a source where only the target person talks alone."
                    )
                prepared_samples.append((clip.path, transcript.strip()))
        except VoiceboxError as exc:
            errors.append(str(exc))

    if len(prepared_samples) < 2:
        quality_warnings.append(
            "Only found one strong speech clip. Cloning works best with 3-4 clear solo-speech samples."
        )

    if not prepared_samples:
        detail = "No usable speech clips found in the source audio."
        if errors:
            detail = f"{detail} {errors[0]}"
        raise HTTPException(status_code=400, detail=detail)

    profile_id: str | None = None
    try:
        profile = await client.create_profile(
            name=name,
            language=language,
            description="Created by Ultron voice ingestion pipeline",
        )
        profile_id = profile.get("id") or profile.get("profile_id")
        if not profile_id:
            raise HTTPException(status_code=502, detail=f"Unexpected profile response: {profile}")

        for clip_path, transcript in prepared_samples:
            await client.add_profile_sample(profile_id, clip_path, transcript)

        saved_samples = await client.get_profile_samples(profile_id)
        if not saved_samples:
            raise HTTPException(
                status_code=502,
                detail="Voice samples failed to save in Voicebox. Make sure Voicebox is running and try again.",
            )

        try:
            await client.update_profile_effects(profile_id, ultron_cinematic_effects_chain())
        except VoiceboxError:
            pass

        return CloneResponse(
            profile_id=profile_id,
            name=name,
            samples_added=len(saved_samples),
            engine=engine,
            quality_warning=quality_warnings[0] if quality_warnings else None,
        )
    except Exception:
        if profile_id:
            try:
                await client.delete_profile(profile_id)
            except VoiceboxError:
                pass
        raise


@router.get("/tools/status")
async def tools_status():
    return {
        "ffmpeg": ensure_ffmpeg(),
        "yt_dlp": ensure_yt_dlp(),
        "supports_youtube": ensure_yt_dlp(),
    }
