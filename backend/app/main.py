from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app.config import settings
from app.routers import speak, transcribe, voices
from app.services.media_ingest import ensure_ffmpeg, ensure_yt_dlp
from app.services.voicebox_client import VoiceboxClient

health_router = APIRouter(tags=["health"])
voicebox = VoiceboxClient()


@health_router.get("/api/health")
async def health():
    voicebox_ok = False
    voicebox_detail: dict | str = "unreachable"
    try:
        voicebox_detail = await voicebox.health()
        voicebox_ok = True
    except Exception as exc:
        voicebox_detail = str(exc)

    return {
        "status": "ok" if voicebox_ok else "degraded",
        "voicebox": {"ok": voicebox_ok, "detail": voicebox_detail},
        "tools": {
            "ffmpeg": ensure_ffmpeg(),
            "yt_dlp": ensure_yt_dlp(),
        },
        "ollama_configured": bool(settings.ollama_api_key),
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Ultron Voice API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(voices.router)
    app.include_router(transcribe.router)
    app.include_router(speak.router)
    return app


app = create_app()
