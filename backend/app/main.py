from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import brain, chat, local_intelligence, speak, transcribe, voices
from app.services.media_ingest import ensure_ffmpeg, ensure_yt_dlp
from app.services.voice_engine import get_engine_manager, shutdown_direct_engine
from app.services.voicebox_client import VoiceboxClient

logger = logging.getLogger(__name__)

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
        "voicebox": {
            "ok": voicebox_ok,
            "detail": voicebox_detail,
            "mode": settings.voice_engine_mode,
        },
        "tools": {
            "ffmpeg": ensure_ffmpeg(),
            "yt_dlp": ensure_yt_dlp(),
        },
        "ollama_configured": False,
        "brain": {
            "ok": True,
            "model": settings.brain_model or "local:general",
        },
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.voice_engine_mode == "embedded":
            try:
                await get_engine_manager().ensure_running()
                logger.info("Embedded voice engine is ready")
            except Exception as exc:
                logger.warning("Embedded voice engine not ready at startup: %s", exc)
        elif settings.voice_engine_mode == "direct":
            from app.services.voice_engine import init_direct_engine

            await init_direct_engine()
            logger.info("Direct voice engine initialized")
        yield
        if settings.voice_engine_mode == "embedded":
            await get_engine_manager().shutdown()
        elif settings.voice_engine_mode == "direct":
            await shutdown_direct_engine()

    app = FastAPI(title="Ultron Voice API", version="0.2.0", lifespan=lifespan)
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

    @app.middleware("http")
    async def basic_deployment_guards(request: Request, call_next):
        """Small framework-level guard; privileged research has its own token."""
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.api_max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    app.include_router(health_router)
    app.include_router(brain.router)
    app.include_router(local_intelligence.router)
    app.include_router(voices.router)
    app.include_router(transcribe.router)
    app.include_router(chat.router)
    app.include_router(speak.router)
    return app


app = create_app()
