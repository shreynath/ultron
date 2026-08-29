"""Embedded Voicebox engine integration for Ultron."""

from app.services.voice_engine.direct import (
    direct_generate,
    direct_transcribe,
    init_direct_engine,
    shutdown_direct_engine,
)
from app.services.voice_engine.manager import EmbeddedVoiceEngineManager, get_engine_manager

__all__ = [
    "EmbeddedVoiceEngineManager",
    "direct_generate",
    "direct_transcribe",
    "get_engine_manager",
    "init_direct_engine",
    "shutdown_direct_engine",
]
