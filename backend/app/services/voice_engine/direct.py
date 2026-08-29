from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_initialized = False
_init_lock = asyncio.Lock()


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[4] / "vendor" / "voicebox"


def _ensure_vendor_path() -> None:
    vendor_root = str(_vendor_root())
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)


async def init_direct_engine() -> None:
    global _initialized
    async with _init_lock:
        if _initialized:
            return
        _ensure_vendor_path()

        from backend import config, database
        from backend.services.task_queue import init_queue

        data_dir = settings.voicebox_data_dir
        if not data_dir:
            default_app_data = Path.home() / "Library/Application Support/sh.voicebox.app"
            data_dir = str(default_app_data if default_app_data.exists() else Path(settings.data_dir) / "voicebox")

        config.set_data_dir(data_dir)
        database.init_db()
        init_queue()
        _initialized = True
        logger.info("Direct voice engine initialized (data-dir=%s)", data_dir)


async def shutdown_direct_engine() -> None:
    global _initialized
    if not _initialized:
        return
    _ensure_vendor_path()
    try:
        from backend.services import llm, transcribe, tts

        tts.unload_tts_model()
        transcribe.unload_whisper_model()
        llm.unload_llm_model()
    except Exception as exc:
        logger.warning("Direct engine shutdown issue: %s", exc)
    _initialized = False


async def direct_generate(
    *,
    text: str,
    profile_id: str,
    language: str = "en",
    engine: str | None = None,
    instruct: str | None = None,
    effects_chain: list[dict[str, Any]] | None = None,
) -> bytes:
    await init_direct_engine()
    _ensure_vendor_path()

    from backend.services import tts
    from backend.services.generation import generate_audio_sync
    from backend.utils.audio import load_audio
    from backend.utils.effects import apply_effects

    normalized_engine = engine or "qwen"
    audio_bytes = await generate_audio_sync(
        profile_id=profile_id,
        text=text,
        language=language,
        engine=normalized_engine,
        model_size="1.7B",
        instruct=instruct,
        normalize=True,
    )

    if effects_chain:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio, sample_rate = load_audio(tmp.name)
            audio = apply_effects(audio, sample_rate, effects_chain)
            audio_bytes = tts.audio_to_wav_bytes(audio, sample_rate)

    return audio_bytes


async def direct_transcribe(audio_path: Path, model: str = "turbo") -> str:
    await init_direct_engine()
    _ensure_vendor_path()

    from backend.backends import get_stt_backend, load_stt_model

    backend = get_stt_backend()
    await load_stt_model(model)
    return await backend.transcribe(str(audio_path), model_size=model)
