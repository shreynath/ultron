from __future__ import annotations

# Ultron UI labels -> Voicebox API engine IDs
ENGINE_TO_VOICEBOX: dict[str, str] = {
    "qwen": "qwen",
    "qwen3-tts": "qwen",
    "qwen_custom_voice": "qwen_custom_voice",
    "luxtts": "luxtts",
    "chatterbox": "chatterbox",
    "chatterbox-multilingual": "chatterbox",
    "chatterbox_turbo": "chatterbox_turbo",
    "chatterbox-turbo": "chatterbox_turbo",
    "tada": "tada",
    "kokoro": "kokoro",
}

DEFAULT_ENGINE = "qwen"


def normalize_engine(engine: str | None) -> str | None:
    if not engine:
        return None
    return ENGINE_TO_VOICEBOX.get(engine, engine)
