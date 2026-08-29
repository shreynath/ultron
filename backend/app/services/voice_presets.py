from __future__ import annotations

from typing import Any

ULTRON_INSTRUCT = (
    "Speak slowly, deeply, and deliberately with weight and authority. "
    "Use a low, theatrical, menacing tone. Pause slightly between phrases."
)


def ultron_cinematic_effects_chain() -> list[dict[str, Any]]:
    """Voicebox effects chain tuned for deep, reverberant, slightly modulated speech."""
    return [
        {
            "type": "pitch_shift",
            "enabled": True,
            "params": {"semitones": -4.5},
        },
        {
            "type": "chorus",
            "enabled": True,
            "params": {
                "rate_hz": 0.18,
                "depth": 0.65,
                "feedback": 0.28,
                "centre_delay_ms": 8.0,
                "mix": 0.38,
            },
        },
        {
            "type": "reverb",
            "enabled": True,
            "params": {
                "room_size": 0.78,
                "damping": 0.42,
                "wet_level": 0.4,
                "dry_level": 0.6,
                "width": 1.0,
            },
        },
        {
            "type": "compressor",
            "enabled": True,
            "params": {
                "threshold_db": -20.0,
                "ratio": 4.5,
                "attack_ms": 8.0,
                "release_ms": 140.0,
            },
        },
        {
            "type": "gain",
            "enabled": True,
            "params": {"gain_db": 3.5},
        },
    ]


def resolve_voice_delivery(voice_style: str) -> tuple[str | None, list[dict[str, Any]] | None]:
    if voice_style == "natural":
        return None, None
    return ULTRON_INSTRUCT, ultron_cinematic_effects_chain()
