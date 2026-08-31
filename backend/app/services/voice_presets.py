from __future__ import annotations

from typing import Any

ULTRON_INSTRUCT = (
    "Speak as a wry, precise, slightly metallic human baritone. "
    "Conversational and theatrical. Keep contractions. Pause at periods. "
    "Do not growl, cry, or cartoon the voice."
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


ULTRON_REFERENCE_INSTRUCT = (
    "Ultron. Same cloned voice, unhurried, slightly metallic. "
    "Pause after every period. Keep contractions. Do not cry or growl."
)


def ultron_theater_effects_chain() -> list[dict[str, Any]]:
    """Keep the clone; only a little room. Chorus doubles F0 and kills the Spader timbre."""
    return [
        {
            "type": "compressor",
            "enabled": True,
            "params": {
                "threshold_db": -18.0,
                "ratio": 1.6,
                "attack_ms": 32.0,
                "release_ms": 180.0,
            },
        },
        {
            "type": "reverb",
            "enabled": True,
            "params": {
                "room_size": 0.34,
                "damping": 0.52,
                "wet_level": 0.07,
                "dry_level": 0.93,
                "width": 1.0,
            },
        },
    ]


def theater_clean_effects_chain() -> list[dict[str, Any]]:
    """Minimal chain for theater mode — clone timbre is enough; avoid reverb/compression artifacts."""
    return [
        {
            "type": "highpass",
            "enabled": True,
            "params": {"cutoff_frequency_hz": 92.0},
        },
    ]


def reference_match_effects_chain() -> list[dict[str, Any]]:
    """Light touch-up only — reference audio already has cinematic processing."""
    return [
        {
            "type": "pitch_shift",
            "enabled": True,
            "params": {"semitones": -1.5},
        },
        {
            "type": "reverb",
            "enabled": True,
            "params": {
                "room_size": 0.55,
                "damping": 0.5,
                "wet_level": 0.22,
                "dry_level": 0.78,
                "width": 1.0,
            },
        },
        {
            "type": "compressor",
            "enabled": True,
            "params": {
                "threshold_db": -22.0,
                "ratio": 3.5,
                "attack_ms": 12.0,
                "release_ms": 160.0,
            },
        },
        {
            "type": "gain",
            "enabled": True,
            "params": {"gain_db": 2.0},
        },
    ]


def resolve_voice_delivery(
    voice_style: str,
    *,
    profile_delivery: dict | None = None,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    if voice_style == "natural":
        return None, None
    if voice_style == "reference":
        return ULTRON_REFERENCE_INSTRUCT, reference_match_effects_chain()
    if voice_style == "theater_auto":
        return ULTRON_REFERENCE_INSTRUCT, ultron_theater_effects_chain()
    return ULTRON_INSTRUCT, ultron_cinematic_effects_chain()
