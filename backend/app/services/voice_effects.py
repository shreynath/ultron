from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[3] / "vendor" / "voicebox"


def _partition_chain(effects_chain: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    pre_reverb: list[dict] = []
    gain: list[dict] = []
    reverb: list[dict] = []
    for effect in effects_chain:
        if not effect.get("enabled", True):
            continue
        etype = effect.get("type")
        if etype == "reverb":
            reverb.append(effect)
        elif etype == "gain":
            gain.append(effect)
        else:
            pre_reverb.append(effect)
    return pre_reverb, gain, reverb


def _delayed_blend(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
    *,
    delay_ms: int,
    ramp_ms: int,
) -> np.ndarray:
    length = min(len(dry), len(wet))
    if length == 0:
        return wet
    dry = dry[:length]
    wet = wet[:length]
    out = dry.copy()

    delay = min(int(sample_rate * delay_ms / 1000), length)
    ramp = min(int(sample_rate * ramp_ms / 1000), max(0, length - delay))
    if delay >= length:
        return dry

    end = delay + ramp
    if ramp > 0:
        curve = np.sin(np.linspace(0.0, np.pi / 2.0, ramp, dtype=np.float32)) ** 2
        out[delay:end] = dry[delay:end] * (1.0 - curve) + wet[delay:end] * curve
    if end < length:
        out[end:] = wet[end:]
    return out


def apply_effects_to_wav(audio_bytes: bytes, effects_chain: list[dict[str, Any]]) -> bytes:
    """Apply the chain evenly across the take so the voice does not change character mid-line."""
    if not effects_chain:
        return audio_bytes

    vendor_root = str(_vendor_root())
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    try:
        from backend.utils.effects import apply_effects
    except ModuleNotFoundError:
        return audio_bytes

    audio, sample_rate = wav_bytes_to_mono_float(audio_bytes)
    pre_reverb, gain, reverb = _partition_chain(effects_chain)

    body = apply_effects(audio, sample_rate, pre_reverb) if pre_reverb else audio
    if gain:
        body = apply_effects(body, sample_rate, gain)
    if reverb:
        body = apply_effects(body, sample_rate, reverb)
    return mono_float_to_wav_bytes(body, sample_rate)
