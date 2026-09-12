from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_FRAME_MS = 20
_RASP_PERCENTILE = 58.0


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "voicebox"


def _apply_chorus(
    audio: np.ndarray,
    sample_rate: int,
    *,
    rate_hz: float,
    depth: float,
    mix: float,
) -> np.ndarray:
    vendor_root = str(_vendor_root())
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)
    from backend.utils.effects import apply_effects

    chain: list[dict[str, Any]] = [
        {
            "type": "chorus",
            "enabled": True,
            "params": {
                "rate_hz": rate_hz,
                "depth": depth,
                "feedback": 0.14,
                "centre_delay_ms": 6.5,
                "mix": mix,
            },
        }
    ]
    return apply_effects(audio, sample_rate, chain)


def _rasp_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = max(1, int(sample_rate * _FRAME_MS / 1000))
    if audio.size < frame_len * 2:
        return np.zeros(audio.size, dtype=np.float32)

    frame_scores: list[float] = []
    for start in range(0, len(audio) - frame_len + 1, frame_len):
        chunk = audio[start : start + frame_len]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < 1e-4:
            frame_scores.append(0.0)
            continue
        hf = float(np.sqrt(np.mean(np.diff(chunk) ** 2)))
        frame_scores.append(hf / rms)

    if not frame_scores:
        return np.zeros(audio.size, dtype=np.float32)

    threshold = float(np.percentile(frame_scores, _RASP_PERCENTILE))
    envelope = np.zeros(len(audio), dtype=np.float32)
    span = max(threshold, 1e-6)
    for index, score in enumerate(frame_scores):
        if score <= threshold:
            continue
        weight = min(1.0, (score - threshold) / span)
        start = index * frame_len
        end = min(len(audio), start + frame_len)
        envelope[start:end] = weight

    smooth = max(1, int(sample_rate * 0.012))
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    return np.convolve(envelope, kernel, mode="same")


def apply_sinister_vibration(
    audio: np.ndarray,
    sample_rate: int,
    *,
    intensity: float = 0.5,
) -> np.ndarray:
    """Add slow metallic vibration only where the clone already sounds raspy."""
    if audio.size < sample_rate // 8:
        return audio

    intensity = max(0.0, min(1.0, intensity))
    max_blend = 0.07 + intensity * 0.15
    envelope = _rasp_envelope(audio, sample_rate)
    if float(np.max(envelope)) < 0.05:
        return audio

    envelope = np.clip(envelope * max_blend, 0.0, max_blend)
    rate_hz = 0.11 + intensity * 0.07
    depth = 0.22 + intensity * 0.18
    chorused = _apply_chorus(audio, sample_rate, rate_hz=rate_hz, depth=depth, mix=0.55)
    length = min(len(audio), len(chorused))
    return audio[:length] * (1.0 - envelope[:length]) + chorused[:length] * envelope[:length]
