from __future__ import annotations

import numpy as np

_FRAME_MS = 20
_SMOOTH_MS = 50
_VIBRATO_RATE_HZ = 3.2
_VIBRATO_DEPTH = 0.12
_MAX_VIBRATO_MIX = 0.22
_RUMBLE_RATE_HZ = 1.65
_RUMBLE_DEPTH = 0.08


def _frame_rms(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    frame_len = max(1, int(sample_rate * _FRAME_MS / 1000))
    n_frames = max(1, len(audio) // frame_len)
    rms = np.zeros(n_frames, dtype=np.float32)
    for index in range(n_frames):
        frame = audio[index * frame_len : (index + 1) * frame_len]
        rms[index] = float(np.sqrt(np.mean(frame**2)))
    return rms, frame_len


def rasp_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-sample 0–1 mask where the voice is roughest / most metallic."""
    if audio.size < sample_rate // 20:
        return np.zeros_like(audio)

    rms, frame_len = _frame_rms(audio, sample_rate)
    frame_scores = np.zeros(rms.size, dtype=np.float32)
    for index, energy in enumerate(rms):
        start = index * frame_len
        end = min(len(audio), start + frame_len)
        frame = audio[start:end]
        if frame.size < 2:
            continue
        hf = np.diff(frame, prepend=frame[0])
        hf_rms = float(np.sqrt(np.mean(hf**2)))
        ratio = hf_rms / max(energy, 1e-6)
        frame_scores[index] = float(np.clip((ratio - 0.42) / 0.38, 0.0, 1.0))

    envelope = np.zeros(len(audio), dtype=np.float32)
    for index, score in enumerate(frame_scores):
        start = index * frame_len
        end = min(len(audio), start + frame_len)
        if start >= end:
            break
        envelope[start:end] = score

    smooth = max(1, int(sample_rate * _SMOOTH_MS / 1000))
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    smoothed = np.convolve(envelope, kernel, mode="same")
    return smoothed[: len(audio)].astype(np.float32)


def _one_pole_lowpass(audio: np.ndarray, sample_rate: int, freq: float) -> np.ndarray:
    rc = 1.0 / (2.0 * np.pi * max(40.0, freq))
    alpha = (1.0 / sample_rate) / (rc + 1.0 / sample_rate)
    acc = 0.0
    out = np.empty_like(audio, dtype=np.float32)
    a = float(alpha)
    b = 1.0 - a
    for index, sample in enumerate(audio):
        acc = a * float(sample) + b * acc
        out[index] = acc
    return out


def apply_chest_body(
    audio: np.ndarray,
    sample_rate: int,
    *,
    intensity: float = 0.85,
) -> np.ndarray:
    """Chest depth + chassis vibration on the low band only — voice body stays one take."""
    if audio.size < sample_rate // 8:
        return audio

    amount = float(np.clip(intensity, 0.0, 1.0))
    low = _one_pole_lowpass(audio, sample_rate, 88.0)
    low = _one_pole_lowpass(low, sample_rate, 72.0)
    time = np.arange(len(audio), dtype=np.float32) / sample_rate
    if len(audio) / max(1, sample_rate) > 8.0:
        lfo = np.ones(len(audio), dtype=np.float32)
    else:
        lfo = 0.88 + 0.12 * np.sin(2.0 * np.pi * 1.85 * time)
    rms, frame_len = _frame_rms(audio, sample_rate)
    gate = np.zeros(len(audio), dtype=np.float32)
    for index, energy in enumerate(rms):
        start = index * frame_len
        end = min(len(audio), start + frame_len)
        gate[start:end] = float(np.clip((energy - 0.028) / 0.06, 0.0, 1.0))
    layer = low * lfo * gate
    mix = 0.14 * amount
    return (audio + mix * layer).astype(np.float32)


def apply_chest_rumble(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Slow amplitude throb — chassis vibration, not a sung vibrato."""
    if audio.size < sample_rate // 10:
        return audio
    time = np.arange(len(audio), dtype=np.float32) / sample_rate
    lfo = 1.0 - _RUMBLE_DEPTH * (0.5 + 0.5 * np.sin(2.0 * np.pi * _RUMBLE_RATE_HZ * time))
    return audio * lfo.astype(np.float32)


def apply_sinister_vibrato(
    audio: np.ndarray,
    sample_rate: int,
    *,
    intensity: float = 0.5,
) -> np.ndarray:
    """Light metallic rasp on noisy frames only — no pitch warble, no chassis LFO."""
    if audio.size < sample_rate // 10:
        return audio

    envelope = rasp_envelope(audio, sample_rate)
    if float(np.max(envelope)) < 0.08:
        return audio

    time = np.arange(len(audio), dtype=np.float32) / sample_rate
    depth = _VIBRATO_DEPTH * 0.45 * float(np.clip(intensity, 0.0, 1.0))
    lfo = 1.0 - depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * _VIBRATO_RATE_HZ * time))
    modulated = audio * lfo.astype(np.float32)
    mix = envelope * min(0.14, _MAX_VIBRATO_MIX * 0.5)
    return audio * (1.0 - mix) + modulated * mix


def apply_energy_contour(
    audio: np.ndarray,
    sample_rate: int,
    contour: str = "swell",
    intensity: float = 0.5,
) -> np.ndarray:
    """Phrase-shaped gain so energy moves instead of sitting flat."""
    if audio.size < sample_rate // 12:
        return audio

    t = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    if contour == "punch":
        env = 1.14 - 0.20 * t
    elif contour == "land":
        env = 0.86 + 0.30 * (t ** 1.35)
    elif contour == "rise":
        env = 0.90 + 0.24 * t
    elif contour == "wound":
        env = 0.98 - 0.12 * np.sin(np.pi * t)
    else:
        env = 0.88 + 0.24 * np.sin(np.pi * t)
    amount = 0.55 + 0.45 * float(np.clip(intensity, 0.0, 1.0))
    shaped = 1.0 + (env - 1.0) * amount
    return audio * shaped.astype(np.float32)
