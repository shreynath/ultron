from __future__ import annotations

import numpy as np

from app.services.f0 import FMAX, FMIN


def impose_f0_contour(
    audio: np.ndarray,
    sample_rate: int,
    times: np.ndarray,
    target_f0: np.ndarray,
) -> np.ndarray:
    """Replace the pitch of this take with target_f0 (Hz, 0=unvoiced) via Praat PSOLA."""
    if audio.size < sample_rate // 10 or times.size == 0:
        return audio

    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return audio

    try:
        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
        duration = float(sound.xmax - sound.xmin)
        manipulation = call(sound, "To Manipulation", 0.01, FMIN, FMAX)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Remove points between", 0.0, duration)

        added = 0
        for time, freq in zip(times, target_f0):
            hz = float(freq)
            if hz < FMIN or hz > FMAX:
                continue
            t = float(np.clip(time, 0.001, max(0.002, duration - 0.001)))
            call(pitch_tier, "Add point", t, hz)
            added += 1
        if added < 4:
            return audio

        call([pitch_tier, manipulation], "Replace pitch tier")
        out = call(manipulation, "Get resynthesis (overlap-add)")
        values = np.array(out.values, dtype=np.float64)
        if values.ndim == 2:
            values = values[0]
        n = min(len(audio), values.size)
        result = audio.copy()
        result[:n] = values[:n].astype(audio.dtype)
        return result
    except Exception:
        return audio


def lengthen_keep_pitch(
    audio: np.ndarray,
    sample_rate: int,
    factor: float,
    *,
    wide: bool = False,
) -> np.ndarray:
    """Change duration without retuning F0 (Praat overlap-add)."""
    lo, hi = (0.40, 2.50) if wide else (0.82, 1.22)
    factor = float(np.clip(factor, lo, hi))
    if abs(factor - 1.0) < 0.02 or audio.size < sample_rate // 8:
        return audio
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return audio
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
    try:
        out = call(sound, "Lengthen (overlap-add)", FMIN, FMAX, factor)
    except Exception:
        return audio
    values = np.array(out.values, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    return values.astype(np.float32)
