from __future__ import annotations

import numpy as np

FMIN = 55.0
FMAX = 185.0
HOP_SEC = 0.01
FRAME_SEC = 0.04


def track_f0(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_sec, f0_hz) with 0 for unvoiced frames. Prefers Praat."""
    if audio.size < sample_rate // 20:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    try:
        return _track_f0_praat(audio, sample_rate)
    except Exception:
        return _track_f0_ac(audio, sample_rate)


def _track_f0_praat(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    import parselmouth

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
    pitch = sound.to_pitch(time_step=HOP_SEC, pitch_floor=FMIN, pitch_ceiling=FMAX)
    times = pitch.xs()
    values = pitch.selected_array["frequency"]
    values = np.where(np.isfinite(values), values, 0.0)
    return times.astype(np.float64), values.astype(np.float64)


def _track_f0_ac(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame = int(sample_rate * FRAME_SEC)
    hop = int(sample_rate * HOP_SEC)
    minlag, maxlag = int(sample_rate / FMAX), int(sample_rate / FMIN)
    win = np.hanning(frame)
    times: list[float] = []
    values: list[float] = []
    for start in range(0, max(1, len(audio) - frame), hop):
        window = audio[start : start + frame] * win
        times.append((start + frame / 2) / sample_rate)
        rms = float(np.sqrt(np.mean(window**2)))
        if rms < 0.012:
            values.append(0.0)
            continue
        corr = np.correlate(window, window, mode="full")[frame - 1 :][: maxlag + 1]
        lag = int(np.argmax(corr[minlag:]) + minlag)
        if corr[lag] / (corr[0] + 1e-12) < 0.38:
            values.append(0.0)
            continue
        f0 = sample_rate / lag
        values.append(float(f0) if FMIN <= f0 <= FMAX else 0.0)
    return np.array(times, dtype=np.float64), np.array(values, dtype=np.float64)


def track_f0_range(
    audio: np.ndarray,
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """F0 in an explicit range. Used to tell Ultron (~70 Hz) from another talker."""
    if audio.size < sample_rate // 20 or pitch_floor >= pitch_ceiling:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    try:
        import parselmouth

        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
        pitch = sound.to_pitch(
            time_step=HOP_SEC, pitch_floor=float(pitch_floor), pitch_ceiling=float(pitch_ceiling)
        )
        times = pitch.xs()
        values = pitch.selected_array["frequency"]
        values = np.where(np.isfinite(values), values, 0.0)
        return times.astype(np.float64), values.astype(np.float64)
    except Exception:
        return _track_f0_ac_range(audio, sample_rate, pitch_floor, pitch_ceiling)


def _track_f0_ac_range(
    audio: np.ndarray,
    sample_rate: int,
    pitch_floor: float,
    pitch_ceiling: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(sample_rate * FRAME_SEC)
    hop = int(sample_rate * HOP_SEC)
    minlag, maxlag = int(sample_rate / pitch_ceiling), int(sample_rate / pitch_floor)
    if minlag < 2 or maxlag <= minlag:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    win = np.hanning(frame)
    times: list[float] = []
    values: list[float] = []
    for start in range(0, max(1, len(audio) - frame), hop):
        window = audio[start : start + frame] * win
        times.append((start + frame / 2) / sample_rate)
        rms = float(np.sqrt(np.mean(window**2)))
        if rms < 0.012:
            values.append(0.0)
            continue
        corr = np.correlate(window, window, mode="full")[frame - 1 :][: maxlag + 1]
        lag = int(np.argmax(corr[minlag:]) + minlag)
        if corr[lag] / (corr[0] + 1e-12) < 0.38:
            values.append(0.0)
            continue
        f0 = sample_rate / lag
        values.append(float(f0) if pitch_floor <= f0 <= pitch_ceiling else 0.0)
    return np.array(times, dtype=np.float64), np.array(values, dtype=np.float64)


def voiced_median(f0: np.ndarray) -> float:
    voiced = f0[f0 > 1.0]
    if voiced.size == 0:
        return 0.0
    return float(np.median(voiced))


def resample_contour(values: np.ndarray, n_out: int) -> np.ndarray:
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n_out, dtype=np.float64)
    if values.size == 1:
        return np.full(n_out, values[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, values.astype(np.float64))
