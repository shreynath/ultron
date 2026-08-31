from __future__ import annotations

import numpy as np

from app.services.f0 import FMAX, FMIN

LTAS_BANDS: tuple[tuple[float, float], ...] = (
    (60.0, 90.0),
    (90.0, 130.0),
    (130.0, 180.0),
    (180.0, 250.0),
    (250.0, 350.0),
    (350.0, 500.0),
    (500.0, 750.0),
    (750.0, 1100.0),
    (1100.0, 1600.0),
    (1600.0, 2400.0),
    (2400.0, 3500.0),
    (3500.0, 5500.0),
)

# Chest can come up from the clip. Metal ring at 1.6–3.5 kHz was capped at +1.2 dB
# while the film sits ~8 dB brighter there — that is the "shallow / not the same" hole.
_MAX_BOOST = np.array([8.0, 5.0, 3.0, 1.5, 0.5, 0.3, 0.8, 1.5, 4.5, 8.5, 6.0, 6.5])
_MAX_CUT = np.array([2.0, 3.5, 5.5, 5.5, 5.0, 4.0, 2.0, 2.0, 2.0, 1.5, 2.0, 2.5])


def measure_ltas_db(audio: np.ndarray, sample_rate: int) -> list[float]:
    """Median log-power per band on frames that look like speech, not score hits."""
    if audio.size < sample_rate // 20:
        return [0.0] * len(LTAS_BANDS)

    frame = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.01)
    win = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    bands = [(freqs >= lo) & (freqs < hi) for lo, hi in LTAS_BANDS]
    acc = [[] for _ in LTAS_BANDS]
    for start in range(0, max(1, len(audio) - frame), hop):
        window = audio[start : start + frame] * win
        rms = float(np.sqrt(np.mean(window**2)))
        if rms < 0.02:
            continue
        power = np.abs(np.fft.rfft(window)) ** 2
        voice = float(np.mean(power[(freqs >= 80.0) & (freqs < 400.0)]))
        air = float(np.mean(power[(freqs >= 400.0) & (freqs < 4000.0)]))
        if voice < air * 0.15:
            continue
        for index, mask in enumerate(bands):
            acc[index].append(float(np.mean(power[mask]) + 1e-18))
    out: list[float] = []
    for values in acc:
        out.append(float(10.0 * np.log10(np.median(values))) if values else 0.0)
    return out


def _gain_curve(freqs: np.ndarray, gains_db: np.ndarray) -> np.ndarray:
    centers = np.array([(lo + hi) / 2.0 for lo, hi in LTAS_BANDS], dtype=np.float64)
    padded_f = np.concatenate([[0.0], centers, [float(freqs[-1]) if freqs.size else 8000.0]])
    padded_g = np.concatenate([[gains_db[0]], gains_db, [gains_db[-1]]])
    return np.interp(freqs, padded_f, padded_g)


def apply_ltas_match(
    audio: np.ndarray,
    sample_rate: int,
    target_ltas_db: list[float],
    *,
    loose: bool = False,
) -> np.ndarray:
    """Move this take's voiced spectrum toward Ultron's measured bands."""
    if audio.size < sample_rate // 20 or len(target_ltas_db) != len(LTAS_BANDS):
        return audio
    source = np.array(measure_ltas_db(audio, sample_rate), dtype=np.float64)
    target = np.array(target_ltas_db, dtype=np.float64)
    if float(np.max(np.abs(target))) < 1.0 or float(np.max(np.abs(source))) < 1.0:
        return audio

    delta = target - source
    peak_t = float(np.max(target))
    measured = target > (peak_t - 36.0)
    delta = np.where(measured, delta, 0.0)
    boost = np.array([20.0, 16.0, 14.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]) if loose else _MAX_BOOST
    cut = np.full(len(LTAS_BANDS), 14.0) if loose else _MAX_CUT
    gains = np.clip(delta, -cut, boost)
    if float(np.max(np.abs(gains))) < 0.35:
        return audio

    n_fft = 2048
    hop = 512
    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mag_gain = 10.0 ** (_gain_curve(freqs, gains) / 20.0)

    padded = np.pad(audio.astype(np.float64), n_fft, mode="constant")
    acc = np.zeros(len(padded), dtype=np.float64)
    weight = np.zeros(len(padded), dtype=np.float64)
    for start in range(0, len(padded) - n_fft + 1, hop):
        frame = padded[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        spec *= mag_gain
        acc[start : start + n_fft] += np.fft.irfft(spec, n=n_fft) * window
        weight[start : start + n_fft] += window**2
    weight = np.maximum(weight, 1e-9)
    out = (acc / weight)[n_fft : n_fft + len(audio)]
    if not loose:
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        src_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6 and src_peak > 1e-6:
            out *= src_peak / peak
    return out.astype(audio.dtype)


def measure_formants(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    if audio.size < sample_rate // 8:
        return 0.0, 0.0
    try:
        import parselmouth
    except Exception:
        return 0.0, 0.0

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
    formant = sound.to_formant_burg(
        time_step=0.01, max_number_of_formants=5, maximum_formant=5000.0
    )
    f1: list[float] = []
    f2: list[float] = []
    for time in formant.xs():
        one = formant.get_value_at_time(1, time)
        two = formant.get_value_at_time(2, time)
        if one is not None and np.isfinite(one) and 200.0 < one < 900.0:
            f1.append(float(one))
        if two is not None and np.isfinite(two) and 600.0 < two < 2500.0:
            f2.append(float(two))
    return (
        float(np.median(f1)) if f1 else 0.0,
        float(np.median(f2)) if f2 else 0.0,
    )


def shift_formants(audio: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
    """Darken/brighten the vocal tract toward Ultron without retuning F0."""
    if audio.size < sample_rate // 8 or abs(ratio - 1.0) < 0.02:
        return audio
    ratio = float(np.clip(ratio, 0.86, 1.10))
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        return audio

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sample_rate))
    try:
        out = call(sound, "Change gender", FMIN, ratio, 0.0, 1.0, 1.0)
    except Exception:
        try:
            out = call(sound, "Change speaker", FMIN, FMAX, ratio, 0.0, 1.0)
        except Exception:
            return audio
    values = np.array(out.values, dtype=np.float64)
    if values.ndim == 2:
        values = values[0]
    result = audio.copy()
    n = min(len(audio), values.size)
    result[:n] = values[:n].astype(audio.dtype)
    return result


def darken_air(audio: np.ndarray, sample_rate: int, amount: float = 0.0) -> np.ndarray:
    """Kept for tests. A blanket lowpass made Ultron muffled — do not use on the hot path."""
    del amount
    return audio


def apply_ultron_metal(audio: np.ndarray, sample_rate: int, amount: float = 0.55) -> np.ndarray:
    """Light chassis grit. No lowpass — that is what buried the words."""
    if audio.size < sample_rate // 8:
        return audio
    amt = float(np.clip(amount, 0.0, 1.0))
    drive = 1.25 + 0.45 * amt
    sat = np.tanh(audio.astype(np.float64) * drive) / np.tanh(drive)
    mixed = (1.0 - 0.16 * amt) * audio + (0.16 * amt) * sat.astype(audio.dtype)
    delay = max(2, int(sample_rate * 0.0015))
    comb = np.zeros_like(mixed)
    comb[delay:] = mixed[:-delay]
    win = max(8, int(sample_rate * 0.03))
    kernel = np.ones(win, dtype=np.float64) / win
    env = np.convolve(np.abs(mixed.astype(np.float64)), kernel, mode="same")[: len(mixed)]
    gate = np.clip((env - 0.02) / 0.05, 0.0, 1.0).astype(mixed.dtype)
    out = mixed + (0.07 * amt) * comb * gate
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    src_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6 and src_peak > 1e-6:
        out = out * (src_peak / peak)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
