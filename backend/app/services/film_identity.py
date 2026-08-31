from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from app.services.dramatic_planner import _split_sentences
from app.services.f0 import HOP_SEC, track_f0, track_f0_range
from app.services.living_delivery import detect_quiet_spans
from app.services.td_psola import impose_f0_contour, lengthen_keep_pitch
from app.services.tts_text import normalize_for_tts
from app.services.ultron_atlas import (
    energy_split_ranges,
    film_weighted_split_ranges,
    phrases_from_paragraph,
    profile_atlas_path,
    save_atlas,
    sentence_sample_ranges,
)
from app.services.ultron_tone import apply_ltas_match, measure_formants, measure_ltas_db, shift_formants
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float

_GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "ultron_ref" / "gold"
_MANIFEST = _GOLD_DIR / "manifest.json"
_SOURCE = Path(__file__).resolve().parents[2] / "data" / "ultron_ref" / "source.wav"
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Frame clock shared with F0 tracking — this is the "frame" we match.
FRAME_SEC = HOP_SEC

# Hard gates: a line is not "the same" if any of these fail.
MAX_DURATION_ERR_SEC = 0.03
MAX_F0_MAE_HZ = 8.0
MAX_LTAS_MAE_DB = 2.2
MIN_RMS_CORR = 0.82
MAX_BASS_ERR_DB = 4.0
MAX_VOICED_FRAC_ERR = 0.22
GOLD_COVER_MIN = 0.95

METEOR_TEXT = (
    "I wasn't sure you'd wake up. I hoped you would. I wanted to show you. "
    "I... don't have anyone else. I think a lot about meteors. The purity of them. "
    "Boom! The end. Start again. The world made clean for the new man to rebuild. "
    "I was meant to be new. I was meant to be beautiful. "
    "The world would have looked to the sky and seen hope. Seen mercy. "
    "Instead they'll look up in horror because of you. You've wounded me. "
    "I give you full marks for that. But, like my old man said, "
    "what doesn't kill you... just makes me stronger!"
)

# Compilation windows on source.wav (YouTube Ultron compilation, not movie order).
FILM_WINDOWS: list[tuple[float, float, str]] = [
    (
        366.48,
        384.48,
        "I think a lot about meteors. The purity of them. Boom! The end. Start again. "
        "The world made clean for the new man to rebuild. You've wounded me. "
        "I give you full marks for that.",
    ),
    (
        384.48,
        402.48,
        "But like my old man said, what doesn't kill you just makes me stronger. "
        "I wasn't sure you'd wake up. I hoped you would. I wanted to show you. "
        "I don't have anyone else.",
    ),
    (
        402.48,
        411.48,
        "I was meant to be new. I was meant to be beautiful. "
        "The world would have looked to the sky and seen hope.",
    ),
    (412.18, 413.24, "Seen mercy."),
    (414.12, 417.34, "Instead they'll look up in horror because of you."),
]


@dataclass
class GoldClip:
    clip_id: str
    text: str
    path: Path
    duration_sec: float
    audio: np.ndarray
    sample_rate: int


@dataclass
class FrameReport:
    clip_id: str
    text: str
    duration_err_sec: float
    f0_mae_hz: float
    ltas_mae_db: float
    rms_corr: float
    bass_err_db: float
    voiced_frac_err: float
    n_frames: int
    passed: bool
    failures: list[str]


def gold_dir() -> Path:
    return _GOLD_DIR


def gold_covers_text(text: str, clips: list[GoldClip] | None = None) -> bool:
    return gold_cover_ratio(text, clips) >= GOLD_COVER_MIN


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_for_tts(text).lower())


def _slug(text: str) -> str:
    words = _words(text)
    return "-".join(words[:8]) or "line"


def _resample_rate(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate or audio.size < 8:
        return audio.astype(np.float32)
    n_out = max(8, int(round(len(audio) * target_rate / sample_rate)))
    src = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    dst = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(dst, src, audio.astype(np.float64)).astype(np.float32)


def _time_map(audio: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return audio.astype(np.float32)
    if len(audio) == n:
        return audio.astype(np.float32)
    if audio.size < 8:
        return np.pad(audio.astype(np.float32), (0, max(0, n - len(audio))))[:n]
    src = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    dst = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(dst, src, audio.astype(np.float64)).astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    n = max(1, int(np.ceil(len(audio) / hop)))
    out = np.zeros(n, dtype=np.float64)
    for index in range(n):
        chunk = audio[index * hop : (index + 1) * hop]
        if chunk.size:
            out[index] = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return out


def _dense_env(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    env = _frame_rms(audio, sample_rate)
    if env.size == 0:
        return np.zeros(len(audio), dtype=np.float64)
    centers = (np.arange(len(env)) + 0.5) * hop
    return np.interp(np.arange(len(audio)), centers, env).astype(np.float64)


def _impose_rms(clone: np.ndarray, film: np.ndarray, sample_rate: int) -> np.ndarray:
    film_env = _dense_env(film, sample_rate)
    clone_env = _dense_env(clone, sample_rate)
    n = min(len(film_env), len(clone_env), len(clone))
    if n < 8:
        return clone
    gain = np.clip(film_env[:n] / (clone_env[:n] + 1e-6), 0.0, 8.0)
    out = clone.copy()
    out[:n] = (out[:n].astype(np.float64) * gain).astype(np.float32)
    return out


def _f0_mae(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    voiced = (left[:n] > 1.0) & (right[:n] > 1.0)
    if int(np.sum(voiced)) < 4:
        return 99.0
    return float(np.mean(np.abs(left[:n][voiced] - right[:n][voiced])))


_ULTRON_F0_MAX = 145.0
_OTHER_F0_MIN = 160.0
_OTHER_F0_MAX = 340.0
# Ultron holds a beat after a period. Dropping anything over 450ms was eating those stops.
_MAX_KEEP_PAUSE_SEC = 1.15
_FILL_UNVOICED_SEC = 0.09
_SENTENCE_MATCH = 0.92


def _frame_count(n_audio: int, sample_rate: int) -> tuple[int, int]:
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    return hop, max(1, int(np.ceil(n_audio / hop)))


def _f0_on_hops(times: np.ndarray, f0: np.ndarray, n_frames: int) -> np.ndarray:
    hops = np.zeros(n_frames, dtype=np.float64)
    for time, hz in zip(times, f0):
        index = int(np.clip(round(time / FRAME_SEC), 0, n_frames - 1))
        hops[index] = hz
    return hops


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Keep unvoiced consonants that sit between vowels. Do not fill real pauses."""
    if max_gap < 1 or mask.size == 0:
        return mask
    out = mask.copy()
    index = 0
    n = len(mask)
    while index < n:
        if out[index]:
            index += 1
            continue
        end = index
        while end < n and not mask[end]:
            end += 1
        if index > 0 and end < n and (end - index) <= max_gap:
            out[index:end] = True
        index = end
    return out


def _cap_long_pauses(keep: np.ndarray, is_speech: np.ndarray, first: int, last: int) -> np.ndarray:
    """Keep Ultron's holds as silence. Collapse only scene-length score beds."""
    max_pause = max(1, int(_MAX_KEEP_PAUSE_SEC / FRAME_SEC))
    run_start: int | None = None
    for index in range(first, last + 1):
        silent = keep[index] and not is_speech[index]
        if silent and run_start is None:
            run_start = index
        if (not silent or index == last) and run_start is not None:
            run_end = index if not silent else index + 1
            extra = run_end - run_start - max_pause
            if extra > 0:
                keep[run_start + max_pause : run_end] = False
            run_start = None
    return keep


def _median_noise_mag(
    audio: np.ndarray,
    sample_rate: int,
    is_score: np.ndarray,
    n_fft: int,
    hop: int,
) -> np.ndarray | None:
    if audio.size < n_fft or not np.any(is_score):
        return None
    window = np.hanning(n_fft)
    mags: list[np.ndarray] = []
    for start in range(0, max(1, len(audio) - n_fft + 1), hop):
        cls = int(np.clip(round((start + n_fft / 2) / (sample_rate * FRAME_SEC)), 0, len(is_score) - 1))
        if not is_score[cls]:
            continue
        frame = audio[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        mag = np.abs(np.fft.rfft(frame.astype(np.float64) * window))
        if float(np.mean(mag)) > 1e-6:
            mags.append(mag)
    if len(mags) < 3:
        return None
    return np.median(np.stack(mags), axis=0)


def _peak_hz(freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> float:
    band = (freqs >= lo) & (freqs <= hi)
    if not np.any(band):
        return 0.0
    idx = np.flatnonzero(band)
    peak = int(np.argmax(mag[band]))
    global_k = int(idx[peak])
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    if 0 < global_k < len(mag) - 1:
        a, b, c = float(mag[global_k - 1]), float(mag[global_k]), float(mag[global_k + 1])
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            return float(freqs[global_k] + 0.5 * (a - c) / denom * df)
    return float(freqs[global_k])


def _harmonic_gain(freqs: np.ndarray, f0: float) -> np.ndarray:
    gain = np.full(len(freqs), 0.08, dtype=np.float64)
    gain[freqs < 90.0] = 0.92
    for harmonic in range(1, 14):
        center = f0 * harmonic
        if center > (freqs[-1] if freqs.size else 0):
            break
        width = max(10.0, min(16.0, 0.028 * center))
        gain[np.abs(freqs - center) <= width] = 1.0
    formant = (freqs >= 280.0) & (freqs < 3800.0)
    close = np.zeros(len(gain), dtype=bool)
    for harmonic in range(1, 14):
        center = f0 * harmonic
        close |= np.abs(freqs - center) <= max(10.0, min(16.0, 0.028 * center))
    gain[formant & close] = np.maximum(gain[formant & close], 0.45)
    return gain


def _denoise_score(
    audio: np.ndarray,
    sample_rate: int,
    is_score: np.ndarray,
    ultron_f0: np.ndarray,
) -> np.ndarray:
    """Pull the film bed out of Ultron's words. Do not paste the mix."""
    n_fft = 2048
    hop = 512
    noise = _median_noise_mag(audio, sample_rate, is_score, n_fft, hop)
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    pad = n_fft
    padded = np.pad(audio.astype(np.float64), pad, mode="constant")
    acc = np.zeros(len(padded), dtype=np.float64)
    weight = np.zeros(len(padded), dtype=np.float64)
    score_frames = int(np.sum(is_score))
    for start in range(0, len(padded) - n_fft + 1, hop):
        spec = np.fft.rfft(padded[start : start + n_fft] * window)
        mag = np.abs(spec)
        if noise is not None and score_frames >= 3:
            mag = np.maximum(mag - 1.35 * noise, mag * 0.16)
            chest = (freqs >= 62.0) & (freqs <= 105.0)
            if np.any(chest) and float(np.max(mag[chest])) > 1e-6:
                chest_f0 = _peak_hz(freqs, mag, 62.0, 105.0)
                if chest_f0 >= 62.0:
                    mag = mag * _harmonic_gain(freqs, chest_f0)
        rec = np.fft.irfft(mag * np.exp(1j * np.angle(spec)), n=n_fft)
        acc[start : start + n_fft] += rec * window
        weight[start : start + n_fft] += window**2
    rec = (acc / np.maximum(weight, 1e-9))[pad : pad + len(audio)]
    return rec.astype(np.float32)


def _dominant_hz(chunk: np.ndarray, sample_rate: int) -> float:
    if chunk.size < 16:
        return 0.0
    window = np.hanning(len(chunk))
    spec = np.abs(np.fft.rfft(chunk.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
    band = (freqs >= 55.0) & (freqs <= 340.0)
    if not np.any(band):
        return 0.0
    return float(freqs[band][np.argmax(spec[band])])


def isolate_ultron_voice(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Keep Ultron's words and his pauses. Replace score with silence of the same length."""
    audio = audio.astype(np.float32)
    if audio.size < sample_rate // 16:
        return audio
    hop, n_frames = _frame_count(len(audio), sample_rate)
    times, f0 = track_f0(audio, sample_rate)
    _ht, high = track_f0_range(audio, sample_rate, _OTHER_F0_MIN, _OTHER_F0_MAX)
    ultron = _f0_on_hops(times, f0, n_frames)
    other = _f0_on_hops(_ht, high, n_frames)
    rms = _frame_rms(audio, sample_rate)
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    elif len(rms) > n_frames:
        rms = rms[:n_frames]

    is_ultron = (ultron >= 58.0) & (ultron <= _ULTRON_F0_MAX)
    is_other = other >= _OTHER_F0_MIN
    conflict = np.flatnonzero(is_ultron & is_other)
    for index in conflict:
        start = index * hop
        dom = _dominant_hz(audio[start : start + hop], sample_rate)
        if abs(dom - float(other[index])) < abs(dom - float(ultron[index])):
            is_ultron[index] = False
        else:
            is_other[index] = False
    voiced = is_ultron
    ultron_idx = np.flatnonzero(voiced)
    if ultron_idx.size == 0:
        return audio

    first, last = int(ultron_idx[0]), int(ultron_idx[-1])
    is_speech = _fill_short_gaps(voiced, max(1, int(_FILL_UNVOICED_SEC / FRAME_SEC)))
    is_speech[is_other] = False
    pause_thresh = max(0.012, float(np.median(rms[voiced])) * 0.18 if np.any(voiced) else 0.012)
    is_score = (~is_speech) & ~is_other & (rms >= pause_thresh)
    edge_pad = max(1, int(0.08 / FRAME_SEC))
    steps = 0
    while first > 0 and not is_other[first - 1] and rms[first - 1] >= pause_thresh and steps < edge_pad:
        first -= 1
        is_speech[first] = True
        steps += 1
    steps = 0
    while last + 1 < n_frames and not is_other[last + 1] and rms[last + 1] >= pause_thresh and steps < edge_pad:
        last += 1
        is_speech[last] = True
        steps += 1
    is_score = (~is_speech) & ~is_other & (rms >= pause_thresh)

    denoised = (
        _denoise_score(audio, sample_rate, is_score, ultron)
        if int(np.sum(is_score)) >= 3
        else audio
    )

    keep = np.zeros(n_frames, dtype=bool)
    # Keep the whole Ultron span so his pauses stay the same length. Other talkers
    # before/after are dropped; overlap inside the line becomes silence, not a cut.
    keep[first : last + 1] = True
    keep = _cap_long_pauses(keep, is_speech, first, last)

    pieces: list[np.ndarray] = []
    for index in range(n_frames):
        if not keep[index]:
            continue
        start_sample = 0 if index == 0 else index * hop
        end_sample = len(audio) if index == n_frames - 1 else min(len(audio), (index + 1) * hop)
        if end_sample <= start_sample:
            continue
        if is_speech[index]:
            pieces.append(denoised[start_sample:end_sample].copy())
        else:
            pieces.append(np.zeros(end_sample - start_sample, dtype=np.float32))
    if not pieces:
        return audio
    return np.concatenate(pieces).astype(np.float32)


def _talker_frame_masks(
    audio: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    hop, n_frames = _frame_count(len(audio), sample_rate)
    times, f0 = track_f0(audio, sample_rate)
    _ht, high = track_f0_range(audio, sample_rate, _OTHER_F0_MIN, _OTHER_F0_MAX)
    ultron = _f0_on_hops(times, f0, n_frames)
    other = _f0_on_hops(_ht, high, n_frames)
    rms = _frame_rms(audio, sample_rate)
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    elif len(rms) > n_frames:
        rms = rms[:n_frames]

    is_ultron = (ultron >= 58.0) & (ultron <= _ULTRON_F0_MAX)
    is_other = other >= _OTHER_F0_MIN
    for index in np.flatnonzero(is_ultron & is_other):
        start = index * hop
        dom = _dominant_hz(audio[start : start + hop], sample_rate)
        if abs(dom - float(other[index])) < abs(dom - float(ultron[index])):
            is_ultron[index] = False
        else:
            is_other[index] = False

    is_speech = _fill_short_gaps(is_ultron, max(1, int(_FILL_UNVOICED_SEC / FRAME_SEC)))
    is_speech[is_other] = False
    pause_thresh = max(0.012, float(np.median(rms[is_ultron])) * 0.18 if np.any(is_ultron) else 0.012)
    is_score = (~is_speech) & ~is_other & (rms >= pause_thresh)
    return is_speech, is_other, is_score, ultron, hop


def _gate_frames(
    audio: np.ndarray,
    sample_rate: int,
    mask: np.ndarray,
    hop: int,
    *,
    target_gain: float = 0.0,
) -> np.ndarray:
    if not np.any(mask):
        return audio.astype(np.float32)
    out = audio.astype(np.float64).copy()
    fade = max(1, int(sample_rate * 0.007))
    n_frames = int(np.ceil(len(audio) / hop))
    for index in range(min(len(mask), n_frames)):
        if not mask[index]:
            continue
        start = index * hop
        end = min(len(out), (index + 1) * hop)
        length = end - start
        if length <= 0:
            continue
        gain = np.full(length, target_gain, dtype=np.float64)
        if length > fade * 2:
            ramp = np.linspace(1.0, target_gain, fade, dtype=np.float64)
            gain[:fade] = ramp
            gain[-fade:] = ramp[::-1]
        out[start:end] *= gain
    return out.astype(np.float32)


def _highpass_gentle(audio: np.ndarray, sample_rate: int, cutoff: float = 58.0) -> np.ndarray:
    if audio.size < 32 or cutoff <= 0.0:
        return audio
    spec = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    gain = np.ones(len(freqs), dtype=np.float64)
    gain[freqs < cutoff] = 0.0
    soft = (freqs >= cutoff) & (freqs < cutoff * 1.8)
    if np.any(soft):
        fade = (freqs[soft] - cutoff) / (0.8 * cutoff)
        gain[soft] = np.clip(fade, 0.0, 1.0)
    out = np.fft.irfft(spec * gain, n=len(audio))
    return out.astype(np.float32)


def polish_gold_clip(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mute score/other voices, trim walk-in rumble, keep endings and speech level."""
    audio = audio.astype(np.float32)
    if audio.size < sample_rate // 16:
        return audio

    is_speech, is_other, is_score, ultron, hop = _talker_frame_masks(audio, sample_rate)
    out = audio.copy()
    n_frames = len(is_speech)

    if int(np.sum(is_other)) >= 1:
        out = _gate_frames(out, sample_rate, is_other, hop, target_gain=0.0)

    for index in range(n_frames):
        if is_score[index] and not is_speech[index]:
            start = index * hop
            end = min(len(out), (index + 1) * hop)
            out[start:end] = 0.0

    speech_frames = np.flatnonzero(is_speech)
    if speech_frames.size == 0:
        islands = _speech_islands(out, sample_rate, min_speech_ms=40)
        if not islands:
            return _normalize_speech_level(_highpass_gentle(out, sample_rate), sample_rate)
        start_sample, end_sample = islands[0][0], islands[-1][1]
    else:
        first, last = int(speech_frames[0]), int(speech_frames[-1])
        while first < last:
            chunk = out[first * hop : (first + 1) * hop]
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if chunk.size else 0.0
            if ultron[first] >= 58.0 or chunk_rms < 0.018:
                break
            dom = _dominant_hz(chunk, sample_rate)
            if dom >= 58.0:
                break
            first += 1
        start_sample = max(0, first * hop - int(0.010 * sample_rate))
        end_sample = min(len(out), (last + 1) * hop + int(0.14 * sample_rate))

    trimmed = out[start_sample:end_sample]
    if trimmed.size < sample_rate // 16:
        trimmed = out
    trimmed = _highpass_gentle(trimmed, sample_rate)
    return _normalize_speech_level(trimmed, sample_rate)


def _normalize_speech_level(audio: np.ndarray, sample_rate: int, target_rms: float = 0.12) -> np.ndarray:
    if audio.size < sample_rate // 32:
        return audio.astype(np.float32)
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    n_frames = max(1, int(np.ceil(len(audio) / hop)))
    speech_rms: list[float] = []
    for index in range(n_frames):
        chunk = audio[index * hop : (index + 1) * hop]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms >= 0.012:
            speech_rms.append(rms)
    if not speech_rms:
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-5:
            return audio.astype(np.float32)
        return (audio * min(0.98, 0.92 / peak)).astype(np.float32)
    current = float(np.median(speech_rms))
    gain = np.float32(np.clip(target_rms / max(current, 1e-5), 0.5, 4.0))
    scaled = (audio * gain).astype(np.float32)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.98:
        scaled = (scaled * (0.98 / peak)).astype(np.float32)
    return scaled


def _speech_islands(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_speech_ms: int = 60,
    min_pause_ms: int = 50,
) -> list[tuple[int, int]]:
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    n = max(1, int(np.ceil(len(audio) / hop)))
    rms = np.array(
        [
            float(np.sqrt(np.mean(audio[i * hop : (i + 1) * hop].astype(np.float64) ** 2)))
            if audio[i * hop : (i + 1) * hop].size
            else 0.0
            for i in range(n)
        ],
        dtype=np.float64,
    )
    thresh = max(0.012, float(np.percentile(rms, 55)) * 0.22)
    speech = rms >= thresh
    min_pause = max(1, int(min_pause_ms / (FRAME_SEC * 1000)))
    min_speech = max(1, int(min_speech_ms / (FRAME_SEC * 1000)))
    islands: list[tuple[int, int]] = []
    index = 0
    while index < n:
        if not speech[index]:
            index += 1
            continue
        start = index
        while index < n:
            if speech[index]:
                index += 1
                continue
            look = index
            while look < n and not speech[look]:
                look += 1
            if look - index < min_pause and look < n:
                index = look
                continue
            break
        end = index
        if end - start >= min_speech:
            islands.append((start * hop, min(len(audio), end * hop)))
        index = max(index, end)
    return islands


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    audio = audio.astype(np.float32)
    if len(audio) == n:
        return audio
    if abs(len(audio) - n) <= max(8, int(0.02 * n)):
        return _time_map(audio, n)
    if len(audio) < n:
        return np.concatenate([audio, np.zeros(n - len(audio), dtype=np.float32)])
    return audio[:n]


def _align_islands(
    clone: np.ndarray,
    sample_rate: int,
    clone_islands: list[tuple[int, int]],
    target_islands: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not target_islands:
        return clone_islands
    if len(clone_islands) == len(target_islands):
        return clone_islands
    n = len(target_islands)
    speak0 = clone_islands[0][0] if clone_islands else 0
    speak1 = clone_islands[-1][1] if clone_islands else len(clone)
    region = clone[speak0:speak1]
    weights = [max(0.12, (t1 - t0) / sample_rate) for t0, t1 in target_islands]
    ranges = energy_split_ranges(region, sample_rate, ["x"] * n, piece_weights=weights)
    return [(speak0 + a, speak0 + b) for a, b in ranges]


def _stamp_islands(
    clone: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Stretch each spoken island to the film word, then insert the film's pauses as silence."""
    clone_islands = _speech_islands(clone, sample_rate)
    target_islands = _speech_islands(target, sample_rate)
    if not target_islands:
        factor = len(target) / max(1, len(clone))
        stretched = lengthen_keep_pitch(clone, sample_rate, factor, wide=True)
        return _fit_length(stretched, len(target))
    if not clone_islands:
        clone_islands = [(0, len(clone))]

    clone_islands = _align_islands(clone, sample_rate, clone_islands, target_islands)
    pieces: list[np.ndarray] = []
    if target_islands[0][0] > 0:
        pieces.append(np.zeros(target_islands[0][0], dtype=np.float32))
    for index, ((c0, c1), (t0, t1)) in enumerate(zip(clone_islands, target_islands)):
        chunk = clone[c0:c1]
        need = max(8, t1 - t0)
        if chunk.size < 8:
            pieces.append(np.zeros(need, dtype=np.float32))
        else:
            # Never linear-resample speech shorter — that raises pitch (the mouse squeak).
            factor = float(np.clip(need / max(1, len(chunk)), 0.90, 1.28))
            stretched = lengthen_keep_pitch(chunk, sample_rate, factor, wide=True)
            if len(stretched) < need:
                stretched = np.concatenate(
                    [stretched, np.zeros(need - len(stretched), dtype=np.float32)]
                )
            elif len(stretched) > need:
                stretched = stretched[:need]
            pieces.append(stretched.astype(np.float32))
        if index + 1 < len(target_islands):
            gap = target_islands[index + 1][0] - t1
            if gap > 0:
                pieces.append(np.zeros(gap, dtype=np.float32))
    trail = len(target) - target_islands[-1][1]
    if trail > 0:
        pieces.append(np.zeros(trail, dtype=np.float32))
    if not pieces:
        return clone
    return _fit_length(np.concatenate(pieces).astype(np.float32), len(target))


def _film_target(film: np.ndarray, sample_rate: int) -> np.ndarray:
    """Use isolated film only when it keeps the line; gold clips are already clean."""
    film = film.astype(np.float32)
    if film.size < sample_rate // 16:
        return film
    isolated = isolate_ultron_voice(film, sample_rate)
    if isolated.size >= max(sample_rate // 4, int(len(film) * 0.72)):
        return isolated
    return film


def apply_film_identity(
    clone: np.ndarray,
    film: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Match Ultron's pitch, pace, and tone. Never paste the film mix."""
    if clone.size < sample_rate // 16 or film.size < sample_rate // 16:
        return clone.astype(np.float32)

    clone = clone.astype(np.float32)
    target = _film_target(film, sample_rate)

    clone = _stamp_islands(clone, target, sample_rate)

    target_ltas = measure_ltas_db(target, sample_rate)
    clone = apply_ltas_match(clone, sample_rate, target_ltas, loose=True)
    if target_ltas and abs(target_ltas[0]) > 1.0:
        now = measure_ltas_db(clone, sample_rate)
        if now and abs(now[0]) > 0.5:
            clone = _low_shelf(
                clone, sample_rate, float(np.clip(target_ltas[0] - now[0], -6.0, 6.0))
            )

    clone = _impose_rms(clone, target, sample_rate)
    target_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    clone_rms = float(np.sqrt(np.mean(clone.astype(np.float64) ** 2)))
    if clone_rms > 1e-8 and target_rms > 1e-8:
        clone = clone * np.float32(target_rms / clone_rms)
    return np.clip(_fit_length(clone, len(target)), -1.0, 1.0).astype(np.float32)


def _speech_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """True on frames that look like a voice, not the film score."""
    hop = max(1, int(round(sample_rate * FRAME_SEC)))
    frame = max(hop, int(sample_rate * 0.04))
    n = max(1, (len(audio) - frame) // hop + 1)
    mask = np.zeros(n, dtype=bool)
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    voice_band = (freqs >= 80.0) & (freqs < 400.0)
    air_band = (freqs >= 400.0) & (freqs < 4000.0)
    for index in range(n):
        chunk = audio[index * hop : index * hop + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < 0.02:
            continue
        power = np.abs(np.fft.rfft(chunk * window)) ** 2
        voice = float(np.mean(power[voice_band]))
        air = float(np.mean(power[air_band]))
        mask[index] = voice >= air * 0.15
    return mask


def _voiced_frac(f0: np.ndarray) -> float:
    if f0.size == 0:
        return 0.0
    return float(np.mean(f0 > 1.0))


def _low_shelf(audio: np.ndarray, sample_rate: int, gain_db: float, cutoff: float = 78.0) -> np.ndarray:
    if abs(gain_db) < 0.35 or audio.size < 32:
        return audio
    spec = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    gain = np.ones(len(freqs), dtype=np.float64)
    lo = freqs <= cutoff
    gain[lo] = 10.0 ** (gain_db / 20.0)
    mid = (freqs > cutoff) & (freqs < cutoff * 3.0)
    if np.any(mid):
        fade = 1.0 - (freqs[mid] - cutoff) / (2.0 * cutoff)
        gain[mid] = 10.0 ** ((gain_db * np.clip(fade, 0.0, 1.0)) / 20.0)
    out = np.fft.irfft(spec * gain, n=len(audio))
    return out.astype(np.float32)


def compare_frames(ours: np.ndarray, film: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compare two takes on the shared 10 ms frame clock."""
    n = min(len(ours), len(film))
    ours_f = ours[:n].astype(np.float32)
    film_f = film[:n].astype(np.float32)
    _times_o, f0_o = track_f0(ours_f, sample_rate)
    _times_f, f0_f = track_f0(film_f, sample_rate)
    n_f0 = min(len(f0_o), len(f0_f))
    speech = _speech_mask(film_f, sample_rate)
    n_sp = min(n_f0, len(speech))
    speech_n = speech[:n_sp]
    paired = speech_n & (f0_o[:n_sp] > 1.0) & (f0_f[:n_sp] > 1.0)
    if int(np.sum(paired)) >= 4:
        f0_mae = float(np.mean(np.abs(f0_o[:n_sp][paired] - f0_f[:n_sp][paired])))
    else:
        f0_mae = 0.0
    if int(np.sum(speech_n)) >= 8:
        voiced_o = float(np.mean(f0_o[:n_sp][speech_n] > 1.0))
        voiced_f = float(np.mean(f0_f[:n_sp][speech_n] > 1.0))
    else:
        voiced_o = float(np.mean(f0_o[:n_f0] > 1.0)) if n_f0 else 0.0
        voiced_f = float(np.mean(f0_f[:n_f0] > 1.0)) if n_f0 else 0.0

    ltas_o = np.array(measure_ltas_db(ours_f, sample_rate), dtype=np.float64)
    ltas_f = np.array(measure_ltas_db(film_f, sample_rate), dtype=np.float64)
    peak_f = float(np.max(ltas_f)) if ltas_f.size else 0.0
    valid = ltas_f > (peak_f - 36.0)
    if int(np.sum(valid)) >= 1:
        ltas_mae = float(np.mean(np.abs(ltas_o[valid] - ltas_f[valid])))
    else:
        ltas_mae = 0.0
    if ltas_o.size > 0 and valid.size > 0 and bool(valid[0]):
        bass_err = abs(float(ltas_o[0] - ltas_f[0]))
    else:
        bass_err = 0.0

    env_o = _frame_rms(ours_f, sample_rate)
    env_f = _frame_rms(film_f, sample_rate)
    n_env = min(len(env_o), len(env_f))
    if n_env >= 8 and float(np.std(env_o[:n_env])) > 1e-6 and float(np.std(env_f[:n_env])) > 1e-6:
        rms_corr = float(np.corrcoef(env_o[:n_env], env_f[:n_env])[0, 1])
        if not np.isfinite(rms_corr):
            rms_corr = 0.0
    else:
        rms_corr = 0.0

    return {
        "duration_err_sec": abs(len(ours) - len(film)) / float(sample_rate),
        "f0_mae_hz": f0_mae,
        "ltas_mae_db": ltas_mae,
        "rms_corr": rms_corr,
        "bass_err_db": bass_err,
        "voiced_frac_err": abs(voiced_o - voiced_f),
        "n_frames": n_env,
    }


def report_passes(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metrics["duration_err_sec"] > MAX_DURATION_ERR_SEC:
        failures.append(
            f"duration {metrics['duration_err_sec']*1000:.0f}ms > {MAX_DURATION_ERR_SEC*1000:.0f}ms"
        )
    if metrics["f0_mae_hz"] > MAX_F0_MAE_HZ:
        failures.append(f"f0 mae {metrics['f0_mae_hz']:.1f}Hz > {MAX_F0_MAE_HZ:.1f}Hz")
    if metrics["ltas_mae_db"] > MAX_LTAS_MAE_DB:
        failures.append(f"eq mae {metrics['ltas_mae_db']:.1f}dB > {MAX_LTAS_MAE_DB:.1f}dB")
    if metrics["rms_corr"] < MIN_RMS_CORR:
        failures.append(f"energy corr {metrics['rms_corr']:.2f} < {MIN_RMS_CORR:.2f}")
    if metrics["bass_err_db"] > MAX_BASS_ERR_DB:
        failures.append(f"bass {metrics['bass_err_db']:.1f}dB > {MAX_BASS_ERR_DB:.1f}dB")
    if metrics.get("voiced_frac_err", 0.0) > MAX_VOICED_FRAC_ERR + 1e-6:
        failures.append(
            f"voiced frac err {metrics['voiced_frac_err']:.2f} > {MAX_VOICED_FRAC_ERR:.2f}"
        )
    return not failures, failures


def frame_report(clip: GoldClip, ours: np.ndarray, sample_rate: int) -> FrameReport:
    film = isolate_ultron_voice(
        _resample_rate(clip.audio, clip.sample_rate, sample_rate), sample_rate
    )
    metrics = compare_frames(ours, film, sample_rate)
    passed, failures = report_passes(metrics)
    return FrameReport(
        clip_id=clip.clip_id,
        text=clip.text,
        duration_err_sec=metrics["duration_err_sec"],
        f0_mae_hz=metrics["f0_mae_hz"],
        ltas_mae_db=metrics["ltas_mae_db"],
        rms_corr=metrics["rms_corr"],
        bass_err_db=metrics["bass_err_db"],
        voiced_frac_err=metrics["voiced_frac_err"],
        n_frames=int(metrics["n_frames"]),
        passed=passed,
        failures=failures,
    )


def load_gold_manifest() -> list[dict]:
    if not _MANIFEST.exists():
        return []
    return list(json.loads(_MANIFEST.read_text()).get("clips", []))


_clips_cache: list[GoldClip] | None = None


def load_gold_clips(*, force: bool = False) -> list[GoldClip]:
    global _clips_cache
    if _clips_cache is not None and not force:
        return _clips_cache
    clips: list[GoldClip] = []
    for item in load_gold_manifest():
        path = _GOLD_DIR / item["file"]
        if not path.exists():
            continue
        audio, sample_rate = wav_bytes_to_mono_float(path.read_bytes())
        clips.append(
            GoldClip(
                clip_id=str(item["id"]),
                text=str(item["text"]),
                path=path,
                duration_sec=float(item.get("duration_sec", len(audio) / sample_rate)),
                audio=audio,
                sample_rate=sample_rate,
            )
        )
    _clips_cache = clips
    return clips


def covering_gold(text: str, clips: list[GoldClip] | None = None) -> list[GoldClip]:
    """Gold clips whose words appear, in order, inside `text`."""
    clips = clips if clips is not None else load_gold_clips()
    query = _words(text)
    if not query or not clips:
        return []
    remaining = list(clips)
    ordered: list[GoldClip] = []
    pos = 0
    while pos < len(query) and remaining:
        best: GoldClip | None = None
        best_len = 0
        for clip in remaining:
            gold_words = _words(clip.text)
            n = len(gold_words)
            if n < 1 or pos + n > len(query):
                continue
            if query[pos : pos + n] == gold_words and n > best_len:
                best = clip
                best_len = n
        if best is None:
            pos += 1
            continue
        ordered.append(best)
        remaining.remove(best)
        pos += best_len
    return ordered


def gold_cover_ratio(text: str, clips: list[GoldClip] | None = None) -> float:
    query = _words(text)
    if not query:
        return 0.0
    covered = sum(len(_words(clip.text)) for clip in covering_gold(text, clips))
    return min(1.0, covered / len(query))


def _clip_match_ratio(words: list[str], clip: GoldClip) -> float:
    gold = _words(clip.text)
    if not words or not gold:
        return 0.0
    if words == gold:
        return 1.0
    return SequenceMatcher(None, words, gold).ratio()


def best_gold_for_sentence(
    sentence: str,
    clips: list[GoldClip],
    *,
    min_ratio: float = _SENTENCE_MATCH,
) -> GoldClip | None:
    words = _words(sentence)
    best: GoldClip | None = None
    best_ratio = 0.0
    for clip in clips:
        ratio = _clip_match_ratio(words, clip)
        if ratio > best_ratio:
            best = clip
            best_ratio = ratio
    if best is None or best_ratio < min_ratio:
        return None
    return best


def gold_sentence_cover_ratio(text: str, clips: list[GoldClip] | None = None) -> float:
    clips = clips if clips is not None else load_gold_clips()
    sentences = _split_sentences(text) or [text]
    if not sentences or not clips:
        return 0.0
    hits = sum(1 for sentence in sentences if best_gold_for_sentence(sentence, clips) is not None)
    return hits / len(sentences)


def _stamp_by_sentence(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    clips: list[GoldClip],
) -> np.ndarray:
    sentences = _split_sentences(text) or [text]
    if len(sentences) == 1:
        gold = best_gold_for_sentence(text, clips)
        if gold is None:
            return audio
        film = _resample_rate(gold.audio, gold.sample_rate, sample_rate)
        return apply_film_identity(audio, film, sample_rate)

    spans = sentence_sample_ranges(audio, sample_rate, sentences)
    pieces: list[np.ndarray] = []
    for sentence, (start, end) in zip(sentences, spans):
        chunk = audio[start:end]
        if chunk.size < sample_rate // 20:
            if chunk.size:
                pieces.append(chunk.astype(np.float32))
            continue
        gold = best_gold_for_sentence(sentence, clips)
        if gold is None:
            pieces.append(chunk.astype(np.float32))
            continue
        film = _resample_rate(gold.audio, gold.sample_rate, sample_rate)
        pieces.append(apply_film_identity(chunk, film, sample_rate))
    if not pieces:
        return audio
    return np.concatenate(pieces).astype(np.float32)


def apply_gold_performance(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    clips: list[GoldClip] | None = None,
) -> np.ndarray:
    """Shape the clone take toward film references. Never paste movie audio."""
    clips = clips if clips is not None else load_gold_clips()
    if not text.strip() or not clips or audio.size < sample_rate // 16:
        return audio

    ordered = covering_gold(text, clips)
    if not ordered:
        return _stamp_by_sentence(audio, sample_rate, text, clips)

    if len(ordered) == 1:
        film = _resample_rate(ordered[0].audio, ordered[0].sample_rate, sample_rate)
        return apply_film_identity(audio, film, sample_rate)

    film_lens = [
        len(_resample_rate(clip.audio, clip.sample_rate, sample_rate)) for clip in ordered
    ]
    spans = sentence_sample_ranges(audio, sample_rate, [c.text for c in ordered])
    if len(spans) != len(ordered):
        spans = film_weighted_split_ranges(audio, sample_rate, film_lens)
    if len(spans) != len(ordered):
        return _stamp_by_sentence(audio, sample_rate, text, clips)

    pieces: list[np.ndarray] = []
    for clip, (start, end) in zip(ordered, spans):
        chunk = audio[start:end]
        film = _resample_rate(clip.audio, clip.sample_rate, sample_rate)
        if chunk.size < sample_rate // 20:
            pieces.append(chunk.astype(np.float32))
            continue
        pieces.append(apply_film_identity(chunk, film, sample_rate))
    if not pieces:
        return audio
    return np.concatenate(pieces).astype(np.float32)


def extract_gold_from_source(source_path: Path | None = None) -> list[dict]:
    """Cut every known Ultron line out of source.wav and write gold/<slug>.wav."""
    source_path = source_path or _SOURCE
    if not source_path.exists():
        return []
    audio, sample_rate = wav_bytes_to_mono_float(source_path.read_bytes())
    _GOLD_DIR.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    used_ids: dict[str, int] = {}
    for start_sec, end_sec, text in FILM_WINDOWS:
        window = audio[int(start_sec * sample_rate) : int(end_sec * sample_rate)]
        sentences = _split_sentences(text) or [text]
        ranges = energy_split_ranges(window, sample_rate, sentences)
        for sentence, (start, end) in zip(sentences, ranges):
            clip = polish_gold_clip(window[start:end], sample_rate)
            if clip.size < sample_rate // 8:
                clip = window[start:end]
            if clip.size < sample_rate // 8:
                continue
            slug = _slug(sentence)
            used_ids[slug] = used_ids.get(slug, 0) + 1
            clip_id = slug if used_ids[slug] == 1 else f"{slug}-{used_ids[slug]}"
            filename = f"{clip_id}.wav"
            path = _GOLD_DIR / filename
            path.write_bytes(mono_float_to_wav_bytes(clip.astype(np.float32), sample_rate))
            abs_start = start_sec + start / sample_rate
            abs_end = start_sec + end / sample_rate
            clips.append(
                {
                    "id": clip_id,
                    "text": sentence,
                    "file": filename,
                    "duration_sec": round(len(clip) / sample_rate, 4),
                    "start_sec": round(abs_start, 3),
                    "end_sec": round(abs_end, 3),
                }
            )
    _MANIFEST.write_text(json.dumps({"clips": clips}, indent=2))
    load_gold_clips(force=True)
    return clips


def ensure_gold_clips() -> list[GoldClip]:
    clips = load_gold_clips()
    if clips:
        return clips
    if _SOURCE.exists():
        extract_gold_from_source()
        return load_gold_clips(force=True)
    return []


def profile_gold_dir(data_dir: Path, profile_id: str) -> Path:
    return Path(data_dir) / "gold" / profile_id


def load_profile_gold(data_dir: Path | str, profile_id: str) -> list[GoldClip]:
    root = profile_gold_dir(Path(data_dir), profile_id)
    manifest = root / "manifest.json"
    if not manifest.exists():
        return []
    clips: list[GoldClip] = []
    try:
        items = list(json.loads(manifest.read_text()).get("clips", []))
    except (json.JSONDecodeError, OSError):
        return []
    for item in items:
        path = root / str(item.get("file", ""))
        if not path.exists():
            continue
        audio, sample_rate = wav_bytes_to_mono_float(path.read_bytes())
        clips.append(
            GoldClip(
                clip_id=str(item.get("id", path.stem)),
                text=str(item.get("text", "")),
                path=path,
                duration_sec=float(item.get("duration_sec", len(audio) / max(1, sample_rate))),
                audio=audio,
                sample_rate=sample_rate,
            )
        )
    return clips


def save_profile_gold_clips(
    data_dir: Path | str,
    profile_id: str,
    samples: list[tuple[Path, str]],
) -> list[GoldClip]:
    """Persist isolated YouTube lines so generation can copy their pace, tone, and pauses."""
    root = profile_gold_dir(Path(data_dir), profile_id)
    root.mkdir(parents=True, exist_ok=True)
    meta: list[dict] = []
    clips: list[GoldClip] = []
    used: dict[str, int] = {}
    for path, text in samples:
        if not path.exists() or not str(text).strip():
            continue
        audio, sample_rate = wav_bytes_to_mono_float(path.read_bytes())
        isolated = isolate_ultron_voice(audio, sample_rate)
        if isolated.size < sample_rate // 8:
            isolated = audio
        slug = _slug(text)
        used[slug] = used.get(slug, 0) + 1
        clip_id = slug if used[slug] == 1 else f"{slug}-{used[slug]}"
        filename = f"{clip_id}.wav"
        dest = root / filename
        dest.write_bytes(mono_float_to_wav_bytes(isolated.astype(np.float32), sample_rate))
        # Keep the clone samples full-fidelity for TTS — gold clips are timing references only.
        item = {
            "id": clip_id,
            "text": str(text).strip(),
            "file": filename,
            "duration_sec": round(len(isolated) / max(1, sample_rate), 4),
        }
        meta.append(item)
        clips.append(
            GoldClip(
                clip_id=clip_id,
                text=str(text).strip(),
                path=dest,
                duration_sec=float(item["duration_sec"]),
                audio=isolated.astype(np.float32),
                sample_rate=sample_rate,
            )
        )
    (root / "manifest.json").write_text(json.dumps({"clips": meta}, indent=2))
    atlas_phrases: list = []
    for clip in clips:
        built = phrases_from_paragraph(clip.text, clip.audio, clip.sample_rate)
        atlas_phrases.extend(built)
    if atlas_phrases:
        save_atlas(atlas_phrases, profile_atlas_path(Path(data_dir), profile_id))
    return clips


def delivery_gold_clips(
    *,
    data_dir: Path | str | None = None,
    profile_id: str | None = None,
) -> list[GoldClip]:
    clips = list(ensure_gold_clips())
    if data_dir and profile_id:
        clips.extend(load_profile_gold(data_dir, profile_id))
    return clips


def typical_pause_ms(clips: list[GoldClip] | None = None) -> int:
    """Median in-scene pause from isolated gold — used when the line is not a known clip."""
    clips = clips if clips is not None else load_gold_clips()
    pauses: list[float] = []
    for clip in clips:
        isolated = isolate_ultron_voice(clip.audio, clip.sample_rate)
        for start, end in detect_quiet_spans(isolated, clip.sample_rate, min_ms=80):
            ms = (end - start) / max(1, clip.sample_rate) * 1000.0
            if 90.0 <= ms <= 1100.0:
                pauses.append(ms)
    if not pauses:
        return 0
    return int(round(float(np.median(np.array(pauses, dtype=np.float64)))))
