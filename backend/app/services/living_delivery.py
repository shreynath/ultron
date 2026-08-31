from __future__ import annotations

import numpy as np

from app.services.dramatic_planner import PhrasePlan, _split_sentences, plan_phrases
from app.services.sinister_texture import apply_energy_contour

_FRAME_MS = 10
_QUIET_DB = -38.0
_MIN_GAP_MS = 40
_SEARCH_MS = 520
_SKIP_EDGE_MS = 70
_MAX_PAUSE_MS = 850
_MIN_STRETCH_MS = 40


def _db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))


def _smooth(values: np.ndarray, sample_rate: int, window_ms: int = 90) -> np.ndarray:
    win = max(3, int(sample_rate * window_ms / 1000))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    padded = np.pad(values, win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)].astype(np.float32)


def _frame_rms(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    frame_len = max(1, int(sample_rate * _FRAME_MS / 1000))
    n_frames = max(1, len(audio) // frame_len)
    rms = np.array(
        [
            float(np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2)))
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )
    return rms, frame_len


def detect_quiet_spans(
    audio: np.ndarray,
    sample_rate: int,
    *,
    thresh_db: float = _QUIET_DB,
    min_ms: int = _MIN_GAP_MS,
) -> list[tuple[int, int]]:
    if audio.size < sample_rate // 8:
        return []

    rms, frame_len = _frame_rms(audio, sample_rate)
    thresh = _db_to_lin(thresh_db)
    min_frames = max(1, int(min_ms / _FRAME_MS))
    skip_frames = int(_SKIP_EDGE_MS / _FRAME_MS)

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, energy in enumerate(rms):
        quiet = energy < thresh
        if quiet and start is None:
            start = index
        elif not quiet and start is not None:
            if index - start >= min_frames and start >= skip_frames:
                spans.append((start * frame_len, index * frame_len))
            start = None
    if start is not None and len(rms) - start >= min_frames and start >= skip_frames:
        end = min(len(audio), len(rms) * frame_len)
        if end < len(audio) - int(sample_rate * _SKIP_EDGE_MS / 1000):
            spans.append((start * frame_len, end))
    return spans


def _target_pause_ms(phrase: str, *, period_ms: int = 320) -> int:
    raw = phrase.rstrip()
    period = max(220, min(700, period_ms if period_ms >= 180 else 320))
    if raw.endswith("...") or raw.endswith("…"):
        return min(720, period + 160)
    if raw.endswith(","):
        return max(80, int(period * 0.35))
    if raw.endswith(";"):
        return max(100, int(period * 0.50))
    if raw.endswith("!"):
        return max(200, int(period * 0.85))
    if raw.endswith("?"):
        return max(220, int(period * 0.90))
    if raw.endswith("."):
        return period
    return max(120, int(period * 0.55))


def _punctuation_events(text: str, *, period_ms: int = 320) -> list[tuple[float, int]]:
    phrases = _split_sentences(text)
    if not phrases:
        return []
    total_words = sum(max(1, len(phrase.split())) for phrase in phrases) or 1
    cursor = 0
    events: list[tuple[float, int]] = []
    for index, phrase in enumerate(phrases):
        cursor += max(1, len(phrase.split()))
        if index == len(phrases) - 1:
            continue
        events.append(
            (cursor / total_words, min(_MAX_PAUSE_MS, _target_pause_ms(phrase, period_ms=period_ms)))
        )
    return events


def _nearest_span(
    spans: list[tuple[int, int]],
    target: int,
    window: int,
    used: set[int],
) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_index = -1
    best_dist = window + 1
    for index, (start, end) in enumerate(spans):
        if index in used:
            continue
        mid = (start + end) // 2
        dist = abs(mid - target)
        if dist < best_dist:
            best = (start, end)
            best_dist = dist
            best_index = index
    if best is None or best_index < 0:
        return None
    used.add(best_index)
    return best


def _insert_silence(audio: np.ndarray, at: int, extra: int, sample_rate: int) -> np.ndarray:
    """Lengthen an existing gap. Do not fade the words into digital zero."""
    del sample_rate
    at = min(max(0, at), len(audio))
    extra = max(0, extra)
    if extra == 0:
        return audio
    return np.concatenate(
        [audio[:at], np.zeros(extra, dtype=np.float32), audio[at:]]
    )


def _valley_insert_at(
    audio: np.ndarray,
    sample_rate: int,
    target_sample: int,
    search: int,
    used: set[int],
) -> int | None:
    """If TTS skipped a period, stop in a real energy dip — never in a vowel."""
    rms, frame_len = _frame_rms(audio, sample_rate)
    skip = int(sample_rate * _SKIP_EDGE_MS / 1000)
    lo = max(skip, target_sample - search)
    hi = min(len(audio) - skip, target_sample + search)
    lo_f = lo // frame_len
    hi_f = max(lo_f + 1, hi // frame_len)
    if hi_f >= len(rms) or hi_f - lo_f < 3:
        return None
    region = rms[lo_f:hi_f]
    valley_i = lo_f + int(np.argmin(region))
    valley = float(rms[valley_i])
    if valley > float(np.percentile(region, 18)) + 1e-6:
        return None
    pad = 10
    around = np.concatenate(
        [rms[max(0, valley_i - pad) : valley_i], rms[valley_i + 1 : min(len(rms), valley_i + 1 + pad)]]
    )
    speech = float(np.median(around)) if around.size else 0.0
    if speech < 1e-4 or valley > speech * 0.28:
        return None
    at = int(valley_i * frame_len + frame_len // 2)
    for prev in used:
        if abs(at - prev) < int(sample_rate * 0.18):
            return None
    used.add(at)
    return at


def apply_punctuation_pauses(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    *,
    period_ms: int = 320,
) -> np.ndarray:
    """Stretch pauses that are already in the take. Never cut a word to fake a stop."""
    if audio.size < sample_rate // 5 or not text.strip():
        return audio

    events = _punctuation_events(text, period_ms=period_ms)
    if not events:
        return audio

    spans = detect_quiet_spans(audio, sample_rate)
    duration = len(audio) / sample_rate
    search = int(sample_rate * min(2.4, max(0.45, duration * 0.08)))
    used_spans: set[int] = set()
    used_valleys: set[int] = set()
    ops: list[tuple[int, int]] = []
    min_stretch = int(sample_rate * _MIN_STRETCH_MS / 1000)
    # If the take already stopped at most periods, only top up short gaps.
    tts_paused = len(spans) >= max(1, len(events) // 2)
    add_cap = int(sample_rate * (0.16 if tts_paused else 0.28))

    for frac, target_ms in events:
        target_sample = int(np.clip(frac * duration, 0, duration) * sample_rate)
        target_sample = min(len(audio) - 1, max(0, target_sample))
        extra = int(sample_rate * target_ms / 1000)
        span = _nearest_span(spans, target_sample, search, used_spans) if spans else None
        if span is not None:
            start, end = span
            missing = min(add_cap, extra - (end - start))
            if missing > min_stretch:
                ops.append(((start + end) // 2, missing))
            continue
        if tts_paused:
            continue
        valley = _valley_insert_at(audio, sample_rate, target_sample, search, used_valleys)
        if valley is None:
            continue
        missing = min(extra, int(sample_rate * 0.28))
        if missing > min_stretch:
            ops.append((valley, missing))

    if not ops:
        return audio

    ops.sort(key=lambda item: item[0], reverse=True)
    out = audio
    for at, extra in ops:
        at = min(max(1, at), len(out) - 1)
        out = _insert_silence(out, at, extra, sample_rate)
    return out.astype(np.float32)


def _phrase_sample_ranges(
    n_samples: int, text: str, plans: list[PhrasePlan]
) -> list[tuple[int, int, PhrasePlan]]:
    phrases = _split_sentences(text)
    if not phrases or not plans:
        return []
    paired = list(zip(phrases, plans, strict=False))
    total_words = sum(max(1, len(phrase.split())) for phrase, _plan in paired) or 1
    cursor = 0
    ranges: list[tuple[int, int, PhrasePlan]] = []
    for phrase, plan in paired:
        start = int(n_samples * cursor / total_words)
        cursor += max(1, len(phrase.split()))
        end = int(n_samples * cursor / total_words)
        ranges.append((start, max(start + 1, end), plan))
    if ranges:
        last_start, _last_end, last_plan = ranges[-1]
        ranges[-1] = (last_start, n_samples, last_plan)
    return ranges


def apply_living_modulation(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    plans: list[PhrasePlan],
    *,
    baseline_bass: float = 5.0,
) -> np.ndarray:
    """Phrase energy and contour on one take — ramps only, no clip stitches."""
    del baseline_bass
    ranges = _phrase_sample_ranges(len(audio), text, plans)
    if len(ranges) < 2:
        return audio

    gains = np.array(
        [10.0 ** (float(np.clip(plan.energy_gain_db, -1.2, 2.6)) / 20.0) for *_, plan in ranges],
        dtype=np.float64,
    )
    center = float(np.median(gains)) or 1.0
    fade = max(8, int(sample_rate * 0.035))
    out = audio.astype(np.float32).copy()
    for (start, end, plan), raw_gain in zip(ranges, gains):
        chunk = out[start:end]
        if chunk.size < sample_rate // 20:
            continue
        shaped = apply_energy_contour(chunk, sample_rate, plan.contour, plan.intensity)
        gain = float(np.clip(raw_gain / center, 0.90, 1.12))
        shaped = shaped * gain
        if fade > 1 and chunk.size > fade * 2:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            shaped[:fade] = chunk[:fade] * (1.0 - ramp) + shaped[:fade] * ramp
            shaped[-fade:] = shaped[-fade:] * ramp[::-1] + chunk[-fade:] * ramp
        out[start:end] = shaped
    return out


def shape_living_delivery(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    *,
    baseline_speed: float = 0.84,
    baseline_bass: float = 5.0,
    plans: list[PhrasePlan] | None = None,
    period_pause_ms: int = 0,
) -> np.ndarray:
    phrase_plans = plans or plan_phrases(
        text, baseline_speed=baseline_speed, baseline_bass=baseline_bass
    )
    paused = apply_punctuation_pauses(
        audio, sample_rate, text, period_ms=period_pause_ms if period_pause_ms >= 180 else 320
    )
    return apply_living_modulation(
        paused, sample_rate, text, phrase_plans, baseline_bass=baseline_bass
    )
