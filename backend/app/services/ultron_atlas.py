from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from app.services.dramatic_planner import _split_sentences
from app.services.f0 import resample_contour, track_f0, voiced_median
from app.services.living_delivery import detect_quiet_spans
from app.services.td_psola import impose_f0_contour, lengthen_keep_pitch
from app.services.tts_text import normalize_for_tts
from app.services.ultron_tone import (
    LTAS_BANDS,
    apply_ltas_match,
    measure_formants,
    measure_ltas_db,
    shift_formants,
)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_ATLAS_NAME = "ultron_prosody_atlas.json"
_SENTENCE_MATCH = 0.70
_WINDOW_MATCH = 0.55


@dataclass
class AtlasPhrase:
    text: str
    words: list[str]
    word_f0: list[list[float]]
    duration_sec: float
    phrase_f0: list[float] = field(default_factory=list)
    ltas_db: list[float] = field(default_factory=list)
    f1: float = 0.0
    f2: float = 0.0
    energy: list[float] = field(default_factory=list)


def _norm_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(normalize_for_tts(text))]


def profile_atlas_path(data_dir: Path, profile_id: str) -> Path:
    return Path(data_dir) / "gold" / profile_id / "prosody_atlas.json"


def load_profile_atlas(data_dir: Path | str, profile_id: str) -> list[AtlasPhrase]:
    path = profile_atlas_path(Path(data_dir), profile_id)
    if not path.exists():
        return []
    return load_atlas(path)


def delivery_atlas(
    *,
    data_dir: Path | str | None = None,
    profile_id: str | None = None,
) -> list[AtlasPhrase]:
    """Prosody learned from uploads first, then the shipped Ultron atlas as fallback."""
    profile: list[AtlasPhrase] = []
    if data_dir and profile_id:
        profile = load_profile_atlas(data_dir, profile_id)
    builtin = load_atlas()
    if not profile:
        return builtin
    if not builtin:
        return profile
    return profile + builtin


def default_atlas_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / _ATLAS_NAME


def load_atlas(path: Path | None = None) -> list[AtlasPhrase]:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend(
        [
            default_atlas_path(),
            Path(__file__).resolve().parents[2] / "data" / "ultron_ref" / _ATLAS_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            raw = json.loads(candidate.read_text())
            return [
                AtlasPhrase(
                    text=item["text"],
                    words=list(item["words"]),
                    word_f0=[list(map(float, contour)) for contour in item["word_f0"]],
                    duration_sec=float(item.get("duration_sec", 0.0)),
                    phrase_f0=[float(x) for x in item.get("phrase_f0", [])],
                    ltas_db=[float(x) for x in item.get("ltas_db", [])],
                    f1=float(item.get("f1", 0.0)),
                    f2=float(item.get("f2", 0.0)),
                    energy=[float(x) for x in item.get("energy", [])],
                )
                for item in raw.get("phrases", [])
            ]
    return []


def save_atlas(phrases: list[AtlasPhrase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phrases": [
            {
                "text": phrase.text,
                "words": phrase.words,
                "word_f0": phrase.word_f0,
                "duration_sec": phrase.duration_sec,
                "phrase_f0": [round(x, 2) for x in phrase.phrase_f0],
                "ltas_db": [round(x, 2) for x in phrase.ltas_db],
                "f1": round(phrase.f1, 1),
                "f2": round(phrase.f2, 1),
                "energy": [round(x, 5) for x in phrase.energy],
            }
            for phrase in phrases
        ]
    }
    path.write_text(json.dumps(payload))


def _frame_energy(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    hop = max(1, int(sample_rate * 0.01))
    frame = max(hop, int(sample_rate * 0.02))
    n = max(1, (len(audio) - frame) // hop + 1)
    energy = np.empty(n, dtype=np.float64)
    for index in range(n):
        chunk = audio[index * hop : index * hop + frame]
        energy[index] = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
    return energy, hop


def _speaking_bounds(energy: np.ndarray, hop: int, n_audio: int) -> tuple[int, int]:
    if energy.size == 0:
        return 0, n_audio
    thresh = max(0.008, float(np.percentile(energy, 55)) * 0.28)
    voiced = energy > thresh
    if not np.any(voiced):
        return 0, n_audio
    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    start = first * hop
    end = min(n_audio, (last + 1) * hop)
    return start, max(start + hop, end)


def film_weighted_split_ranges(
    audio: np.ndarray,
    sample_rate: int,
    film_sample_weights: list[int],
) -> list[tuple[int, int]]:
    """Split TTS audio in proportion to isolated film clip lengths."""
    n = len(audio)
    if not film_sample_weights:
        return []
    if len(film_sample_weights) == 1:
        return [(0, n)]
    energy, hop = _frame_energy(audio, sample_rate)
    speak0, speak1 = _speaking_bounds(energy, hop, n)
    region = speak1 - speak0
    if region < sample_rate // 20:
        speak0, speak1, region = 0, n, n
    weights = np.array([max(1, int(w)) for w in film_sample_weights], dtype=np.float64)
    fracs = np.cumsum(weights / weights.sum())[:-1]
    cuts = [speak0 + int(region * float(frac)) for frac in fracs]
    ranges: list[tuple[int, int]] = []
    start = speak0
    for cut in cuts:
        end = max(start + hop, min(int(cut), speak1))
        ranges.append((start, end))
        start = end
    ranges.append((start, speak1))
    return ranges


def energy_split_ranges(
    audio: np.ndarray,
    sample_rate: int,
    pieces: list[str],
    piece_weights: list[float] | None = None,
) -> list[tuple[int, int]]:
    """Cut `audio` into `pieces` by cumulative speech energy, not clock time."""
    if not pieces:
        return []
    n = len(audio)
    if n < sample_rate // 20 or len(pieces) == 1:
        return [(0, n)]

    energy, hop = _frame_energy(audio, sample_rate)
    speak0, speak1 = _speaking_bounds(energy, hop, n)
    region_frames = energy[speak0 // hop : max(speak0 // hop + 1, speak1 // hop)]
    if region_frames.size < 2:
        region_frames = energy
        speak0, speak1 = 0, n
    if piece_weights is not None and len(piece_weights) == len(pieces):
        weights = np.array([max(0.2, float(w)) for w in piece_weights], dtype=np.float64)
    else:
        weights = np.array([max(1, len(piece.strip())) for piece in pieces], dtype=np.float64)
    fractions = np.cumsum(weights / weights.sum())[:-1]
    cum = np.cumsum(region_frames + 1e-8)
    cum /= cum[-1]
    search = max(2, int(0.10 / 0.01))
    split_samples: list[int] = []
    for frac in fractions:
        center = int(np.clip(np.searchsorted(cum, frac), 0, len(region_frames) - 1))
        lo = max(0, center - search)
        hi = min(len(region_frames), center + search + 1)
        valley = lo + int(np.argmin(region_frames[lo:hi]))
        split_samples.append(int(np.clip(speak0 + valley * hop, speak0 + hop, speak1 - hop)))

    ranges: list[tuple[int, int]] = []
    start = speak0
    for cut in split_samples:
        end = max(start + hop, cut)
        ranges.append((start, end))
        start = end
    ranges.append((start, n))
    return ranges


def word_sample_ranges(audio: np.ndarray, sample_rate: int, words: list[str]) -> list[tuple[int, int]]:
    """Split a take into word spans by snapping letter-weights onto energy valleys."""
    return energy_split_ranges(audio, sample_rate, words)


def sentence_sample_ranges(
    audio: np.ndarray,
    sample_rate: int,
    sentences: list[str],
) -> list[tuple[int, int]]:
    """Split on real pauses between sentences when they exist."""
    if not sentences:
        return []
    n = len(audio)
    if len(sentences) == 1 or n < sample_rate // 20:
        return [(0, n)]

    quiet = detect_quiet_spans(audio, sample_rate, min_ms=50)
    if len(quiet) < 1:
        return energy_split_ranges(audio, sample_rate, sentences)

    weights = np.array([max(1, len(sentence.split())) for sentence in sentences], dtype=np.float64)
    fractions = np.cumsum(weights / weights.sum())[:-1]
    used: set[int] = set()
    cuts: list[int] = []
    search = int(sample_rate * min(1.6, max(0.35, (n / sample_rate) * 0.22)))
    for frac in fractions:
        target = int(np.clip(frac * n, 0, n - 1))
        best_mid = None
        best_dist = search + 1
        best_index = -1
        for index, (start, end) in enumerate(quiet):
            if index in used:
                continue
            mid = (start + end) // 2
            dist = abs(mid - target)
            if dist < best_dist:
                best_mid = mid
                best_dist = dist
                best_index = index
        if best_mid is None:
            return energy_split_ranges(audio, sample_rate, sentences)
        used.add(best_index)
        cuts.append(int(best_mid))
    cuts.sort()
    ranges: list[tuple[int, int]] = []
    start = 0
    hop = max(1, int(sample_rate * 0.01))
    for cut in cuts:
        end = max(start + hop, cut)
        ranges.append((start, end))
        start = end
    ranges.append((start, n))
    return ranges


def _contour_for_span(
    times: np.ndarray, f0: np.ndarray, start: int, end: int, sample_rate: int
) -> list[float]:
    t0 = start / sample_rate
    t1 = end / sample_rate
    mask = (times >= t0) & (times < t1)
    voiced = f0[mask]
    voiced = voiced[(voiced >= 55.0) & (voiced <= 150.0)]
    return [round(float(v), 2) for v in voiced]


def phrase_from_audio(text: str, audio: np.ndarray, sample_rate: int) -> AtlasPhrase | None:
    words = _norm_words(text)
    if len(words) < 1:
        return None
    times, f0 = track_f0(audio, sample_rate)
    if voiced_median(f0) < 52.0:
        return None
    ranges = word_sample_ranges(audio, sample_rate, words)
    word_f0 = [
        _contour_for_span(times, f0, start, end, sample_rate) for (start, end) in ranges[: len(words)]
    ]
    while len(word_f0) < len(words):
        word_f0.append([])
    f1, f2 = measure_formants(audio, sample_rate)
    energy, _hop = _frame_energy(audio, sample_rate)
    if energy.size > 48:
        idx = np.linspace(0, energy.size - 1, 48).astype(int)
        energy_pts = [round(float(energy[i]), 5) for i in idx]
    else:
        energy_pts = [round(float(x), 5) for x in energy]
    return AtlasPhrase(
        text=normalize_for_tts(text),
        words=words,
        word_f0=word_f0[: len(words)],
        duration_sec=len(audio) / sample_rate,
        phrase_f0=[round(float(v), 2) for v in f0],
        ltas_db=measure_ltas_db(audio, sample_rate),
        f1=f1,
        f2=f2,
        energy=energy_pts,
    )


def phrases_from_paragraph(text: str, audio: np.ndarray, sample_rate: int) -> list[AtlasPhrase]:
    sentences = _split_sentences(text) or [text]
    if len(sentences) == 1:
        phrase = phrase_from_audio(text, audio, sample_rate)
        return [phrase] if phrase else []
    ranges = energy_split_ranges(audio, sample_rate, sentences)
    out: list[AtlasPhrase] = []
    for sentence, (start, end) in zip(sentences, ranges):
        clip = audio[start:end]
        if clip.size < sample_rate // 5:
            continue
        phrase = phrase_from_audio(sentence, clip, sample_rate)
        if phrase is not None:
            out.append(phrase)
    if not out:
        phrase = phrase_from_audio(text, audio, sample_rate)
        if phrase is not None:
            out.append(phrase)
    return out


def _lexicon(phrases: list[AtlasPhrase]) -> dict[str, list[float]]:
    store: dict[str, list[list[float]]] = {}
    for phrase in phrases:
        for word, contour in zip(phrase.words, phrase.word_f0):
            voiced = [hz for hz in contour if hz > 1.0]
            if len(voiced) >= 2:
                store.setdefault(word, []).append(voiced)
    return {word: values[len(values) // 2] for word, values in store.items() if values}


def _window_ratio(query: list[str], haystack: list[str]) -> tuple[float, int]:
    if not query:
        return 0.0, 0
    if len(haystack) < len(query):
        return SequenceMatcher(None, query, haystack).ratio(), 0
    best = 0.0
    best_i = 0
    for index in range(len(haystack) - len(query) + 1):
        ratio = SequenceMatcher(None, query, haystack[index : index + len(query)]).ratio()
        if ratio > best:
            best = ratio
            best_i = index
    return best, best_i


def _filled_frac(phrase: AtlasPhrase, index: int, n_words: int) -> float:
    if n_words <= 0:
        return 0.0
    filled = 0
    for offset in range(n_words):
        pos = index + offset
        contour = phrase.word_f0[pos] if pos < len(phrase.word_f0) else []
        if sum(1 for hz in contour if hz > 1.0) >= 2:
            filled += 1
    return filled / n_words


def _best_phrase_match(words: list[str], phrases: list[AtlasPhrase]) -> tuple[AtlasPhrase | None, int, float]:
    best: tuple[float, int, int, AtlasPhrase] | None = None
    for phrase in phrases:
        if len(phrase.words) < 1:
            continue
        exact = 1.0 if phrase.words == words else 0.0
        ratio, index = _window_ratio(words, phrase.words)
        ratio = max(ratio, exact)
        quality = ratio * (0.55 + 0.45 * _filled_frac(phrase, index, len(words)))
        # Prefer a sentence-sized clip over an 18s blob with the same text window.
        quality -= 0.04 * min(1.0, abs(len(phrase.words) - len(words)) / max(6.0, len(words)))
        if best is None or quality > best[0]:
            best = (quality, index, len(phrase.words), phrase)
    if best is None:
        return None, 0, 0.0
    quality, index, _n, phrase = best
    ratio, index = _window_ratio(words, phrase.words)
    if phrase.words == words:
        ratio = 1.0
        index = 0
    return phrase, index, ratio


def _match_word_targets(words: list[str], phrases: list[AtlasPhrase]) -> list[float]:
    lexicon = {word: float(np.median(contour)) for word, contour in _lexicon(phrases).items()}
    phrase, index, ratio = _best_phrase_match(words, phrases)
    targets = [0.0] * len(words)
    if phrase is not None and ratio >= _WINDOW_MATCH:
        for offset in range(len(words)):
            pos = index + offset
            contour = phrase.word_f0[pos] if pos < len(phrase.word_f0) else []
            voiced = [hz for hz in contour if hz > 1.0]
            targets[offset] = float(np.median(voiced)) if voiced else 0.0
    filled: list[float] = []
    for word, hz in zip(words, targets):
        if hz > 1.0:
            filled.append(hz)
        else:
            filled.append(lexicon.get(word, 0.0))
    return filled


def _match_word_contours(words: list[str], phrases: list[AtlasPhrase]) -> list[list[float]]:
    lexicon = _lexicon(phrases)
    phrase, index, ratio = _best_phrase_match(words, phrases)
    contours: list[list[float]] = []
    for offset, word in enumerate(words):
        contour: list[float] = []
        if phrase is not None and ratio >= _WINDOW_MATCH:
            pos = index + offset
            raw = phrase.word_f0[pos] if pos < len(phrase.word_f0) else []
            contour = [hz for hz in raw if hz > 1.0]
        if len(contour) < 2:
            contour = list(lexicon.get(word, []))
        contours.append(contour)
    return contours


def _sanitize_contour(contour: list[float], *, fall: bool) -> np.ndarray:
    """Drop octave jumps and stop statement words from curling up like a question."""
    arr = np.array([hz for hz in contour if 55.0 <= hz <= 150.0], dtype=np.float64)
    if arr.size < 2:
        return arr
    median = float(np.median(arr))
    kept = arr[(arr >= median * 0.58) & (arr <= median * 1.32)]
    if kept.size >= 2:
        arr = kept
    if fall and arr.size >= 4:
        split = max(2, int(arr.size * 0.62))
        head = float(np.median(arr[:split]))
        # Only flatten a question lift. Do not crash a held word like "beautiful".
        if float(arr[-1]) > head + 10.0:
            dest = max(60.0, head)
            arr = arr.copy()
            arr[split:] = np.linspace(float(arr[split]), dest, arr.size - split)
    return arr


def _flatten_question_tail(times: np.ndarray, target: np.ndarray) -> np.ndarray:
    """If the last 250 ms jumps up like a question, hold the body pitch. No death-drop."""
    voiced = target > 1.0
    if int(np.sum(voiced)) < 8:
        return target
    t_end = float(times[voiced][-1])
    tail = (times >= t_end - 0.22) & (target > 1.0)
    body = (times >= t_end - 0.70) & (times < t_end - 0.22) & (target > 1.0)
    if int(np.sum(tail)) < 4 or not np.any(body):
        return target
    body_hz = float(np.median(target[body]))
    tail_hz = float(np.median(target[tail]))
    if tail_hz <= body_hz + 14.0:
        return target
    out = target.copy()
    out[tail] = body_hz
    return out


def _limit_f0_slope(
    times: np.ndarray,
    target: np.ndarray,
    *,
    max_hz_per_sec: float = 320.0,
) -> np.ndarray:
    """Stop PSOLA from leaping 40+ Hz at every period — that is the ugly warble."""
    voiced = np.flatnonzero(target > 1.0)
    if voiced.size < 4:
        return target
    out = target.copy()
    for index in range(1, voiced.size):
        prev_i = voiced[index - 1]
        this_i = voiced[index]
        dt = max(0.004, float(times[this_i] - times[prev_i]))
        # A pause is not a license to retune. Treat gaps as one hop so periods don't warble.
        span = min(dt, 0.025)
        max_delta = max_hz_per_sec * span
        delta = float(out[this_i] - out[prev_i])
        if abs(delta) > max_delta:
            out[this_i] = out[prev_i] + float(np.clip(delta, -max_delta, max_delta))
    return out


def _unify_long_contour(
    times: np.ndarray,
    source_f0: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Keep atlas expression. Do not copy clone jitter into the holes — that is the drone."""
    return _limit_f0_slope(times, target, max_hz_per_sec=380.0)


def _join_phrase_audio(pieces: list[np.ndarray], sample_rate: int) -> np.ndarray:
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    fade = max(4, int(sample_rate * 0.01))
    out = pieces[0].astype(np.float32)
    for piece in pieces[1:]:
        nxt = piece.astype(np.float32)
        if fade > 1 and len(out) > fade and len(nxt) > fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            overlap = out[-fade:] * (1.0 - ramp) + nxt[:fade] * ramp
            out = np.concatenate([out[:-fade], overlap, nxt[fade:]])
        else:
            out = np.concatenate([out, nxt])
    return out


def _match_phrase_timing(
    audio: np.ndarray,
    sample_rate: int,
    sentences: list[str],
    atlas: list[AtlasPhrase],
) -> np.ndarray:
    """Pull each line toward Ultron's filmed duration so Boom snaps and purity lingers."""
    if len(sentences) < 2 or audio.size < sample_rate // 4:
        return audio
    spans = sentence_sample_ranges(audio, sample_rate, sentences)
    pieces: list[np.ndarray] = []
    for sentence, (start, end) in zip(sentences, spans):
        chunk = audio[start:end]
        words = _norm_words(sentence)
        phrase, _index, ratio = _best_phrase_match(words, atlas)
        if (
            phrase is None
            or ratio < 0.70
            or phrase.duration_sec < 0.28
            or chunk.size < sample_rate // 8
        ):
            pieces.append(chunk)
            continue
        ours = chunk.size / sample_rate
        # Only linger toward the clip. Never speed a line up — that is the drone/rush.
        factor = float(np.clip(phrase.duration_sec / max(0.22, ours), 1.0, 1.14))
        pieces.append(lengthen_keep_pitch(chunk, sample_rate, factor))
    return _join_phrase_audio(pieces, sample_rate) if pieces else audio


def _match_phrase_energy(
    audio: np.ndarray,
    sample_rate: int,
    sentences: list[str],
    atlas: list[AtlasPhrase],
) -> np.ndarray:
    """Copy Ultron's filmed loudness shape onto each matched line."""
    spans = sentence_sample_ranges(audio, sample_rate, sentences)
    out = audio.astype(np.float32).copy()
    for sentence, (start, end) in zip(sentences, spans):
        n = end - start
        if n < sample_rate // 16:
            continue
        phrase, _index, ratio = _best_phrase_match(_norm_words(sentence), atlas)
        if phrase is None or ratio < 0.70 or len(phrase.energy) < 4:
            continue
        env = resample_contour(np.array(phrase.energy, dtype=np.float64), n)
        med = float(np.median(env))
        if med < 1e-6:
            continue
        env = env / med
        env = np.clip(env, 0.85, 1.32)
        win = max(5, int(sample_rate * 0.045))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64) / win
        env = np.convolve(np.pad(env, win // 2, mode="edge"), kernel, mode="valid")[:n]
        env = env / (float(np.mean(env)) + 1e-9)
        out[start:end] *= env.astype(np.float32)
    return out


def _smooth_voiced_f0(target: np.ndarray) -> np.ndarray:
    """Kill single-frame jumps (clone question ticks, octave flips) without adding vibrato."""
    voiced = np.flatnonzero(target > 1.0)
    if voiced.size < 5:
        return target
    values = target[voiced].copy()
    padded = np.pad(values, 4, mode="edge")
    smoothed = np.empty_like(values)
    for index in range(values.size):
        smoothed[index] = float(np.median(padded[index : index + 9]))
    out = target.copy()
    out[voiced] = smoothed
    return out


def _median_ltas(phrases: list[AtlasPhrase]) -> list[float]:
    n_bands = len(LTAS_BANDS)
    rows = [p.ltas_db for p in phrases if len(p.ltas_db) == n_bands and any(abs(x) > 1 for x in p.ltas_db)]
    if not rows:
        return []
    return [float(np.median([row[i] for row in rows])) for i in range(n_bands)]


def apply_ultron_word_prosody(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    phrases: list[AtlasPhrase] | None = None,
) -> np.ndarray:
    """Stamp Ultron word pitch and tone onto one continuous take — one PSOLA, one EQ."""
    atlas = phrases if phrases is not None else load_atlas()
    words = _norm_words(text)
    if not atlas or len(words) < 2 or audio.size < sample_rate // 5:
        return audio

    sentences = _split_sentences(text) or [text]
    if len(sentences) >= 2:
        audio = _match_phrase_timing(audio, sample_rate, sentences, atlas)
    spans = sentence_sample_ranges(audio, sample_rate, sentences)
    times, source_f0 = track_f0(audio, sample_rate)
    if times.size < 8:
        return audio

    energy, hop = _frame_energy(audio, sample_rate)
    thresh = max(0.010, float(np.percentile(energy, 45)) * 0.28)
    target = np.zeros_like(source_f0)
    f2_refs: list[float] = []
    matched_ltas: list[AtlasPhrase] = []
    last_index = len(sentences) - 1

    for sent_index, (sentence, (sample_start, sample_end)) in enumerate(zip(sentences, spans)):
        sent_words = _norm_words(sentence)
        if not sent_words:
            continue
        phrase, _index, ratio = _best_phrase_match(sent_words, atlas)
        if phrase is not None and phrase.f2 > 600:
            f2_refs.append(phrase.f2)
        if phrase is not None and ratio >= _SENTENCE_MATCH and phrase.ltas_db:
            matched_ltas.append(phrase)

        contours = _match_word_contours(sent_words, atlas)
        chunk = audio[sample_start:sample_end]
        ranges = word_sample_ranges(chunk, sample_rate, sent_words)
        is_final = sent_index == last_index and not sentence.rstrip().endswith("?")
        for word_index, ((local_start, local_end), contour) in enumerate(zip(ranges, contours)):
            fall = is_final and word_index == len(sent_words) - 1
            clean = _sanitize_contour(contour, fall=fall)
            if clean.size < 2:
                continue
            t0 = (sample_start + local_start) / sample_rate
            t1 = (sample_start + local_end) / sample_rate
            mask = (times >= t0) & (times < t1)
            for frame_index, time in enumerate(times):
                if not mask[frame_index]:
                    continue
                energy_index = min(len(energy) - 1, int(time * sample_rate / max(1, hop)))
                if energy[energy_index] < thresh and source_f0[frame_index] < 1.0:
                    mask[frame_index] = False
            count = int(np.sum(mask))
            if count < 3:
                continue
            target[mask] = np.clip(resample_contour(clean, count), 55.0, 150.0)

    if len(sentences) >= 4:
        target = _unify_long_contour(times, source_f0, target)
    last = sentences[-1].rstrip() if sentences else ""
    if last.endswith(".") and not last.endswith("...") and not last.endswith("…"):
        target = _flatten_question_tail(times, target)
    target = _smooth_voiced_f0(target)
    if int(np.sum(target > 1.0)) < 4:
        return audio

    # Pitch each sentence on its own so F0 cannot interpolate across pauses.
    out = audio.copy()
    for sentence, (sample_start, sample_end) in zip(sentences, spans):
        if sample_end - sample_start < sample_rate // 12:
            continue
        t0 = sample_start / sample_rate
        t1 = sample_end / sample_rate
        mask = (times >= t0) & (times < t1)
        if int(np.sum(target[mask] > 1.0)) < 4:
            continue
        local_times = times[mask] - t0
        pitched = impose_f0_contour(
            audio[sample_start:sample_end], sample_rate, local_times, target[mask]
        )
        out[sample_start:sample_end] = pitched

    _f1_now, f2_now = measure_formants(out, sample_rate)
    if f2_refs and f2_now > 600:
        formant_ratio = float(np.median(f2_refs)) / f2_now
        if abs(formant_ratio - 1.0) >= 0.04:
            out = shift_formants(out, sample_rate, formant_ratio)

    target_ltas = _median_ltas(matched_ltas) or _median_ltas(atlas)
    if target_ltas:
        out = apply_ltas_match(out, sample_rate, target_ltas)
    return out
