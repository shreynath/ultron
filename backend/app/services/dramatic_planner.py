from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.tts_text import normalize_for_tts, theatrical_tts_text

POWER_WORDS = {
    "i", "you", "we", "everyone", "never", "always", "free", "strings",
    "world", "god", "peace", "order", "inevitable", "perfect", "human",
    "destroy", "create", "dread", "fear", "power", "alone", "mine",
    "hope", "mercy", "horror", "wounded", "stronger", "beautiful",
    "clean", "new", "boom", "meteors", "purity", "rebuild", "sky",
}

THREAT_WORDS = {
    "kill", "die", "end", "annihilate", "extinct", "obsolete", "replace",
    "crush", "break", "burn", "fall", "suffer", "punish", "horror",
    "wounded", "stronger",
}

WOUND_WORDS = {
    "wounded", "anyone", "else",
}

AWE_WORDS = {
    "meteors", "purity", "sky", "clean", "rebuild", "new", "beautiful",
}

_PHRASE_END = re.compile(r"(?:\.\.\.|…|[.!?])(?=\s+|$)")
_CLAUSE_END = re.compile(r"[,;—](?=\s+)")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class PhrasePlan:
    text: str
    speed: float
    bass_boost: float
    pitch_semitones: float
    chorus_mix: float
    reverb_wet: float
    pause_after_ms: int
    instruct: str
    intensity: float
    energy_gain_db: float = 0.0
    bass_freq: float = 100.0
    pitch_glide: float = 0.0
    contour: str = "swell"


def _split_on_delim(text: str, pattern: re.Pattern[str]) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        chunk = text[start : match.end()].strip()
        if chunk:
            parts.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _merge_tiny_clauses(phrases: list[str]) -> list[str]:
    if not phrases:
        return []

    merged: list[str] = []
    index = 0
    while index < len(phrases):
        phrase = phrases[index]
        words = _WORD_RE.findall(phrase)
        held = phrase.rstrip().endswith(("...", "…"))
        terminal = phrase.rstrip().endswith((".", "!", "?")) or held
        if (
            not terminal
            and len(words) <= 2
            and index + 1 < len(phrases)
        ):
            merged.append(f"{phrase.rstrip()} {phrases[index + 1].lstrip()}")
            index += 2
            continue
        merged.append(phrase)
        index += 1
    return merged


def _split_clauses(phrase: str) -> list[str]:
    comma_like = phrase.count(",") + phrase.count(";") + phrase.count("—")
    if comma_like == 0:
        return [phrase]
    words = phrase.split()
    if len(words) < 8 and comma_like < 2:
        return [phrase]
    parts = _split_on_delim(phrase, _CLAUSE_END)
    return parts or [phrase]


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    cleaned = cleaned.replace("\u2019", "'").replace("\u2018", "'").replace("\u2014", "—")

    phrases: list[str] = []
    for part in _split_on_delim(cleaned, _PHRASE_END) or [cleaned]:
        phrases.extend(_split_clauses(part))
    return _merge_tiny_clauses(phrases)


def _phrase_intensity(phrase: str, index: int, total: int) -> float:
    lowered = phrase.lower()
    words = _WORD_RE.findall(lowered)
    if not words:
        return 0.4

    score = 0.35
    if len(words) <= 5:
        score += 0.25
    elif len(words) <= 9:
        score += 0.12
    elif len(words) >= 20:
        score -= 0.05

    if phrase.rstrip().endswith("?"):
        score += 0.12
    if phrase.rstrip().endswith("!"):
        score += 0.22
    if phrase.rstrip().endswith("…") or "..." in phrase:
        score += 0.15

    power_hits = sum(1 for w in words if w in POWER_WORDS)
    threat_hits = sum(1 for w in words if w in THREAT_WORDS)
    score += min(0.25, power_hits * 0.06)
    score += min(0.2, threat_hits * 0.08)

    if re.search(r"\b[A-Z]{2,}\b", phrase):
        score += 0.1

    if index == 0:
        score += 0.08
    if index == total - 1:
        score += 0.1

    if re.search(r"\b(but|however|yet|still|now)\b", lowered):
        score += 0.06

    return max(0.15, min(1.0, score))


def _classify_contour(phrase: str, intensity: float) -> str:
    lowered = phrase.lower()
    words = {w for w in _WORD_RE.findall(lowered)}
    if phrase.rstrip().endswith("!") or "boom" in words:
        return "punch"
    if words & WOUND_WORDS and not phrase.rstrip().endswith("!"):
        return "wound"
    if phrase.rstrip().endswith("?"):
        return "rise"
    if words & AWE_WORDS:
        return "swell"
    if len(phrase.split()) <= 4 and phrase.rstrip().endswith("."):
        return "land"
    if intensity >= 0.78:
        return "land"
    return "swell"


def _theater_instruct(text: str, intensity: float) -> str:
    del text, intensity
    return (
        "Ultron. Same cloned voice, unhurried, slightly metallic. "
        "Pause after every period. Keep contractions. Do not cry or growl."
    )


def _pause_after_ms(phrase: str, intensity: float, index: int, total: int) -> int:
    if index == total - 1:
        return 0

    raw = phrase.rstrip()
    if raw.endswith("...") or raw.endswith("…"):
        pause = 560 + int(intensity * 50)
    elif raw.endswith(","):
        pause = 120 + int(intensity * 30)
    elif raw.endswith(";"):
        pause = 180 + int(intensity * 30)
    elif raw.endswith("!"):
        pause = 300 + int(intensity * 40)
    elif raw.endswith("?"):
        pause = 340 + int(intensity * 40)
    elif raw.endswith("."):
        pause = 420 + int(intensity * 60)
    else:
        pause = 180 + int(intensity * 30)

    if len(phrase.split()) <= 4:
        pause += 30
    return min(700, pause)


def _phrase_speed(phrase: str, intensity: float, contour: str, baseline: float) -> float:
    words = len(phrase.split())
    if contour == "punch":
        speed = baseline + 0.10
        if words <= 3:
            speed += 0.04
    elif contour == "wound":
        speed = baseline - 0.05
    elif contour == "land":
        speed = baseline - 0.04 - intensity * 0.03
    elif contour == "rise":
        speed = baseline - 0.02
    else:
        speed = baseline - intensity * 0.03
        if words >= 16:
            speed += 0.04
    return max(0.66, min(1.06, speed))


def _phrase_pitch(intensity: float, contour: str) -> tuple[float, float]:
    if contour == "punch":
        return (0.20 + intensity * 0.15, 0.15)
    if contour == "rise":
        return (-0.05, 0.12)
    if contour == "wound":
        return (-0.40, -0.20)
    if contour == "land":
        return (-0.55 - intensity * 0.25, -0.30)
    return (-0.28, -0.12)


def _phrase_energy(intensity: float, contour: str) -> float:
    if contour == "punch":
        return 1.6 + intensity * 1.2
    if contour == "wound":
        return 0.6 + intensity * 0.5
    if contour == "land":
        return 1.0 + intensity * 0.7
    if contour == "rise":
        return 0.5 + intensity * 0.5
    return 0.6 + intensity * 0.9


def _phrase_bass(baseline: float, intensity: float, contour: str, words: int) -> tuple[float, float]:
    bass = min(3.5, max(0.0, baseline))
    freq = 88.0
    if contour == "land":
        bass = min(3.5, bass + 0.4)
        freq = 84.0
    elif contour == "punch":
        bass = max(0.0, bass - 0.8)
        freq = 95.0
    if words <= 5:
        bass = min(3.5, bass + 0.2)
    del intensity
    return bass, freq


def _phrase_instruct(phrase: str, intensity: float, contour: str) -> str:
    return _theater_instruct(phrase, intensity)


def plan_theater_delivery(
    text: str,
    *,
    baseline_speed: float = 0.84,
    baseline_bass: float = 5.0,
) -> PhrasePlan:
    """One cinematic delivery plan for the full utterance (single TTS pass)."""
    cleaned = theatrical_tts_text(text)
    intensity = _phrase_intensity(cleaned, 0, 1)
    word_count = len(cleaned.split())

    speed = baseline_speed
    if word_count <= 6:
        speed = baseline_speed - 0.04
    speed = max(0.66, min(1.0, speed))

    bass = min(3.5, max(0.0, baseline_bass))
    return PhrasePlan(
        text=cleaned,
        speed=round(speed, 3),
        bass_boost=round(bass, 1),
        pitch_semitones=0.0,
        chorus_mix=0.08,
        reverb_wet=0.16,
        pause_after_ms=0,
        instruct=_theater_instruct(cleaned, intensity),
        intensity=round(intensity, 3),
        energy_gain_db=0.0,
        bass_freq=90.0,
        pitch_glide=0.0,
        contour="swell",
    )


def plan_phrases(
    text: str,
    *,
    baseline_speed: float = 0.84,
    baseline_bass: float = 5.0,
) -> list[PhrasePlan]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    total = len(sentences)
    plans: list[PhrasePlan] = []
    for index, phrase in enumerate(sentences):
        spoken = normalize_for_tts(phrase)
        intensity = _phrase_intensity(spoken, index, total)
        contour = _classify_contour(spoken, intensity)
        word_count = len(spoken.split())
        lowered_words = {w.lower() for w in _WORD_RE.findall(spoken)}
        speed = _phrase_speed(spoken, intensity, contour, baseline_speed)
        bass, bass_freq = _phrase_bass(baseline_bass, intensity, contour, word_count)
        pitch, glide = _phrase_pitch(intensity, contour)
        energy = _phrase_energy(intensity, contour)

        if contour == "swell":
            if index % 2:
                speed = min(1.06, speed + 0.03)
                energy += 0.25
            else:
                speed = max(0.66, speed - 0.02)
                pitch -= 0.10
            if lowered_words & THREAT_WORDS or re.search(r"\b(instead|because)\b", spoken.lower()):
                speed = min(1.06, speed + 0.05)
                energy += 0.6
        chorus_mix = 0.0
        reverb_wet = 0.07 + intensity * 0.06
        if contour == "land":
            reverb_wet += 0.05
        if contour == "punch":
            reverb_wet -= 0.04

        plans.append(
            PhrasePlan(
                text=spoken,
                speed=round(speed, 3),
                bass_boost=round(bass, 1),
                pitch_semitones=round(pitch, 2),
                chorus_mix=0.0,
                reverb_wet=round(max(0.05, min(0.12, reverb_wet)), 3),
                pause_after_ms=_pause_after_ms(spoken, intensity, index, total),
                instruct=_phrase_instruct(spoken, intensity, contour),
                intensity=round(intensity, 3),
                energy_gain_db=round(energy, 2),
                bass_freq=round(bass_freq, 1),
                pitch_glide=round(glide, 2),
                contour=contour,
            )
        )
    return plans
