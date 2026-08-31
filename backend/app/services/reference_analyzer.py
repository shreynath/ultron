from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

OTHER_CHARACTER_MARKERS = [
    "tony",
    "stark",
    "captain",
    "thor",
    "banner",
    "bruce",
    "romanoff",
    "barton",
    "fury",
    "jarvis",
    "vision",
    "wanda",
    "pietro",
    "steve",
    "clint",
    "natasha",
]

DIALOGUE_REJECT_PATTERNS = [
    r"\?\s+[A-Z]",  # question then another speaker
    r"^yes[,.]",
    r"^no[,.]",
    r"\bsaid\b",
    r"\basked\b",
]


@dataclass
class ClipCandidate:
    start_sec: float
    end_sec: float
    score: float
    transcript: str = ""
    path: Path | None = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def _band_rms_db(wav_path: Path, start: float, duration: float, low_hz: float, high_hz: float) -> float:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        str(start),
        "-i",
        str(wav_path),
        "-t",
        str(duration),
        "-af",
        f"highpass=f={low_hz},lowpass=f={high_hz},astats=metadata=1:reset=1,ametadata=print:file=-",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    values = [float(v) for v in re.findall(r"RMS level dB: (-?[0-9.]+)", result.stderr)]
    if not values:
        return -60.0
    return sum(values) / len(values)


def estimate_deep_voice_score(wav_path: Path, start: float, duration: float) -> float:
    """Higher score = more energy in low band (deeper voice)."""
    low = _band_rms_db(wav_path, start, duration, 70, 320)
    mid = _band_rms_db(wav_path, start, duration, 320, 1200)
    high = _band_rms_db(wav_path, start, duration, 1200, 5000)
    low_energy = max(-60.0, low)
    high_energy = max(-60.0, high)
    depth = low_energy - high_energy
    mid_balance = low_energy - mid
    return max(0.0, min(1.0, (depth + 8) / 18)) * 0.6 + max(0.0, min(1.0, (mid_balance + 6) / 14)) * 0.4


def score_transcript_solo_speech(transcript: str) -> float:
    text = transcript.strip()
    if not text:
        return 0.0
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    if len(words) < 8:
        return 0.1

    score = 0.5
    if len(words) >= 18:
        score += 0.2
    if len(words) >= 35:
        score += 0.1

    if "?" in text and len(words) < 25:
        score -= 0.25

    for pattern in DIALOGUE_REJECT_PATTERNS:
        if re.search(pattern, text):
            score -= 0.2

    other_hits = sum(1 for name in OTHER_CHARACTER_MARKERS if re.search(rf"\b{name}\b", lowered))
    if other_hits >= 3:
        score -= 0.4
    elif other_hits >= 2 and len(words) < 35:
        score -= 0.25
    elif other_hits == 1 and len(words) < 18:
        score -= 0.15

    # Ultron monologues rarely address people by name mid-sentence in short clips
    if lowered.count(",") >= 3 and len(words) < 40:
        score -= 0.1

    return max(0.0, min(1.0, score))


def is_likely_other_speaker(transcript: str, deep_score: float) -> bool:
    solo = score_transcript_solo_speech(transcript)
    word_count = len(transcript.split())
    if solo < 0.22 and word_count < 25:
        return True
    if deep_score < 0.12 and word_count < 15:
        return True
    return False


def estimate_speaking_rate_wps(transcript: str, duration_sec: float) -> float:
    words = len(re.findall(r"\w+", transcript))
    if duration_sec <= 0:
        return 2.5
    return words / duration_sec


def recommend_delivery_from_clips(
    clips: list[tuple[str, float, float]],
) -> dict[str, float]:
    """Return speed and bass_boost tuned from reference clip stats."""
    if not clips:
        return {"speed": 0.84, "bass_boost": 5.0}

    rates = [estimate_speaking_rate_wps(t, d) for t, d, _ in clips]
    avg_rate = sum(rates) / len(rates)
    if avg_rate <= 2.1:
        speed = 0.86
    elif avg_rate <= 2.5:
        speed = 0.90
    elif avg_rate <= 3.0:
        speed = 0.94
    else:
        speed = 0.97

    avg_deep = sum(deep for _, _, deep in clips) / len(clips)
    bass_boost = 2.0 + avg_deep * 3.5

    return {"speed": round(speed, 2), "bass_boost": round(min(6.0, bass_boost), 1)}
