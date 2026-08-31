from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SILENCE_NOISE_DB = -32
SILENCE_MIN_DURATION = 0.28
MIN_UTTERANCE_SEC = 3.0
MAX_UTTERANCE_SEC = 24.0
MERGE_GAP_SEC = 0.85


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def _run_ffmpeg_silencedetect(wav_path: Path) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(wav_path),
        "-af",
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DURATION}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    silence_starts = [float(v) for v in re.findall(r"silence_start: ([0-9.]+)", output)]
    silence_ends = [float(v) for v in re.findall(r"silence_end: ([0-9.]+)", output)]

    if not silence_ends:
        duration = _probe_duration(wav_path)
        return [(0.0, duration)] if duration > MIN_UTTERANCE_SEC else []

    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for end in silence_ends:
        if end > cursor:
            segments.append((cursor, end))
        if silence_starts:
            cursor = silence_starts.pop(0)
    duration = _probe_duration(wav_path)
    if duration - cursor >= MIN_UTTERANCE_SEC:
        segments.append((cursor, duration))
    return segments


def _probe_duration(wav_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _merge_close_segments(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    if not segments:
        return []
    merged: list[SpeechSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start_sec - prev.end_sec <= MERGE_GAP_SEC:
            merged[-1] = SpeechSegment(prev.start_sec, seg.end_sec)
        else:
            merged.append(seg)
    return merged


def detect_speech_segments(wav_path: Path) -> list[SpeechSegment]:
    raw = _run_ffmpeg_silencedetect(wav_path)
    segments = [
        SpeechSegment(start, end)
        for start, end in raw
        if MIN_UTTERANCE_SEC <= (end - start) <= MAX_UTTERANCE_SEC
    ]
    return _merge_close_segments(segments)


def split_long_segment(segment: SpeechSegment, max_len: float = MAX_UTTERANCE_SEC) -> list[SpeechSegment]:
    if segment.duration <= max_len:
        return [segment]
    parts: list[SpeechSegment] = []
    start = segment.start_sec
    while start < segment.end_sec:
        end = min(start + max_len, segment.end_sec)
        if end - start >= MIN_UTTERANCE_SEC:
            parts.append(SpeechSegment(start, end))
        start = end
    return parts
