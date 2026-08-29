from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 48000
CLIP_TARGET_SECONDS = 18.0
CLIP_MIN_SECONDS = 12.0
CLIP_MAX_SECONDS = 22.0
WINDOW_STEP_SECONDS = 6.0


@dataclass
class SpeechClip:
    start_sec: float
    end_sec: float
    score: float
    path: Path


def _duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _extract_segment(
    wav_path: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
    *,
    normalize: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = ["highpass=f=80"]
    if normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(wav_path),
        "-t",
        str(duration_sec),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-af",
        ",".join(filters),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _measure_segment(wav_path: Path, start_sec: float, duration_sec: float) -> float:
    """Score how speech-like a window is. Higher is better."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        str(start_sec),
        "-i",
        str(wav_path),
        "-t",
        str(duration_sec),
        "-ac",
        "1",
        "-af",
        "astats=metadata=1:reset=1,ametadata=print:file=-",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    rms_values = [float(v) for v in re.findall(r"RMS level dB: (-?[0-9.]+)", output)]
    peak_values = [float(v) for v in re.findall(r"Peak level dB: (-?[0-9.]+)", output)]
    if not rms_values or not peak_values:
        return 0.0

    rms_db = sum(rms_values) / len(rms_values)
    peak_db = max(peak_values)
    dynamic_range = peak_db - rms_db

    # Reject silence and clipping
    if rms_db < -45 or peak_db > -1:
        return 0.0

    # Speech tends to sit in a moderate loudness band with healthy dynamics
    loudness_score = max(0.0, min(1.0, (rms_db + 40) / 25))
    dynamic_score = max(0.0, min(1.0, dynamic_range / 18))
    return loudness_score * 0.55 + dynamic_score * 0.45


def _windows(duration: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + CLIP_MIN_SECONDS <= duration:
        clip_duration = min(CLIP_TARGET_SECONDS, duration - start)
        if clip_duration > CLIP_MAX_SECONDS:
            clip_duration = CLIP_MAX_SECONDS
        if clip_duration >= CLIP_MIN_SECONDS:
            windows.append((start, clip_duration))
        start += WINDOW_STEP_SECONDS
    if not windows and duration >= 8:
        windows.append((0.0, min(duration, CLIP_MAX_SECONDS)))
    return windows


def _overlap(a: SpeechClip, b: SpeechClip) -> bool:
    return not (a.end_sec <= b.start_sec or b.end_sec <= a.start_sec)


def extract_best_clips(
    wav_path: Path,
    output_dir: Path,
    *,
    target_count: int = 4,
) -> list[SpeechClip]:
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _duration_seconds(wav_path)

    ranked: list[SpeechClip] = []
    for start_sec, clip_duration in _windows(duration):
        score = _measure_segment(wav_path, start_sec, clip_duration)
        if score <= 0.05:
            continue
        ranked.append(
            SpeechClip(
                start_sec=start_sec,
                end_sec=start_sec + clip_duration,
                score=score,
                path=output_dir / f"candidate_{len(ranked)}.wav",
            )
        )

    ranked.sort(key=lambda clip: clip.score, reverse=True)

    selected: list[SpeechClip] = []
    for index, candidate in enumerate(ranked):
        if len(selected) >= target_count:
            break
        if any(_overlap(candidate, chosen) for chosen in selected):
            continue
        clip_path = output_dir / f"clip_{index}.wav"
        _extract_segment(
            wav_path,
            candidate.start_sec,
            candidate.end_sec - candidate.start_sec,
            clip_path,
        )
        selected.append(
            SpeechClip(
                start_sec=candidate.start_sec,
                end_sec=candidate.end_sec,
                score=candidate.score,
                path=clip_path,
            )
        )

    if not selected and duration > 0:
        fallback = output_dir / "clip_fallback.wav"
        _extract_segment(wav_path, 0.0, min(duration, CLIP_MAX_SECONDS), fallback)
        selected.append(
            SpeechClip(start_sec=0.0, end_sec=min(duration, CLIP_MAX_SECONDS), score=0.5, path=fallback)
        )

    return selected
