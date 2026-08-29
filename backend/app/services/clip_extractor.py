from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import webrtcvad


@dataclass
class SpeechClip:
    start_sec: float
    end_sec: float
    score: float
    path: Path


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Expected mono 16-bit PCM WAV")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV")
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    return frames, sample_rate


def _write_wav(path: Path, frames: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def _frame_generator(audio: bytes, sample_rate: int, frame_duration_ms: int):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset : offset + n]
        offset += n


def _score_segment(audio: bytes) -> float:
    if not audio:
        return 0.0
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    if peak < 500:
        return 0.0
    return min(rms / 5000.0, 1.0)


def extract_best_clips(
    wav_path: Path,
    output_dir: Path,
    *,
    target_count: int = 3,
    min_duration: float = 10.0,
    max_duration: float = 25.0,
    frame_duration_ms: int = 30,
) -> list[SpeechClip]:
    audio, sample_rate = _read_wav(wav_path)
    if sample_rate not in {8000, 16000, 32000, 48000}:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")

    vad = webrtcvad.Vad(2)
    frames = list(_frame_generator(audio, sample_rate, frame_duration_ms))
    voiced = [vad.is_speech(frame, sample_rate) for frame in frames]

    segments: list[tuple[int, int]] = []
    start = None
    for index, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = index
        elif not is_voiced and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(voiced)))

    frame_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    candidates: list[SpeechClip] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg_start, seg_end in segments:
        duration = (seg_end - seg_start) * (frame_duration_ms / 1000.0)
        if duration < min_duration:
            continue

        seg_start_idx = seg_start * frame_bytes
        seg_end_idx = min(seg_end * frame_bytes, len(audio))
        segment_audio = audio[seg_start_idx:seg_end_idx]

        if duration > max_duration:
            max_bytes = int(max_duration * sample_rate * 2)
            segment_audio = segment_audio[:max_bytes]
            duration = max_duration

        score = _score_segment(segment_audio)
        if score <= 0:
            continue

        clip_path = output_dir / f"clip_{len(candidates)}.wav"
        _write_wav(clip_path, segment_audio, sample_rate)
        candidates.append(
            SpeechClip(
                start_sec=seg_start * frame_duration_ms / 1000.0,
                end_sec=seg_end * frame_duration_ms / 1000.0,
                score=score,
                path=clip_path,
            )
        )

    if not candidates:
        fallback_path = output_dir / "clip_fallback.wav"
        max_bytes = int(max_duration * sample_rate * 2)
        fallback_audio = audio[:max_bytes]
        _write_wav(fallback_path, fallback_audio, sample_rate)
        candidates.append(
            SpeechClip(start_sec=0.0, end_sec=max_duration, score=0.5, path=fallback_path)
        )

    candidates.sort(key=lambda clip: clip.score, reverse=True)
    return candidates[:target_count]
