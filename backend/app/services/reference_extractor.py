from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from app.services.clip_extractor import (
    CLIP_MAX_SECONDS,
    CLIP_TARGET_SECONDS,
    WINDOW_STEP_SECONDS,
    _extract_segment,
    _measure_segment,
)
from app.services.speech_segmenter import MIN_UTTERANCE_SEC
from app.services.reference_analyzer import (
    ClipCandidate,
    estimate_deep_voice_score,
    is_likely_other_speaker,
    score_transcript_solo_speech,
)
from app.services.speech_segmenter import (
    MAX_UTTERANCE_SEC,
    SpeechSegment,
    detect_speech_segments,
    split_long_segment,
)

TranscribeFn = Callable[[Path], Awaitable[str]]

REFERENCE_CLIP_TARGET = 10
REFERENCE_CLIP_MIN = 3
LONG_SOURCE_SECONDS = 120.0


@dataclass
class ReferenceExtractionResult:
    samples: list[tuple[Path, str]]
    delivery: dict[str, float]
    rejected_count: int
    source_duration_sec: float


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


def _overlap(a: ClipCandidate, b: ClipCandidate) -> bool:
    return not (a.end_sec <= b.start_sec or b.end_sec <= a.start_sec)


def _sliding_window_candidates(wav_path: Path, duration: float) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    start = 0.0
    while start + MIN_UTTERANCE_SEC <= duration:
        clip_duration = min(CLIP_TARGET_SECONDS, duration - start)
        if clip_duration > CLIP_MAX_SECONDS:
            clip_duration = CLIP_MAX_SECONDS
        if clip_duration >= MIN_UTTERANCE_SEC:
            score = _measure_segment(wav_path, start, clip_duration)
            if score > 0.08:
                segments.append(SpeechSegment(start, start + clip_duration))
        start += WINDOW_STEP_SECONDS
    return segments


async def extract_reference_clips(
    wav_path: Path,
    output_dir: Path,
    transcribe: TranscribeFn,
    *,
    target_count: int = REFERENCE_CLIP_TARGET,
) -> ReferenceExtractionResult:
    """Extract solo-speech clips from long dialogue sources (e.g. movie scenes)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(wav_path)

    segments: list[SpeechSegment] = []
    for seg in detect_speech_segments(wav_path):
        segments.extend(split_long_segment(seg, MAX_UTTERANCE_SEC))
    segments.extend(_sliding_window_candidates(wav_path, duration))

    candidates: list[ClipCandidate] = []
    rejected = 0

    for index, seg in enumerate(segments):
        clip_duration = seg.duration
        deep_score = estimate_deep_voice_score(wav_path, seg.start_sec, clip_duration)
        temp_path = output_dir / f"probe_{index}.wav"
        _extract_segment(
            wav_path,
            seg.start_sec,
            clip_duration,
            temp_path,
            normalize=False,
            highpass_hz=0.0,
        )
        try:
            transcript = (await transcribe(temp_path)).strip()
        except Exception:
            rejected += 1
            temp_path.unlink(missing_ok=True)
            continue

        if not transcript:
            rejected += 1
            temp_path.unlink(missing_ok=True)
            continue

        if is_likely_other_speaker(transcript, deep_score):
            rejected += 1
            temp_path.unlink(missing_ok=True)
            continue

        solo_score = score_transcript_solo_speech(transcript)
        combined = solo_score * 0.55 + deep_score * 0.45
        candidates.append(
            ClipCandidate(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                score=combined,
                transcript=transcript,
                path=temp_path,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected: list[ClipCandidate] = []
    for cand in candidates:
        if len(selected) >= target_count:
            break
        if any(_overlap(cand, chosen) for chosen in selected):
            continue
        selected.append(cand)

    if len(selected) < REFERENCE_CLIP_MIN:
        raise ValueError(
            f"Only found {len(selected)} clear solo-speech clips. "
            "Try a source with longer Ultron-only monologue sections."
        )

    final_samples: list[tuple[Path, str]] = []
    delivery_stats: list[tuple[str, float, float]] = []
    for i, cand in enumerate(selected):
        out = output_dir / f"clip_{i}.wav"
        _extract_segment(
            wav_path,
            cand.start_sec,
            cand.duration,
            out,
            normalize=False,
            highpass_hz=0.0,
        )
        if cand.path and cand.path.exists() and cand.path != out:
            cand.path.unlink(missing_ok=True)
        deep = estimate_deep_voice_score(wav_path, cand.start_sec, cand.duration)
        delivery_stats.append((cand.transcript, cand.duration, deep))
        final_samples.append((out, cand.transcript))

    from app.services.reference_analyzer import recommend_delivery_from_clips

    delivery = recommend_delivery_from_clips(delivery_stats)

    return ReferenceExtractionResult(
        samples=final_samples,
        delivery=delivery,
        rejected_count=rejected,
        source_duration_sec=duration,
    )


def save_profile_delivery(data_dir: Path, profile_id: str, delivery: dict) -> None:
    path = data_dir / "profile_delivery.json"
    store: dict[str, dict] = {}
    if path.exists():
        store = json.loads(path.read_text())
    store[profile_id] = {
        **delivery,
        "voice_style": "reference",
    }
    path.write_text(json.dumps(store, indent=2))


def load_profile_delivery(data_dir: Path, profile_id: str) -> dict | None:
    path = data_dir / "profile_delivery.json"
    if not path.exists():
        return None
    store = json.loads(path.read_text())
    return store.get(profile_id)
