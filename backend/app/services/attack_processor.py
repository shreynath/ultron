from __future__ import annotations

import copy
from typing import Any

import numpy as np

from app.services.audio_postprocess import apply_presence_eq, process_speech_wav
from app.services.dramatic_planner import PhrasePlan
from app.services.living_delivery import shape_living_delivery
from app.services.reference_delivery import apply_delivery_tone, delivery_pause_ms
from app.services.sinister_texture import apply_chest_body
from app.services.theater_constants import ONSET_PAD_MS
from app.services.tts_cleanup import repair_mlx_start_click
from app.services.voice_effects import apply_effects_to_wav
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float

_ONSET_PAD_MS = ONSET_PAD_MS
_TAIL_PAD_MS = 180


def _prepend_onset_pad(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    pad = np.zeros(int(sample_rate * _ONSET_PAD_MS / 1000), dtype=np.float32)
    return np.concatenate([pad, audio])


def _append_tail_pad(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    pad = np.zeros(int(sample_rate * _TAIL_PAD_MS / 1000), dtype=np.float32)
    return np.concatenate([audio, pad])


def _fade_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_in_ms: int = 6,
    fade_out_ms: int = 14,
) -> np.ndarray:
    out = audio.copy()
    fade_in = min(len(out), int(sample_rate * fade_in_ms / 1000))
    fade_out = min(len(out), int(sample_rate * fade_out_ms / 1000))
    if fade_in > 1:
        out[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 1:
        out[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return out


def process_theater_audio(
    audio_bytes: bytes,
    effects_chain: list[dict[str, Any]],
    speed: float,
    bass_gain_db: float,
    *,
    pitch_semitones: float = 0.0,
    energy_gain_db: float = 0.0,
    bass_freq: float = 100.0,
    contour: str = "swell",
    intensity: float = 0.5,
    pitch_glide: float = 0.0,
    pad_onset: bool = True,
    pad_tail: bool = True,
    text: str = "",
    plans: list[PhrasePlan] | None = None,
    baseline_bass: float | None = None,
    profile_id: str | None = None,
    data_dir: str | None = None,
) -> bytes:
    """Clone-only Ultron delivery: shape tone and pauses, never hard-trim to film clips."""
    audio, sample_rate = wav_bytes_to_mono_float(audio_bytes)
    if audio.size == 0:
        return audio_bytes

    audio = repair_mlx_start_click(audio)
    audio = _fade_edges(audio, sample_rate, fade_in_ms=8, fade_out_ms=0)
    speed = max(0.88, min(1.12, speed))

    if pad_onset:
        audio = _prepend_onset_pad(audio, sample_rate)
    audio_bytes = mono_float_to_wav_bytes(audio, sample_rate)

    body = process_speech_wav(
        audio_bytes,
        speed=speed,
        bass_gain_db=min(5.2, max(4.0, bass_gain_db)),
        use_equalizer_bass=True,
        pitch_semitones=-1.2,
        energy_gain_db=energy_gain_db,
        bass_freq=min(84.0, max(72.0, bass_freq)),
    )
    chain = copy.deepcopy(effects_chain)
    for effect in chain:
        if effect.get("type") == "reverb":
            params = effect.setdefault("params", {})
            params["wet_level"] = min(0.08, float(params.get("wet_level", 0.08)))
            params["dry_level"] = max(0.92, float(params.get("dry_level", 0.92)))
    with_fx = apply_effects_to_wav(body, chain)

    audio, sample_rate = wav_bytes_to_mono_float(with_fx)
    if text.strip():
        audio = shape_living_delivery(
            audio,
            sample_rate,
            text,
            baseline_speed=speed,
            baseline_bass=baseline_bass if baseline_bass is not None else bass_gain_db,
            plans=plans,
            period_pause_ms=delivery_pause_ms(data_dir, profile_id),
        )
        audio = apply_chest_body(audio, sample_rate, intensity=0.38)
        audio = apply_delivery_tone(audio, sample_rate, data_dir, profile_id)

    audio = _fade_edges(audio, sample_rate, fade_in_ms=0, fade_out_ms=12)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.92:
        audio = (audio * (0.92 / peak)).astype(np.float32)
    if pad_tail:
        audio = _append_tail_pad(audio, sample_rate)
    return apply_presence_eq(mono_float_to_wav_bytes(audio, sample_rate), presence_gain_db=1.8)
