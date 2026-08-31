from __future__ import annotations

import numpy as np

from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float

_FRAME_MS = 20
_HALLUCINATION_GAP_MS = 2500
_HALLUCINATION_MIN_TAIL_MS = 1200


def _frame_rms(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    frame_len = int(sample_rate * _FRAME_MS / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return np.array([], dtype=np.float32), frame_len
    n_frames = len(audio) // frame_len
    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )
    return rms, frame_len


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20)


def repair_mlx_start_click(audio: np.ndarray) -> np.ndarray:
    """Fix a single-sample MLX offset at t=0 without fading the voice attack."""
    if audio.size < 3:
        return audio
    out = audio.copy()
    if abs(out[0]) > max(0.02, 2.5 * abs(out[1])):
        out[0] = out[1]
    return out


def strip_trailing_codec_only(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Remove MLX trailing hallucinations only — never touch the attack or phrase endings."""
    rms, frame_len = _frame_rms(audio, sample_rate)
    if rms.size == 0:
        return audio

    speech_threshold = _db_to_linear(-58.0)
    last_speech = 0
    for index in range(rms.size - 1, -1, -1):
        if rms[index] >= speech_threshold:
            last_speech = index
            break

    min_keep = min(
        len(audio),
        (last_speech + 1) * frame_len + int(sample_rate * _HALLUCINATION_MIN_TAIL_MS / 1000),
    )

    gap_frames = int(_HALLUCINATION_GAP_MS / _FRAME_MS)
    cut_sample: int | None = None
    consecutive_silence = 0
    seen_speech = False
    silence_threshold = _db_to_linear(-46.0)

    for index, energy in enumerate(rms):
        if energy >= silence_threshold:
            seen_speech = True
            consecutive_silence = 0
        elif seen_speech:
            consecutive_silence += 1
            if consecutive_silence >= gap_frames:
                candidate = index * frame_len - consecutive_silence * frame_len
                if candidate >= min_keep:
                    cut_sample = candidate
                break

    if cut_sample is None:
        return audio
    return audio[:cut_sample].copy()


def cleanup_mlx_wav(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        return audio_bytes

    audio, sample_rate = wav_bytes_to_mono_float(audio_bytes)
    if audio.size == 0:
        return audio_bytes

    audio = repair_mlx_start_click(audio)
    audio = strip_trailing_codec_only(audio, sample_rate)
    return mono_float_to_wav_bytes(audio, sample_rate)
