from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np

from app.services.audio_postprocess import AudioPostprocessError


def wav_bytes_to_mono_float(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with wave.open(tmp.name, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioPostprocessError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def mono_float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return Path(tmp.name).read_bytes()
