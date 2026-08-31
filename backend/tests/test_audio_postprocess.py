import shutil
import subprocess
from pathlib import Path

import pytest

from app.services.audio_postprocess import process_speech_wav


@pytest.fixture
def sample_wav(tmp_path: Path) -> bytes:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    wav_path = tmp_path / "sample.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=1",
        "-ac",
        "1",
        "-ar",
        "48000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path.read_bytes()


def test_process_speech_wav_slows_audio(sample_wav: bytes):
    slowed = process_speech_wav(sample_wav, speed=0.8, bass_gain_db=4.0)
    assert len(slowed) > len(sample_wav) * 0.9


def test_process_speech_wav_noop(sample_wav: bytes):
    assert process_speech_wav(sample_wav, speed=1.0, bass_gain_db=0.0) == sample_wav


def test_process_speech_wav_pitch_keeps_approx_duration(sample_wav: bytes):
    from app.services.audio_postprocess import wav_duration_ms

    shifted = process_speech_wav(sample_wav, speed=1.0, bass_gain_db=0.0, pitch_semitones=1.2)
    original_ms = wav_duration_ms(sample_wav)
    shifted_ms = wav_duration_ms(shifted)
    assert abs(shifted_ms - original_ms) / original_ms < 0.08
