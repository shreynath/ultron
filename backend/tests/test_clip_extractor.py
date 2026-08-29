from pathlib import Path

from app.services.clip_extractor import extract_best_clips
from app.services.media_ingest import is_youtube_url


def test_is_youtube_url():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://example.com/audio.mp3")


def test_extract_best_clips_from_silence(tmp_path: Path):
    import shutil
    import subprocess
    import wave

    if not shutil.which("ffmpeg"):
        import pytest

        pytest.skip("ffmpeg not installed")

    wav_path = tmp_path / "input.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=25",
        "-ac",
        "1",
        "-ar",
        "48000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    clips = extract_best_clips(wav_path, tmp_path / "clips")
    assert len(clips) >= 1
    assert clips[0].path.exists()

    with wave.open(str(clips[0].path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
