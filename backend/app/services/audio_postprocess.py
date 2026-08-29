from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class AudioPostprocessError(Exception):
    pass


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise AudioPostprocessError("ffmpeg is required for voice post-processing")


def process_speech_wav(
    audio_bytes: bytes,
    *,
    speed: float = 1.0,
    bass_gain_db: float = 0.0,
) -> bytes:
    """Slow speech while preserving pitch and optionally boost sub-bass."""
    if abs(speed - 1.0) < 0.01 and bass_gain_db < 0.1:
        return audio_bytes

    _require_ffmpeg()
    speed = max(0.7, min(1.0, speed))

    filters: list[str] = []
    if abs(speed - 1.0) >= 0.01:
        filters.append(f"atempo={speed:.3f}")
    if bass_gain_db >= 0.1:
        filters.append(f"bass=g={bass_gain_db:.1f}:f=90")

    if not filters:
        return audio_bytes

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or "ffmpeg failed")[-500:]
            raise AudioPostprocessError(detail)
        return dst.read_bytes()
