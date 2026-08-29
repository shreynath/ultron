from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

YOUTUBE_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/",
    re.IGNORECASE,
)


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


def is_youtube_url(value: str) -> bool:
    return bool(YOUTUBE_PATTERN.search(value.strip()))


def download_youtube_audio(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{uuid.uuid4()}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wav_files:
        raise RuntimeError("yt-dlp did not produce a WAV file")
    return wav_files[0]


def convert_to_wav(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{uuid.uuid4()}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return destination


def ingest_media(
    *,
    url: str | None = None,
    upload_path: Path | None = None,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)

    if url:
        if is_youtube_url(url):
            if not ensure_yt_dlp():
                raise RuntimeError("yt-dlp is not installed. Run: brew install yt-dlp")
            return download_youtube_audio(url, work_dir)

        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            import httpx

            response = httpx.get(url, follow_redirects=True, timeout=120.0)
            response.raise_for_status()
            suffix = Path(parsed.path).suffix or ".bin"
            temp_path = work_dir / f"{uuid.uuid4()}{suffix}"
            temp_path.write_bytes(response.content)
            return convert_to_wav(temp_path, work_dir)

        raise ValueError("Unsupported URL format")

    if upload_path:
        if upload_path.suffix.lower() == ".wav":
            return upload_path
        if not ensure_ffmpeg():
            raise RuntimeError("ffmpeg is not installed. Run: brew install ffmpeg")
        return convert_to_wav(upload_path, work_dir)

    raise ValueError("Provide either a URL or an uploaded file")
