from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


class AudioPostprocessError(Exception):
    pass


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise AudioPostprocessError("ffmpeg is required for voice post-processing")


def _write_silence_wav(path: Path, duration_ms: int, sample_rate: int = 48000) -> None:
    duration_sec = max(0.01, duration_ms / 1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        f"{duration_sec:.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioPostprocessError((result.stderr or "silence gen failed")[-300:])


def _atempo_chain(speed: float) -> list[str]:
    """ffmpeg atempo accepts 0.5–2.0; chain filters for values outside that."""
    filters: list[str] = []
    remaining = speed
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    if abs(remaining - 1.0) >= 0.008:
        filters.append(f"atempo={remaining:.4f}")
    return filters


def process_speech_wav(
    audio_bytes: bytes,
    *,
    speed: float = 1.0,
    bass_gain_db: float = 0.0,
    use_equalizer_bass: bool = False,
    pitch_semitones: float = 0.0,
    energy_gain_db: float = 0.0,
    bass_freq: float = 100.0,
) -> bytes:
    """Shape speech tempo, pitch, energy, and sub-bass after cinematic effects."""
    if (
        abs(speed - 1.0) < 0.01
        and bass_gain_db < 0.1
        and abs(pitch_semitones) < 0.05
        and abs(energy_gain_db) < 0.1
    ):
        return audio_bytes

    _require_ffmpeg()
    speed = max(0.55, min(1.12, speed))
    bass_gain_db = min(12.0, max(0.0, bass_gain_db))
    pitch_semitones = max(-3.5, min(3.5, pitch_semitones))
    energy_gain_db = max(-4.0, min(5.0, energy_gain_db))
    bass_freq = max(70.0, min(160.0, bass_freq))

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(audio_bytes)
        with wave.open(str(src), "rb") as wav_file:
            sample_rate = wav_file.getframerate()

        filters: list[str] = []
        atempo_total = speed
        if abs(pitch_semitones) >= 0.05:
            factor = 2 ** (pitch_semitones / 12.0)
            new_rate = max(8000, int(round(sample_rate * factor)))
            filters.append(f"asetrate={new_rate}")
            filters.append(f"aresample={sample_rate}")
            atempo_total = speed / factor
        filters.extend(_atempo_chain(atempo_total))
        if bass_gain_db >= 0.1:
            if use_equalizer_bass:
                filters.append(
                    f"equalizer=f={bass_freq:.0f}:width_type=h:width=50:g={bass_gain_db:.1f}"
                )
            else:
                filters.append(f"bass=g={bass_gain_db:.1f}:f={bass_freq:.0f}")
        if abs(energy_gain_db) >= 0.1:
            filters.append(f"volume={energy_gain_db:.2f}dB")

        if not filters:
            return audio_bytes

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
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or "ffmpeg failed")[-500:]
            raise AudioPostprocessError(detail)
        return dst.read_bytes()


def _mono_pcm(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with wave.open(tmp.name, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioPostprocessError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _band_ratio_db(
    audio: np.ndarray,
    sample_rate: int,
    lo: float,
    hi: float,
    ref_lo: float,
    ref_hi: float,
) -> float:
    frame = int(sample_rate * 0.04)
    if audio.size < frame * 4:
        return 0.0
    hop = max(1, frame // 2)
    win = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    ratios: list[float] = []
    box_mask = (freqs >= lo) & (freqs < hi)
    ref_mask = (freqs >= ref_lo) & (freqs < ref_hi)
    for index in range(0, len(audio) - frame, hop):
        window = audio[index : index + frame] * win
        if float(np.sqrt(np.mean(window**2))) < 0.02:
            continue
        power = np.abs(np.fft.rfft(window)) ** 2
        box = float(np.mean(power[box_mask]))
        ref = float(np.mean(power[ref_mask]))
        ratios.append(10.0 * np.log10((box + 1e-18) / (ref + 1e-18)))
    if not ratios:
        return 0.0
    return float(np.median(np.array(ratios)))


def apply_ultron_timbre_eq(audio_bytes: bytes) -> bytes:
    """Cut TTS box only when this take is boxier than measured Ultron (~+9.5 dB at 250–400 vs 500–1k)."""
    _require_ffmpeg()
    audio, sample_rate = _mono_pcm(audio_bytes)
    box = _band_ratio_db(audio, sample_rate, 250.0, 400.0, 500.0, 1000.0)
    excess = max(0.0, box - 9.5)
    cut_320 = min(12.0, excess)
    cut_450 = min(9.0, excess * 0.75)
    cut_210 = min(3.5, excess * 0.3)
    filters = (
        "highpass=f=48,"
        "equalizer=f=68:width_type=h:width=24:g=7.5,"
        "equalizer=f=108:width_type=h:width=40:g=1.2,"
        "equalizer=f=165:width_type=h:width=50:g=-3.2,"
        "equalizer=f=240:width_type=h:width=80:g=-2.8,"
        "equalizer=f=1600:width_type=h:width=220:g=2.4,"
        f"equalizer=f=210:width_type=h:width=70:g={-cut_210:.1f},"
        f"equalizer=f=320:width_type=h:width=140:g={-cut_320:.1f},"
        f"equalizer=f=450:width_type=h:width=90:g={-cut_450:.1f},"
        "equalizer=f=3100:width_type=h:width=420:g=2.2,"
        "equalizer=f=6500:width_type=h:width=1400:g=-2.0,"
        "volume=2.6dB,alimiter=limit=0.91"
    )
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
            filters,
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioPostprocessError((result.stderr or "ultron eq failed")[-500:])
        return dst.read_bytes()


def apply_presence_eq(audio_bytes: bytes, *, presence_gain_db: float = 1.8) -> bytes:
    """Lift consonant presence for articulation — no pitch or tempo change."""
    if presence_gain_db < 0.1:
        return audio_bytes

    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(audio_bytes)
        with wave.open(str(src), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            f"equalizer=f=2800:width_type=h:width=2:g={presence_gain_db:.1f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioPostprocessError((result.stderr or "presence eq failed")[-500:])
        return dst.read_bytes()


def concat_phrase_audio(segments: list[tuple[bytes, int]]) -> bytes:
    """Concatenate phrase WAVs with trailing pauses (ms) after each segment."""
    if not segments:
        raise AudioPostprocessError("No audio segments to concatenate")
    if len(segments) == 1:
        return segments[0][0]

    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        list_file = tmp_path / "concat.txt"
        part_paths: list[Path] = []

        for index, (audio_bytes, pause_ms) in enumerate(segments):
            part = tmp_path / f"part_{index:03d}.wav"
            part.write_bytes(audio_bytes)
            part_paths.append(part)
            if pause_ms > 0:
                silence = tmp_path / f"pause_{index:03d}.wav"
                _write_silence_wav(silence, pause_ms)
                part_paths.append(silence)

        list_file.write_text(
            "".join(f"file '{path.name}'\n" for path in part_paths),
            encoding="utf-8",
        )

        out = tmp_path / "merged.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
        if result.returncode != 0:
            raise AudioPostprocessError((result.stderr or "concat failed")[-500:])
        return out.read_bytes()


def wav_duration_ms(audio_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with wave.open(tmp.name, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return int(1000 * frames / rate)
