"""Generate a reproducible, clone-only delivery baseline.

The output directory contains independently generated WAV files, inputs, and
acoustic measurements. It never reads from or returns film clips.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.dramatic_planner import plan_theater_delivery
from app.services.f0 import track_f0, voiced_median
from app.services.film_identity import METEOR_TEXT
from app.services.living_delivery import detect_quiet_spans
from app.services.tts_cleanup import cleanup_mlx_wav
from app.services.ultron_tone import measure_ltas_db
from app.services.voice_delivery import VoiceDeliverySettings, deliver_speech
from app.services.voice_presets import ULTRON_REFERENCE_INSTRUCT
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import wav_bytes_to_mono_float

CASES = {
    "known_short": "I was meant to be new. I was meant to be beautiful.",
    "novel_short": "Hello from Ultron. This is a cloned voice speaking new words.",
    "meteor_monologue": METEOR_TEXT,
}
SEED = 7319


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _measure(audio_bytes: bytes) -> dict[str, object]:
    audio, sample_rate = wav_bytes_to_mono_float(audio_bytes)
    times, f0 = track_f0(audio, sample_rate)
    del times
    pauses = detect_quiet_spans(audio, sample_rate)
    return {
        "sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "sample_rate_hz": sample_rate,
        "duration_sec": round(len(audio) / max(1, sample_rate), 3),
        "peak": round(float(np.max(np.abs(audio))) if audio.size else 0.0, 5),
        "rms": round(float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0, 5),
        "voiced_f0_median_hz": round(voiced_median(f0), 2),
        "voiced_f0_p10_hz": round(float(np.percentile(f0[f0 > 1], 10)), 2)
        if np.any(f0 > 1)
        else 0.0,
        "voiced_f0_p90_hz": round(float(np.percentile(f0[f0 > 1], 90)), 2)
        if np.any(f0 > 1)
        else 0.0,
        "pause_spans_ms": [
            [round(start * 1000 / sample_rate), round(end * 1000 / sample_rate)]
            for start, end in pauses
        ],
        "ltas_db": [round(value, 2) for value in measure_ltas_db(audio, sample_rate)],
    }


async def _generate_variants(
    client: VoiceboxClient,
    text: str,
    profile_id: str,
    data_dir: str,
) -> list[tuple[str, bytes, dict[str, object]]]:
    plan = plan_theater_delivery(text)
    raw_kwargs = {
        "text": text,
        "profile_id": profile_id,
        "language": "en",
        "engine": "qwen",
        "effects_chain": [],
        "normalize": False,
        "seed": SEED,
    }
    raw = await client.generate(**raw_kwargs)
    instructed = await client.generate(
        **raw_kwargs, instruct=ULTRON_REFERENCE_INSTRUCT
    )
    theater = await deliver_speech(
        client,
        text=text,
        profile_id=profile_id,
        language="en",
        engine="qwen",
        delivery=VoiceDeliverySettings(voice_style="theater_auto"),
        data_dir=data_dir,
    )
    return [
        ("raw", raw, {**raw_kwargs, "instruct": None, "processing": []}),
        (
            "native_instruction",
            instructed,
            {
                **raw_kwargs,
                "instruct": ULTRON_REFERENCE_INSTRUCT,
                "processing": [],
            },
        ),
        (
            "minimal_cleanup",
            cleanup_mlx_wav(instructed),
            {
                **raw_kwargs,
                "instruct": ULTRON_REFERENCE_INSTRUCT,
                "processing": ["repair_mlx_start_click", "strip_trailing_codec_only"],
            },
        ),
        (
            "current_theater",
            theater,
            {
                "voice_style": "theater_auto",
                "plan": {
                    "text": plan.text,
                    "speed": plan.speed,
                    "bass_boost": plan.bass_boost,
                    "instruct": ULTRON_REFERENCE_INSTRUCT,
                },
                "processing": [
                    "cleanup_mlx_wav",
                    "process_speech_wav: atempo, -1.2 semitone, EQ bass",
                    "reverb",
                    "shape_living_delivery: pauses and phrase energy",
                    "apply_chest_body",
                    "apply_delivery_tone",
                    "presence EQ",
                ],
            },
        ),
    ]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Existing Voicebox profile id")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / "data" / "evaluations" / f"baseline-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "profile_id": args.profile,
        "machine": platform.platform(),
        "variants": {},
    }

    client = VoiceboxClient()
    for case_name, text in CASES.items():
        case_dir = output_dir / case_name
        case_dir.mkdir()
        rows = []
        for variant, audio_bytes, request in await _generate_variants(
            client, text, args.profile, str(ROOT / "data")
        ):
            wav_path = case_dir / f"{variant}.wav"
            wav_path.write_bytes(audio_bytes)
            rows.append(
                {
                    "variant": variant,
                    "wav": str(wav_path.relative_to(output_dir)),
                    "request": request,
                    "features": _measure(audio_bytes),
                }
            )
            print(f"{case_name}: {variant} -> {wav_path}", flush=True)
        manifest["variants"][case_name] = {"text": text, "outputs": rows}

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
