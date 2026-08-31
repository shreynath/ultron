"""Generate every gold Ultron line and fail unless signatures match the film."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.film_identity import (
    METEOR_TEXT,
    compare_frames,
    covering_gold,
    ensure_gold_clips,
    extract_gold_from_source,
    frame_report,
    isolate_ultron_voice,
    polish_gold_clip,
    report_passes,
)
from app.services.voice_delivery import VoiceDeliverySettings, deliver_speech
from app.services.voicebox_client import VoiceboxClient
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float

PROFILE = "706f4fb2-fcc6-476c-9557-43d25bcb8676"
MEANT_PAIR = "I was meant to be new. I was meant to be beautiful."
NOVEL = "Hello from Ultron. This is a cloned voice speaking new words."
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data" / "ultron_ref" / "gold_ours"
AB_DIR = ROOT / "data" / "ultron_ref" / "ab"


async def _generate(client: VoiceboxClient, text: str) -> bytes:
    return await deliver_speech(
        client,
        text=text,
        profile_id=PROFILE,
        delivery=VoiceDeliverySettings(voice_style="theater_auto"),
        data_dir=str(DATA_DIR),
    )


def _report_row(report) -> dict:
    return {
        "id": report.clip_id,
        "text": report.text,
        "passed": report.passed,
        "duration_err_ms": round(report.duration_err_sec * 1000, 1),
        "f0_mae_hz": round(report.f0_mae_hz, 2),
        "ltas_mae_db": round(report.ltas_mae_db, 2),
        "rms_corr": round(report.rms_corr, 3),
        "bass_err_db": round(report.bass_err_db, 2),
        "voiced_frac_err": round(report.voiced_frac_err, 3),
        "n_frames": report.n_frames,
        "failures": report.failures,
    }


def _resample_to(audio: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return audio.astype(np.float32)
    n = max(8, int(round(len(audio) * target_rate / sample_rate)))
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(audio), endpoint=False),
        audio.astype(np.float64),
    ).astype(np.float32)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    clips = ensure_gold_clips()
    if not clips:
        extract_gold_from_source()
        clips = ensure_gold_clips()
    if not clips:
        print("no gold clips — missing source.wav")
        return 1
    print(f"{len(clips)} gold lines", flush=True)
    if args.extract_only:
        for clip in clips:
            print(f"  {clip.duration_sec:5.2f}s  {clip.clip_id}  {clip.text}", flush=True)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AB_DIR.mkdir(parents=True, exist_ok=True)
    client = VoiceboxClient()
    rows = []
    failed = []

    for clip in clips:
        print(f"generate {clip.clip_id} ...", flush=True)
        try:
            audio_bytes = await _generate(client, clip.text)
        except Exception as exc:
            failed.append(f"{clip.clip_id}: generate failed ({exc})")
            continue
        ours, sample_rate = wav_bytes_to_mono_float(audio_bytes)
        (OUT_DIR / f"{clip.clip_id}.wav").write_bytes(audio_bytes)
        report = frame_report(clip, ours, sample_rate)
        rows.append(_report_row(report))
        status = "PASS" if report.passed else "FAIL"
        print(
            f"  {status}  dur={report.duration_err_sec*1000:.0f}ms  "
            f"f0={report.f0_mae_hz:.1f}Hz  eq={report.ltas_mae_db:.1f}dB  "
            f"rms={report.rms_corr:.2f}  bass={report.bass_err_db:.1f}dB  "
            f"voice={report.voiced_frac_err:.2f}  {clip.text}",
            flush=True,
        )
        if not report.passed:
            failed.append(f"{clip.clip_id}: " + "; ".join(report.failures))

    print("generate full meteor speech ...", flush=True)
    meteor_bytes = await _generate(client, METEOR_TEXT)
    meteor_path = ROOT / "data" / "ultron_ref" / "ours_meteor.wav"
    meteor_path.write_bytes(meteor_bytes)
    ours_meteor, meteor_sr = wav_bytes_to_mono_float(meteor_bytes)

    ordered = covering_gold(METEOR_TEXT, clips)
    film_meteor = np.concatenate(
        [
            isolate_ultron_voice(
                _resample_to(clip.audio, clip.sample_rate, meteor_sr), meteor_sr
            )
            for clip in ordered
        ]
    )
    (ROOT / "data" / "ultron_ref" / "film_meteor.wav").write_bytes(
        mono_float_to_wav_bytes(film_meteor, meteor_sr)
    )
    metrics = compare_frames(ours_meteor, film_meteor, meteor_sr)
    passed, reasons = report_passes(metrics)
    rows.append(
        {
            "id": "full-meteor",
            "text": METEOR_TEXT,
            "passed": passed,
            "duration_err_ms": round(metrics["duration_err_sec"] * 1000, 1),
            "f0_mae_hz": round(metrics["f0_mae_hz"], 2),
            "ltas_mae_db": round(metrics["ltas_mae_db"], 2),
            "rms_corr": round(metrics["rms_corr"], 3),
            "bass_err_db": round(metrics["bass_err_db"], 2),
            "voiced_frac_err": round(metrics["voiced_frac_err"], 3),
            "n_frames": int(metrics["n_frames"]),
            "failures": reasons,
        }
    )
    print(
        f"  {'PASS' if passed else 'FAIL'} meteor  dur={metrics['duration_err_sec']*1000:.0f}ms  "
        f"f0={metrics['f0_mae_hz']:.1f}Hz  eq={metrics['ltas_mae_db']:.1f}dB  "
        f"rms={metrics['rms_corr']:.2f}  bass={metrics['bass_err_db']:.1f}dB"
    )
    if not passed:
        failed.append("full-meteor: " + "; ".join(reasons))

    print(f"generate meant pair: {MEANT_PAIR!r} ...", flush=True)
    meant_bytes = await _generate(client, MEANT_PAIR)
    (OUT_DIR / "meant-pair.wav").write_bytes(meant_bytes)
    ours_meant, meant_sr = wav_bytes_to_mono_float(meant_bytes)
    ordered_meant = covering_gold(MEANT_PAIR, clips)
    film_meant = np.concatenate(
        [
            isolate_ultron_voice(
                _resample_to(clip.audio, clip.sample_rate, meant_sr), meant_sr
            )
            for clip in ordered_meant
        ]
    )
    metrics_meant = compare_frames(ours_meant, film_meant, meant_sr)
    passed_meant, reasons_meant = report_passes(metrics_meant)
    rows.append(
        {
            "id": "meant-pair",
            "text": MEANT_PAIR,
            "passed": passed_meant,
            "duration_err_ms": round(metrics_meant["duration_err_sec"] * 1000, 1),
            "f0_mae_hz": round(metrics_meant["f0_mae_hz"], 2),
            "ltas_mae_db": round(metrics_meant["ltas_mae_db"], 2),
            "rms_corr": round(metrics_meant["rms_corr"], 3),
            "bass_err_db": round(metrics_meant["bass_err_db"], 2),
            "voiced_frac_err": round(metrics_meant["voiced_frac_err"], 3),
            "n_frames": int(metrics_meant["n_frames"]),
            "failures": reasons_meant,
        }
    )
    print(
        f"  {'PASS' if passed_meant else 'FAIL'} meant-pair  "
        f"dur={metrics_meant['duration_err_sec']*1000:.0f}ms  "
        f"f0={metrics_meant['f0_mae_hz']:.1f}Hz  eq={metrics_meant['ltas_mae_db']:.1f}dB"
    )
    if not passed_meant:
        failed.append("meant-pair: " + "; ".join(reasons_meant))

    print(f"generate novel line: {NOVEL!r} ...", flush=True)
    novel_bytes = await _generate(client, NOVEL)
    (OUT_DIR / "novel.wav").write_bytes(novel_bytes)
    novel_audio, novel_sr = wav_bytes_to_mono_float(novel_bytes)
    novel_ok = (
        novel_audio.size > novel_sr // 4
        and float(np.max(np.abs(novel_audio))) > 0.02
        and float(np.sqrt(np.mean(novel_audio**2))) > 0.01
    )
    rows.append({"id": "novel", "text": NOVEL, "passed": novel_ok, "failures": []})
    print(f"  {'PASS' if novel_ok else 'FAIL'} novel  len={len(novel_audio)/novel_sr:.2f}s")
    if not novel_ok:
        failed.append("novel: output too short or silent")

    payload = {"clips": rows, "failed": failed}
    (AB_DIR / "identity_report.json").write_text(json.dumps(payload, indent=2))
    if failed:
        print("FAILED:")
        for line in failed:
            print(" ", line)
        return 1
    print(f"all {len(clips)} lines + meteor + meant-pair + novel matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
