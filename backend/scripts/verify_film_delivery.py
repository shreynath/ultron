"""Offline line-by-line verification: warped clone → process_theater_audio → film metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.attack_processor import process_theater_audio
from app.services.film_identity import (
    METEOR_TEXT,
    compare_frames,
    covering_gold,
    ensure_gold_clips,
    gold_covers_text,
    isolate_ultron_voice,
    report_passes,
)
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float

NOVEL = "Hello from Ultron. This is a cloned voice speaking new words."
MEANT_PAIR = "I was meant to be new. I was meant to be beautiful."
AB_DIR = ROOT / "data" / "ultron_ref" / "ab"


def _warp_clone(target: np.ndarray, sample_rate: int) -> np.ndarray:
    warped_n = max(int(len(target) * 0.88), sample_rate // 8)
    return (
        np.interp(
            np.linspace(0.0, 1.0, warped_n, endpoint=False),
            np.linspace(0.0, 1.0, len(target), endpoint=False),
            target.astype(np.float64),
        ).astype(np.float32)
        * 0.65
    )


def _stamp(text: str, parts: list[np.ndarray], sample_rate: int) -> np.ndarray:
    raw = mono_float_to_wav_bytes(np.concatenate(parts), sample_rate)
    try:
        out_bytes = process_theater_audio(raw, [], speed=1.0, bass_gain_db=0.0, text=text)
    except ModuleNotFoundError as exc:
        if "pedalboard" in str(exc):
            print("SKIP: pedalboard not installed")
            raise SystemExit(2) from exc
        raise
    out, _ = wav_bytes_to_mono_float(out_bytes)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="write verify_report.json")
    args = parser.parse_args()

    clips = ensure_gold_clips()
    if not clips:
        print("no gold clips")
        return 1

    rows: list[dict] = []
    failed: list[str] = []

    for clip in clips:
        target = isolate_ultron_voice(clip.audio, clip.sample_rate)
        ours = _stamp(clip.text, [_warp_clone(target, clip.sample_rate)], clip.sample_rate)
        metrics = compare_frames(ours, target, clip.sample_rate)
        passed, reasons = report_passes(metrics)
        rows.append(
            {
                "id": clip.clip_id,
                "text": clip.text,
                "passed": passed,
                "duration_err_ms": round(metrics["duration_err_sec"] * 1000, 1),
                "f0_mae_hz": round(metrics["f0_mae_hz"], 2),
                "failures": reasons,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"{status:4} {clip.clip_id:40} dur={metrics['duration_err_sec']*1000:.0f}ms "
            f"f0={metrics['f0_mae_hz']:.1f}Hz"
        )
        if not passed:
            failed.append(f"{clip.clip_id}: " + "; ".join(reasons))

    ordered = covering_gold(MEANT_PAIR, clips)
    parts = [
        _warp_clone(isolate_ultron_voice(c.audio, c.sample_rate), c.sample_rate) for c in ordered
    ]
    film_pair = np.concatenate([isolate_ultron_voice(c.audio, c.sample_rate) for c in ordered])
    ours_pair = _stamp(MEANT_PAIR, parts, ordered[0].sample_rate)
    metrics = compare_frames(ours_pair, film_pair, ordered[0].sample_rate)
    passed, reasons = report_passes(metrics)
    rows.append({"id": "meant-pair", "text": MEANT_PAIR, "passed": passed, "failures": reasons})
    print(f"{'PASS' if passed else 'FAIL':4} meant-pair dur={metrics['duration_err_sec']*1000:.0f}ms")
    if not passed:
        failed.append("meant-pair: " + "; ".join(reasons))

    ordered_m = covering_gold(METEOR_TEXT, clips)
    parts_m = [
        _warp_clone(isolate_ultron_voice(c.audio, c.sample_rate), c.sample_rate) for c in ordered_m
    ]
    film_m = np.concatenate([isolate_ultron_voice(c.audio, c.sample_rate) for c in ordered_m])
    ours_m = _stamp(METEOR_TEXT, parts_m, ordered_m[0].sample_rate)
    metrics_m = compare_frames(ours_m, film_m, ordered_m[0].sample_rate)
    passed_m, reasons_m = report_passes(metrics_m)
    rows.append({"id": "full-meteor", "text": "...", "passed": passed_m, "failures": reasons_m})
    print(
        f"{'PASS' if passed_m else 'FAIL':4} full-meteor dur={metrics_m['duration_err_sec']*1000:.0f}ms"
    )
    if not passed_m:
        failed.append("full-meteor: " + "; ".join(reasons_m))

    assert not gold_covers_text(NOVEL, clips)
    clip = next(c for c in clips if c.clip_id == "i-was-meant-to-be-new")
    target = isolate_ultron_voice(clip.audio, clip.sample_rate)
    novel_out = _stamp(NOVEL, [_warp_clone(target, clip.sample_rate)], clip.sample_rate)
    novel_ok = len(novel_out) > clip.sample_rate // 4 and float(np.max(np.abs(novel_out))) > 0.02
    rows.append({"id": "novel", "text": NOVEL, "passed": novel_ok, "failures": []})
    print(f"{'PASS' if novel_ok else 'FAIL':4} novel-line (universal path)")
    if not novel_ok:
        failed.append("novel: output too short or silent")

    if args.json:
        AB_DIR.mkdir(parents=True, exist_ok=True)
        (AB_DIR / "verify_report.json").write_text(json.dumps({"clips": rows, "failed": failed}, indent=2))

    if failed:
        print("\nFAILED:")
        for line in failed:
            print(" ", line)
        return 1
    print(f"\nall {len(clips)} gold lines + meant-pair + meteor + novel OK (offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
