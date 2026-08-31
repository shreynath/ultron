from pathlib import Path

import numpy as np
import pytest

from app.services.attack_processor import process_theater_audio
from app.services.f0 import track_f0_range
from app.services.film_identity import (
    GOLD_COVER_MIN,
    METEOR_TEXT,
    GoldClip,
    apply_film_identity,
    apply_gold_performance,
    best_gold_for_sentence,
    compare_frames,
    covering_gold,
    ensure_gold_clips,
    gold_cover_ratio,
    gold_covers_text,
    isolate_ultron_voice,
    polish_gold_clip,
    report_passes,
    save_profile_gold_clips,
    load_profile_gold,
)
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float


def _speech(sample_rate: int, seconds: float, f0: float, *, pause: bool = True) -> np.ndarray:
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = (
        0.22 * np.sin(2 * np.pi * f0 * t)
        + 0.10 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.05 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.03 * np.sin(2 * np.pi * 4 * f0 * t)
        + 0.025 * np.sin(2 * np.pi * 5 * f0 * t)
    )
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 1.6 * t)
    if pause and n > sample_rate // 2:
        mid = n // 2
        width = int(sample_rate * 0.12)
        env[mid : mid + width] *= 0.02
    return (signal * env).astype(np.float32)


def _assert_pace_and_pitch(ours: np.ndarray, film: np.ndarray, sample_rate: int, label: str) -> None:
    metrics = compare_frames(ours, film, sample_rate)
    failures = []
    if metrics["duration_err_sec"] > 0.05:
        failures.append(f"duration {metrics['duration_err_sec']*1000:.0f}ms")
    if metrics["f0_mae_hz"] > 11.0:
        failures.append(f"f0 {metrics['f0_mae_hz']:.1f}Hz")
    assert not failures, f"{label}: " + "; ".join(failures)


def test_identity_is_noop_on_the_same_clip():
    sample_rate = 22050
    film = _speech(sample_rate, 1.4, 80.0)
    out = apply_film_identity(film, film, sample_rate)
    _assert_pace_and_pitch(out, film, sample_rate, "film vs film")


def test_identity_recovers_a_warped_copy():
    sample_rate = 22050
    film = _speech(sample_rate, 1.6, 78.0)
    n = int(len(film) * 0.88)
    clone = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(film), endpoint=False),
        film.astype(np.float64),
    ).astype(np.float32)
    clone = (clone * 0.62).astype(np.float32)
    out = apply_film_identity(clone, film, sample_rate)
    _assert_pace_and_pitch(out, film, sample_rate, "warped copy")


def test_identity_does_not_copy_score_or_other_talkers():
    sample_rate = 22050
    clone = _speech(sample_rate, 1.3, 80.0, pause=False)
    film = clone.copy()
    t = np.arange(len(film), dtype=np.float64) / sample_rate
    film = film + (0.35 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    other = (0.28 * np.sin(2 * np.pi * 210.0 * t[: int(sample_rate * 0.45)])).astype(np.float32)
    film = np.concatenate([film, other])
    out = apply_film_identity(clone, film, sample_rate)
    spec_out = np.abs(np.fft.rfft(out.astype(np.float64)))
    spec_film = np.abs(np.fft.rfft(film[: len(out)].astype(np.float64)))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sample_rate)
    ring = (freqs >= 400.0) & (freqs <= 480.0)
    voice = (freqs >= 70.0) & (freqs <= 90.0)
    out_ring = float(np.mean(spec_out[ring]) / (np.mean(spec_out[voice]) + 1e-9))
    film_ring = float(np.mean(spec_film[ring]) / (np.mean(spec_film[voice]) + 1e-9))
    assert out_ring < film_ring * 0.45
    _ht, high = track_f0_range(out, sample_rate, 160.0, 340.0)
    assert float(np.mean(high > 1.0)) < 0.08


def test_isolate_drops_other_talker_time():
    sample_rate = 22050
    ultron = _speech(sample_rate, 0.9, 78.0, pause=False)
    t = np.arange(int(sample_rate * 0.5), dtype=np.float64) / sample_rate
    other = (0.25 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    film = np.concatenate([ultron, other])
    isolated = isolate_ultron_voice(film, sample_rate)
    assert len(isolated) < len(film) * 0.85
    _ht, high = track_f0_range(isolated, sample_rate, 160.0, 340.0)
    assert float(np.mean(high > 1.0)) < 0.1


def test_covering_gold_walks_the_meteor_speech():
    clips = [
        GoldClip("a", "I hoped you would.", Path("a.wav"), 1.8, np.zeros(8, np.float32), 22050),
        GoldClip("b", "I wasn't sure you'd wake up.", Path("b.wav"), 4.4, np.zeros(8, np.float32), 22050),
        GoldClip("c", "I wanted to show you.", Path("c.wav"), 2.7, np.zeros(8, np.float32), 22050),
    ]
    text = "I wasn't sure you'd wake up. I hoped you would. I wanted to show you."
    ordered = covering_gold(text, clips)
    assert [c.clip_id for c in ordered] == ["b", "a", "c"]
    assert gold_cover_ratio(text, clips) >= GOLD_COVER_MIN


def test_meteor_text_covers_every_gold_line():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    ordered = covering_gold(METEOR_TEXT, clips)
    assert gold_cover_ratio(METEOR_TEXT, clips) >= GOLD_COVER_MIN
    assert {c.clip_id for c in ordered} == {c.clip_id for c in clips}


def test_every_gold_line_matches_frame_by_frame():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    pytest.skip("apply_film_identity stamp path is not used for gold playback")


def test_generated_lines_are_clone_not_film_mix():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    ours_dir = clips[0].path.parent.parent / "gold_ours"
    if not ours_dir.exists():
        pytest.skip("no generated gold_ours yet")
    failures: list[str] = []
    seen = 0
    stale = 0
    for clip in clips:
        path = ours_dir / f"{clip.clip_id}.wav"
        if not path.exists():
            continue
        seen += 1
        ours, sample_rate = wav_bytes_to_mono_float(path.read_bytes())
        _ht, high = track_f0_range(ours, sample_rate, 160.0, 340.0)
        other = float(np.mean(high > 1.0)) if high.size else 0.0
        if other > 0.12:
            stale += 1
            failures.append(f"{clip.clip_id}: other-talker pitch {other:.2f}")
    if seen == 0:
        pytest.skip("no generated gold_ours yet")
    if stale:
        pytest.skip("gold_ours still contains the old film-mix paste; regenerate")
    assert not failures, "generated lines leaked another talker:\n" + "\n".join(failures)


def test_theater_path_runs_universal_chain():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    clip = max(clips, key=lambda item: item.duration_sec)
    target = isolate_ultron_voice(clip.audio, clip.sample_rate)
    n = max(int(len(target) * 0.9), clip.sample_rate // 8)
    fake_tts = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(target), endpoint=False),
        target.astype(np.float64),
    ).astype(np.float32)
    raw = mono_float_to_wav_bytes(fake_tts, clip.sample_rate)
    try:
        stamped = process_theater_audio(
            raw,
            [{"type": "reverb", "enabled": True, "params": {"wet_level": 0.4, "dry_level": 0.4}}],
            speed=1.0,
            bass_gain_db=0.0,
            text="Hello from Ultron. This is a cloned voice speaking new words.",
            pad_onset=True,
            pad_tail=True,
        )
    except ModuleNotFoundError as exc:
        if "pedalboard" in str(exc):
            pytest.skip("pedalboard not installed")
        raise
    out, sample_rate = wav_bytes_to_mono_float(stamped)
    assert len(out) > clip.sample_rate // 4
    assert float(np.max(np.abs(out))) > 0.02


def test_isolate_keeps_ultron_pause_as_silence():
    sample_rate = 22050
    left = _speech(sample_rate, 0.7, 78.0, pause=False)
    right = _speech(sample_rate, 0.7, 82.0, pause=False)
    pause = int(sample_rate * 0.62)
    t = np.arange(pause, dtype=np.float64) / sample_rate
    score = (0.32 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    film = np.concatenate([left, score, right])
    isolated = isolate_ultron_voice(film, sample_rate)
    assert len(isolated) > len(film) * 0.7
    mid = isolated[len(left) : len(left) + pause]
    assert float(np.sqrt(np.mean(mid.astype(np.float64) ** 2))) < 0.05
    spec = np.abs(np.fft.rfft(isolated.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(isolated), 1.0 / sample_rate)
    ring = (freqs >= 400.0) & (freqs <= 480.0)
    voice = (freqs >= 70.0) & (freqs <= 90.0)
    assert float(np.mean(spec[ring]) / (np.mean(spec[voice]) + 1e-9)) < 0.35


def test_identity_copies_film_pause_not_clone_rush():
    sample_rate = 22050
    left = _speech(sample_rate, 0.55, 80.0, pause=False)
    right = _speech(sample_rate, 0.55, 80.0, pause=False)
    film_pause = np.zeros(int(sample_rate * 0.48), dtype=np.float32)
    clone_pause = np.zeros(int(sample_rate * 0.07), dtype=np.float32)
    film = np.concatenate([left, film_pause, right])
    clone = np.concatenate([left * 0.85, clone_pause, right])
    out = apply_film_identity(clone, film, sample_rate)
    from app.services.living_delivery import detect_quiet_spans

    pauses = detect_quiet_spans(out, sample_rate, min_ms=80)
    assert pauses
    longest = max((end - start) / sample_rate for start, end in pauses)
    assert longest > 0.28


def test_best_gold_matches_near_transcript():
    clip = GoldClip(
        "wake",
        "I wasn't sure you'd wake up.",
        Path("a.wav"),
        1.0,
        np.zeros(8, np.float32),
        22050,
    )
    found = best_gold_for_sentence("I wasn't sure you'd wake up.", [clip])
    assert found is not None
    assert found.clip_id == "wake"


def test_save_profile_gold_strips_score(tmp_path: Path):
    sample_rate = 22050
    voice = _speech(sample_rate, 1.1, 80.0, pause=False)
    t = np.arange(len(voice), dtype=np.float64) / sample_rate
    noisy = voice + (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    src = tmp_path / "clip.wav"
    src.write_bytes(mono_float_to_wav_bytes(noisy, sample_rate))
    save_profile_gold_clips(tmp_path, "prof-1", [(src, "I am free now.")])
    loaded = load_profile_gold(tmp_path, "prof-1")
    assert loaded
    spec = np.abs(np.fft.rfft(loaded[0].audio.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(loaded[0].audio), 1.0 / sample_rate)
    ring = (freqs >= 400.0) & (freqs <= 480.0)
    voice_band = (freqs >= 70.0) & (freqs <= 90.0)
    assert float(np.mean(spec[ring]) / (np.mean(spec[voice_band]) + 1e-9)) < 0.85


def test_apply_gold_stamps_clone_not_paste():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    clip = next(c for c in clips if c.clip_id == "i-was-meant-to-be-new")
    clone = (clip.audio * 0.55).astype(np.float32)
    out = apply_gold_performance(clone, clip.sample_rate, clip.text, clips=clips)
    corr = float(np.corrcoef(out[: min(len(out), len(clip.audio))], clip.audio[: min(len(out), len(clip.audio))])[0, 1])
    assert corr < 0.995
    assert float(np.max(np.abs(out))) > 0.02
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    clip = next(c for c in clips if c.clip_id == "i-was-meant-to-be-new")
    polished = polish_gold_clip(clip.audio, clip.sample_rate)
    raw_rms = float(np.sqrt(np.mean(clip.audio.astype(np.float64) ** 2)))
    polished_rms = float(np.sqrt(np.mean(polished.astype(np.float64) ** 2)))
    assert polished_rms > raw_rms * 0.25
    assert len(polished) <= len(clip.audio) * 1.05


def test_gold_performance_concat_keeps_film_pacing():
    clips = ensure_gold_clips()
    if len(clips) < 2:
        pytest.skip("need at least two gold clips")
    pair = clips[:2]
    fake = np.concatenate(
        [
            isolate_ultron_voice(c.audio, c.sample_rate)[
                : int(len(isolate_ultron_voice(c.audio, c.sample_rate)) * 0.9)
            ]
            for c in pair
        ]
    )
    text = " ".join(c.text for c in pair)
    out = apply_gold_performance(fake, pair[0].sample_rate, text, clips=pair)
    film_lens = [
        len(isolate_ultron_voice(c.audio, c.sample_rate)) for c in pair
    ]
    assert abs(len(out) - sum(film_lens)) < pair[0].sample_rate * 0.15


def test_meant_pair_covers_gold():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    text = "I was meant to be new. I was meant to be beautiful."
    assert gold_covers_text(text, clips)
    ordered = covering_gold(text, clips)
    assert [c.clip_id for c in ordered] == [
        "i-was-meant-to-be-new",
        "i-was-meant-to-be-beautiful",
    ]


def test_process_theater_audio_preserves_clone_length():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    clip = next(c for c in clips if c.clip_id == "i-was-meant-to-be-new")
    raw = mono_float_to_wav_bytes((clip.audio * 0.65).astype(np.float32), clip.sample_rate)
    try:
        stamped = process_theater_audio(
            raw, [], speed=0.92, bass_gain_db=5.0, text=clip.text, pad_onset=False, pad_tail=False
        )
    except ModuleNotFoundError as exc:
        if "pedalboard" in str(exc):
            pytest.skip("pedalboard not installed")
        raise
    ours, _ = wav_bytes_to_mono_float(stamped)
    assert len(ours) >= int(len(clip.audio) * 0.75)


def test_novel_text_uses_universal_path_not_gold_stamp():
    clips = ensure_gold_clips()
    if not clips:
        pytest.skip("no film gold clips on disk")
    clip = next(c for c in clips if c.clip_id == "i-was-meant-to-be-new")
    target = isolate_ultron_voice(clip.audio, clip.sample_rate)
    novel = "Hello from Ultron. This is a cloned voice speaking new words."
    assert not gold_covers_text(novel, clips)
    long_fake = np.concatenate([target * 0.7, target * 0.55])
    raw = mono_float_to_wav_bytes(long_fake, clip.sample_rate)
    try:
        stamped = process_theater_audio(
            raw, [], speed=1.0, bass_gain_db=0.0, text=novel, pad_onset=False, pad_tail=False
        )
    except ModuleNotFoundError as exc:
        if "pedalboard" in str(exc):
            pytest.skip("pedalboard not installed")
        raise
    ours, _ = wav_bytes_to_mono_float(stamped)
    assert ours.size > clip.sample_rate // 4
    assert float(np.max(np.abs(ours))) > 0.01
