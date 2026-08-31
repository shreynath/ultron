import numpy as np

from app.services.f0 import track_f0, voiced_median
from app.services.td_psola import impose_f0_contour, lengthen_keep_pitch
from app.services.ultron_atlas import (
    AtlasPhrase,
    _limit_f0_slope,
    _sanitize_contour,
    apply_ultron_word_prosody,
    phrase_from_audio,
    sentence_sample_ranges,
    word_sample_ranges,
)
from app.services.ultron_tone import apply_ltas_match, measure_ltas_db


def _tone(sample_rate: int, seconds: float, hz: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_word_ranges_cover_audio():
    sample_rate = 8000
    audio = np.linspace(0.1, 0.2, sample_rate, dtype=np.float32)
    ranges = word_sample_ranges(audio, sample_rate, ["i", "was", "meant"])
    assert len(ranges) == 3
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(audio)
    assert ranges[0][1] <= ranges[1][0] + 1


def test_word_ranges_skip_leading_silence():
    sample_rate = 8000
    silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    speech = _tone(sample_rate, 0.8, 110.0)
    audio = np.concatenate([silence, speech])
    ranges = word_sample_ranges(audio, sample_rate, ["was", "meant"])
    assert ranges[0][0] >= int(sample_rate * 0.15)
    assert ranges[0][1] - ranges[0][0] > int(sample_rate * 0.15)


def test_phrase_from_audio_stores_word_pitch():
    sample_rate = 22050
    audio = _tone(sample_rate, 1.2, 110.0)
    phrase = phrase_from_audio("I am free now.", audio, sample_rate)
    assert phrase is not None
    assert "free" in phrase.words
    voiced = [hz for contour in phrase.word_f0 for hz in contour]
    assert voiced
    assert 90 < float(np.median(voiced)) < 130
    assert len(phrase.phrase_f0) > 10
    assert len(phrase.energy) >= 8


def test_word_prosody_copies_atlas_contour():
    sample_rate = 22050
    high = _tone(sample_rate, 1.4, 150.0)
    falling = [140.0 - i * 2.0 for i in range(20)]
    atlas = [
        AtlasPhrase(
            text="I am free now.",
            words=["i", "am", "free", "now"],
            word_f0=[falling, falling, falling, falling],
            duration_sec=1.4,
        )
    ]
    out = apply_ultron_word_prosody(high, sample_rate, "I am free now.", phrases=atlas)
    _times, f0 = track_f0(out, sample_rate)
    median = voiced_median(f0)
    assert median < 130
    assert median > 85


def test_impose_f0_on_tone():
    sample_rate = 22050
    tone = _tone(sample_rate, 1.0, 140.0)
    times = np.linspace(0.05, 0.95, 40)
    target = np.full(40, 100.0)
    out = impose_f0_contour(tone, sample_rate, times, target)
    median = voiced_median(track_f0(out, sample_rate)[1])
    assert 88 < median < 115


def test_lengthen_keep_pitch_changes_duration():
    sample_rate = 22050
    tone = _tone(sample_rate, 0.6, 110.0)
    longer = lengthen_keep_pitch(tone, sample_rate, 1.12)
    assert len(longer) > len(tone) * 1.05
    median = voiced_median(track_f0(longer, sample_rate)[1])
    assert 90 < median < 130


def test_sentence_ranges_split_on_pause():
    sample_rate = 8000
    left = _tone(sample_rate, 0.6, 110.0)
    gap = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    right = _tone(sample_rate, 0.6, 100.0)
    audio = np.concatenate([left, gap, right])
    ranges = sentence_sample_ranges(audio, sample_rate, ["I was new.", "I was beautiful."])
    assert len(ranges) == 2
    assert ranges[0][1] > int(sample_rate * 0.45)
    assert ranges[0][1] < int(sample_rate * 0.95)


def test_sanitize_contour_kills_question_lift():
    rising = [70.0, 72.0, 73.0, 88.0, 110.0, 128.0]
    out = _sanitize_contour(rising, fall=True)
    assert float(out[-1]) < 80.0
    assert float(out[-1]) < float(np.median(out[:3]))


def test_limit_f0_slope_kills_period_jumps():
    times = np.linspace(0.0, 0.2, 21)
    target = np.concatenate([np.full(10, 70.0), np.full(11, 130.0)])
    out = _limit_f0_slope(times, target, max_hz_per_sec=320.0)
    step = np.max(np.abs(np.diff(out)))
    assert step < 8.0
    assert float(out[-1]) > 90.0


def test_limit_f0_slope_does_not_leap_across_a_pause():
    times = np.array([0.00, 0.01, 0.02, 0.40, 0.41, 0.42])
    target = np.array([72.0, 73.0, 74.0, 140.0, 141.0, 142.0])
    out = _limit_f0_slope(times, target, max_hz_per_sec=320.0)
    assert abs(float(out[3] - out[2])) < 12.0


def test_statement_lands_down_not_up():
    sample_rate = 22050
    t = np.arange(int(sample_rate * 1.3), dtype=np.float32) / sample_rate
    inst = 118.0 + 42.0 * (t / t[-1])
    phase = np.cumsum(2 * np.pi * inst / sample_rate)
    high = (0.22 * np.sin(phase)).astype(np.float32)
    low = [92.0, 90.0, 88.0, 84.0, 80.0, 76.0, 72.0, 68.0]
    atlas = [
        AtlasPhrase(
            text="I am free now.",
            words=["i", "am", "free", "now"],
            word_f0=[low, low, low, low],
            duration_sec=1.3,
        )
    ]
    out = apply_ultron_word_prosody(high, sample_rate, "I am free now.", phrases=atlas)
    times, f0 = track_f0(out, sample_rate)
    voiced = f0 > 1
    assert np.any(voiced)
    t_end = float(times[voiced][-1])
    tail = (times >= t_end - 0.22) & voiced
    head = (times <= 0.25) & voiced
    if np.any(tail) and np.any(head):
        assert float(np.median(f0[tail])) <= float(np.median(f0[head])) + 6.0


def test_ltas_match_lifts_chest():
    sample_rate = 22050
    t = np.arange(int(sample_rate * 0.8), dtype=np.float32) / sample_rate
    bright = (0.18 * np.sin(2 * np.pi * 110 * t) + 0.12 * np.sin(2 * np.pi * 2200 * t)).astype(
        np.float32
    )
    dark = (0.22 * np.sin(2 * np.pi * 110 * t) + 0.18 * np.sin(2 * np.pi * 95 * t)).astype(np.float32)
    target = measure_ltas_db(dark, sample_rate)
    out = apply_ltas_match(bright, sample_rate, target)
    src = measure_ltas_db(bright, sample_rate)
    dst = measure_ltas_db(out, sample_rate)
    # 90–130 Hz band should move toward the darker take.
    assert dst[1] >= src[1] - 0.2


def test_ultron_metal_stays_bounded():
    sample_rate = 22050
    t = np.arange(int(sample_rate * 0.5), dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    from app.services.ultron_tone import apply_ultron_metal

    out = apply_ultron_metal(tone, sample_rate, amount=0.9)
    assert out.dtype == np.float32
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-5
    assert out.shape == tone.shape
