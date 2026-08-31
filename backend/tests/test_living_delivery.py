import numpy as np

from app.services.living_delivery import apply_punctuation_pauses, detect_quiet_spans, shape_living_delivery


def _tone_with_gap(sample_rate: int = 8000) -> np.ndarray:
    tone = 0.2 * np.sin(2 * np.pi * 140 * np.arange(sample_rate, dtype=np.float32) / sample_rate)
    gap = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
    return np.concatenate([tone, gap, tone]).astype(np.float32)


def test_detects_quiet_gap():
    sample_rate = 8000
    audio = _tone_with_gap(sample_rate)
    spans = detect_quiet_spans(audio, sample_rate, thresh_db=-35, min_ms=40)
    assert spans
    start, end = spans[0]
    assert 0.05 < (end - start) / sample_rate < 0.2


def test_punctuation_pauses_lengthen_existing_gap():
    sample_rate = 8000
    audio = _tone_with_gap(sample_rate)
    text = "Hello there. Start again."
    out = apply_punctuation_pauses(audio, sample_rate, text)
    assert len(out) > len(audio) + int(sample_rate * 0.12)


def test_pause_insert_does_not_duck_the_word():
    sample_rate = 8000
    audio = _tone_with_gap(sample_rate)
    end_first = sample_rate
    before = float(np.sqrt(np.mean(audio[end_first - 240 : end_first] ** 2)))
    out = apply_punctuation_pauses(audio, sample_rate, "Hello there. Start again.")
    after = float(np.sqrt(np.mean(out[end_first - 240 : end_first] ** 2)))
    assert after > before * 0.92


def test_pauses_do_not_cut_continuous_speech():
    sample_rate = 8000
    audio = (0.2 * np.sin(2 * np.pi * 140 * np.arange(sample_rate * 4) / sample_rate)).astype(
        np.float32
    )
    text = (
        "I wasn't sure you'd wake up. I hoped you would. I wanted to show you. "
        "I think a lot about meteors. Boom! The end. Start again."
    )
    out = apply_punctuation_pauses(audio, sample_rate, text)
    assert len(out) == len(audio)


def test_long_monologue_lengthens_real_gaps():
    sample_rate = 8000
    tone = 0.2 * np.sin(2 * np.pi * 140 * np.arange(int(sample_rate * 0.4)) / sample_rate)
    gap = np.zeros(int(sample_rate * 0.06), dtype=np.float32)
    audio = np.concatenate([tone, gap] * 8).astype(np.float32)
    text = (
        "I wasn't sure you'd wake up. I hoped you would. I wanted to show you. "
        "I think a lot about meteors. The purity of them. Boom! The end. Start again."
    )
    out = apply_punctuation_pauses(audio, sample_rate, text)
    assert len(out) > len(audio) + int(sample_rate * 0.20)


def test_living_delivery_keeps_length_in_same_order():
    sample_rate = 8000
    audio = _tone_with_gap(sample_rate)
    out = shape_living_delivery(
        audio,
        sample_rate,
        "I wasn't sure. Boom!",
        baseline_speed=0.84,
        baseline_bass=5.0,
    )
    assert out.shape[0] >= audio.shape[0]
    assert np.isfinite(out).all()
