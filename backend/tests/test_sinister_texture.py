import numpy as np

from app.services.sinister_texture import apply_sinister_vibrato, rasp_envelope


def test_rasp_envelope_rises_on_noisy_frames():
    sample_rate = 1000
    tone = np.full(500, 0.2, dtype=np.float32)
    noisy = np.concatenate([tone, np.random.default_rng(0).normal(0, 0.25, 200).astype(np.float32)])
    calm = float(np.mean(rasp_envelope(tone, sample_rate)))
    rough = float(np.mean(rasp_envelope(noisy, sample_rate)[500:]))
    assert rough > calm + 0.05


def test_sinister_vibrato_matches_input_length():
    sample_rate = 24000
    audio = np.random.default_rng(1).normal(0, 0.1, 125076).astype(np.float32)
    out = apply_sinister_vibrato(audio, sample_rate)
    assert out.shape == audio.shape


def test_sinister_vibrato_leaves_silence_alone():
    sample_rate = 1000
    silence = np.zeros(400, dtype=np.float32)
    assert np.allclose(apply_sinister_vibrato(silence, sample_rate), silence)


def test_chest_body_adds_low_end_without_changing_length():
    sample_rate = 24000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    from app.services.sinister_texture import apply_chest_body

    out = apply_chest_body(tone, sample_rate, intensity=1.0)
    assert out.shape == tone.shape
    assert not np.allclose(out, tone)


def test_chest_body_does_not_throb_on_long_takes():
    sample_rate = 8000
    t = np.arange(int(sample_rate * 9.0), dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    from app.services.sinister_texture import apply_chest_body

    out = apply_chest_body(tone, sample_rate, intensity=1.0)
    delta = out - tone
    lfo = np.sin(2.0 * np.pi * 1.85 * t)
    assert abs(float(np.mean(delta * lfo))) < 0.0008


def test_pitch_vibrato_changes_voiced_audio():
    sample_rate = 24000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = (0.2 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    from app.services.sinister_texture import apply_chest_rumble

    out = apply_chest_rumble(tone, sample_rate)
    assert out.shape == tone.shape
    assert not np.allclose(out, tone)


def test_energy_contour_boosts_punch_front():
    sample_rate = 8000
    audio = np.full(sample_rate, 0.2, dtype=np.float32)
    from app.services.sinister_texture import apply_energy_contour

    punch = apply_energy_contour(audio, sample_rate, "punch", 1.0)
    land = apply_energy_contour(audio, sample_rate, "land", 1.0)
    assert abs(punch[100]) > abs(land[100])
    assert abs(land[-100]) > abs(punch[-100])
