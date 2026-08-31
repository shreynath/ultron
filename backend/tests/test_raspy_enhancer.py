import numpy as np

from app.services.raspy_enhancer import _rasp_envelope, apply_sinister_vibration


def test_rasp_envelope_targets_noisy_frames():
    sample_rate = 1000
    smooth = np.full(500, 0.1, dtype=np.float32)
    raspy = np.random.default_rng(0).normal(0, 0.35, 500).astype(np.float32)
    audio = np.concatenate([smooth, raspy])
    envelope = _rasp_envelope(audio, sample_rate)
    assert float(np.mean(envelope[100:450])) < float(np.mean(envelope[550:950]))


def test_sinister_vibration_leaves_clean_speech_mostly_unchanged():
    sample_rate = 24000
    audio = np.sin(np.linspace(0, 40 * np.pi, sample_rate, dtype=np.float32)) * 0.2
    processed = apply_sinister_vibration(audio, sample_rate, intensity=0.8)
    assert np.mean(np.abs(processed - audio)) < 0.02
