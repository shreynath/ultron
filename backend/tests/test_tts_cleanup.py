import numpy as np

from app.services.tts_cleanup import cleanup_mlx_wav, strip_trailing_codec_only
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float


def test_strip_keeps_phrase_body():
    sample_rate = 1000
    speech = np.concatenate(
        [
            np.full(800, 0.25, dtype=np.float32),
            np.full(600, 0.03, dtype=np.float32),
        ]
    )
    trimmed = strip_trailing_codec_only(speech, sample_rate)
    assert len(trimmed) == len(speech)


def test_cleanup_preserves_attack():
    sample_rate = 24000
    lead = np.full(int(sample_rate * 0.04), 0.004, dtype=np.float32)
    speech = np.full(int(sample_rate * 0.6), 0.2, dtype=np.float32)
    audio = np.concatenate([lead, speech])
    cleaned, _ = wav_bytes_to_mono_float(cleanup_mlx_wav(mono_float_to_wav_bytes(audio, sample_rate)))
    assert len(cleaned) >= len(audio) * 0.98
    assert abs(cleaned[0] - lead[0]) < 0.001
