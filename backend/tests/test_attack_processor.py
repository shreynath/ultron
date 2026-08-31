import numpy as np

from app.services.attack_processor import process_theater_audio
from app.services.wav_io import mono_float_to_wav_bytes, wav_bytes_to_mono_float


def test_onset_pad_zeros_click_window():
    sample_rate = 24000
    audio = np.array([0.5, 0.02, 0.03, *np.full(2400, 0.2)], dtype=np.float32)
    raw = mono_float_to_wav_bytes(audio, sample_rate)
    out, _ = wav_bytes_to_mono_float(process_theater_audio(raw, [], speed=1.0, bass_gain_db=0.0))
    onset = out[: int(sample_rate * 0.02)]
    assert float(np.max(np.abs(onset))) < 0.001
