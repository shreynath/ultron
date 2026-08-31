import numpy as np

from app.services.reference_delivery import apply_reference_tone, reference_pause_ms
from app.services.ultron_atlas import AtlasPhrase, delivery_atlas, phrase_from_audio
from app.services.wav_io import mono_float_to_wav_bytes


def _tone(sample_rate: int, seconds: float, hz: float) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_delivery_atlas_prefers_profile(tmp_path):
  profile_dir = tmp_path / "gold" / "prof-1"
  profile_dir.mkdir(parents=True)
  sample_rate = 22050
  audio = _tone(sample_rate, 1.0, 95.0)
  phrase = phrase_from_audio("I am free now.", audio, sample_rate)
  assert phrase is not None
  from app.services.ultron_atlas import save_atlas, profile_atlas_path

  save_atlas([phrase], profile_atlas_path(tmp_path, "prof-1"))
  merged = delivery_atlas(data_dir=tmp_path, profile_id="prof-1")
  assert merged
  assert merged[0].text == "I am free now."


def test_reference_tone_moves_toward_profile(tmp_path):
    from app.services.film_identity import save_profile_gold_clips

    sample_rate = 22050
    bright = _tone(sample_rate, 1.0, 180.0)
    deep = _tone(sample_rate, 1.0, 80.0)
    src = tmp_path / "clip.wav"
    src.write_bytes(mono_float_to_wav_bytes(deep, sample_rate))
    save_profile_gold_clips(tmp_path, "prof-1", [(src, "I am free now.")])
    out = apply_reference_tone(bright, sample_rate, tmp_path, "prof-1")
    assert out.shape == bright.shape
    assert float(np.std(out - bright)) > 0.0
