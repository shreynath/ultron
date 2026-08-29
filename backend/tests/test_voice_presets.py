from app.services.voice_presets import resolve_voice_delivery, ultron_cinematic_effects_chain


def test_ultron_effects_chain_has_depth_and_space():
    chain = ultron_cinematic_effects_chain()
    types = [effect["type"] for effect in chain]
    assert "pitch_shift" in types
    assert "reverb" in types
    assert "chorus" in types


def test_resolve_voice_delivery_cinematic():
    instruct, effects = resolve_voice_delivery("ultron_cinematic")
    assert instruct
    assert effects


def test_resolve_voice_delivery_natural():
    instruct, effects = resolve_voice_delivery("natural")
    assert instruct is None
    assert effects is None
