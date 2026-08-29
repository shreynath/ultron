from app.services.engines import normalize_engine


def test_normalize_engine_aliases():
    assert normalize_engine("qwen3-tts") == "qwen"
    assert normalize_engine("chatterbox-multilingual") == "chatterbox"
    assert normalize_engine("chatterbox-turbo") == "chatterbox_turbo"
    assert normalize_engine("qwen") == "qwen"
