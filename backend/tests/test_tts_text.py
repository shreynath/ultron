from app.services.tts_text import normalize_for_tts, theatrical_tts_text


def test_preserves_contractions_for_tts():
    assert normalize_for_tts("but now I'm free") == "but now I'm free."
    assert normalize_for_tts("I wasn't sure you'd wake up.") == "I wasn't sure you'd wake up."
    assert "do not" not in normalize_for_tts("I... don't have anyone else.")
    assert "don't" in normalize_for_tts("I... don't have anyone else.")


def test_normalizes_curly_apostrophes_without_expanding():
    assert "wasn't" in normalize_for_tts("I wasn’t sure")
    assert "you'd" in normalize_for_tts("you’d wake up")


def test_no_leading_space():
    assert not normalize_for_tts("Hello").startswith(" ")


def test_theatrical_keeps_words_and_punctuation():
    result = theatrical_tts_text("I had strings, but now I'm free.")
    assert result == "I had strings, but now I'm free."
    assert "..." not in result


def test_theatrical_keeps_contractions():
    result = theatrical_tts_text("I don't have anyone else.")
    assert "don't" in result
    assert "do not" not in result


def test_theatrical_pause_cues_keep_words():
    result = theatrical_tts_text(
        "I was meant to be new. I was meant to be beautiful. "
        "Seen mercy. Instead they'll look up in horror because of you."
    )
    assert "meant to be new" in result
    assert "meant to be beautiful" in result
    assert "they'll" in result
    assert "they will" not in result
    assert "\n" in result


def test_theatrical_short_line_has_no_break():
    result = theatrical_tts_text("I was meant to be new. I was meant to be beautiful.")
    assert "\n" not in result
    assert "meant to be beautiful" in result
