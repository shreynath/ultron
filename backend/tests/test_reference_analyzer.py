from app.services.reference_analyzer import (
    score_transcript_solo_speech,
    estimate_speaking_rate_wps,
    recommend_delivery_from_clips,
)


def test_solo_speech_scores_long_monologue_higher():
    monologue = (
        "There were no strings on me. I know you mean well, but you just didn't think it through. "
        "You want to protect the world, but you don't want it to change."
    )
    short = "Yes. No. What?"
    assert score_transcript_solo_speech(monologue) > score_transcript_solo_speech(short)


def test_solo_speech_penalizes_dialogue():
    dialogue = "Tony? Are you there? Steve, we need to talk. What do you want?"
    monologue = "I am going to bring about what the world needs. A new age begins now."
    assert score_transcript_solo_speech(monologue) > score_transcript_solo_speech(dialogue)


def test_recommend_delivery_slow_speech():
    clips = [
        ("one two three four five six seven eight nine ten", 8.0, 0.7),
        ("eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen", 9.0, 0.65),
    ]
    delivery = recommend_delivery_from_clips(clips)
    assert delivery["speed"] <= 0.90
    assert 2.0 <= delivery["bass_boost"] <= 6.0


def test_speaking_rate():
    assert estimate_speaking_rate_wps("one two three four five", 5.0) == 1.0
