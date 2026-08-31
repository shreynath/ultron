from app.services.dramatic_planner import plan_phrases, plan_theater_delivery, _split_sentences
from app.services.phrase_effects import effects_chain_for_phrase


def test_split_sentences():
    parts = _split_sentences("I had strings. But now I am free!")
    assert len(parts) == 2


def test_short_punchline_gets_more_intensity():
    plans = plan_phrases("I am free.", baseline_speed=0.84, baseline_bass=5.0)
    assert len(plans) == 1
    assert plans[0].intensity >= 0.5
    assert plans[0].speed <= 0.90
    assert 0.0 <= plans[0].bass_boost <= 4.0


def test_multi_sentence_varies_delivery():
    text = "Everyone creates the thing they dread. I had strings, but now I am free."
    plans = plan_phrases(text, baseline_speed=0.84, baseline_bass=5.0)
    assert len(plans) >= 2
    assert plans[0].pause_after_ms >= 150
    assert plans[-1].pause_after_ms == 0


def test_theater_plan_keeps_contractions():
    plan = plan_theater_delivery("I wasn't sure you'd wake up. I don't have anyone else.")
    assert "wasn't" in plan.text
    assert "you'd" in plan.text
    assert "don't" in plan.text
    assert "was not" not in plan.text
    assert "do not" not in plan.text
    plan = plan_theater_delivery(
        "Everyone creates the thing they dread. I had strings, but now I am free.",
        baseline_speed=0.84,
        baseline_bass=9.8,
    )
    assert "dread" in plan.text
    assert 0.0 <= plan.bass_boost <= 4.0
    assert 0.66 <= plan.speed <= 1.0


def test_theater_speed_spans_wide_range_by_intensity():
    punchline = plan_theater_delivery("I am free.", baseline_speed=0.84, baseline_bass=5.0)
    exposition = plan_theater_delivery(
        "There was an idea to bring together a group of remarkable people "
        "to see if they could become something more.",
        baseline_speed=0.84,
        baseline_bass=5.0,
    )
    assert punchline.speed < exposition.speed
    assert exposition.speed - punchline.speed >= 0.02


def test_theater_effects_are_clone_first():
    plan = plan_theater_delivery("I am free.", baseline_speed=0.84, baseline_bass=5.0)
    chain = effects_chain_for_phrase("theater_auto", plan)
    types = {effect["type"] for effect in chain}
    assert types <= {"reverb", "chorus", "compressor"}
    assert "pitch_shift" not in types
    reverb = next(effect for effect in chain if effect["type"] == "reverb")
    assert reverb["params"]["wet_level"] <= 0.50
    compressor = next(effect for effect in chain if effect["type"] == "compressor")
    assert compressor.get("enabled") is False


def test_phrase_effects_scale_with_intensity():
    plans = plan_phrases("You will all die!", baseline_speed=0.84, baseline_bass=5.0)
    chain = effects_chain_for_phrase("ultron_cinematic", plans[0])
    reverb = next(e for e in chain if e["type"] == "reverb")
    assert reverb["params"]["wet_level"] > 0.15


METEOR_SPEECH = (
    "I wasn’t sure you’d wake up. I hoped you would. I wanted to show you. "
    "I... don't have anyone else. I think a lot about meteors. The purity of them. "
    "Boom! The end. Start again. The world made clean for the new man to rebuild. "
    "I was meant to be new. I was meant to be beautiful. The world would have looked "
    "to the sky and seen hope. Seen mercy. Instead they'll look up in horror because of you. "
    "You've wounded me. I give you full marks for that. But, like my old man said, "
    "what doesn't kill you... just makes me stronger!"
)


def test_splits_on_punctuation_and_ellipsis():
    parts = _split_sentences(METEOR_SPEECH)
    joined = " ".join(parts).lower()
    assert len(parts) >= 16
    assert any(p.lower().startswith("boom") for p in parts)
    assert any("purity" in p.lower() for p in parts)
    assert any(p.rstrip().endswith("...") or p.rstrip().endswith("…") for p in parts)
    assert "don't have anyone else" in joined
    assert "just makes me stronger" in joined


def test_preserves_contractions_in_plans():
    plans = plan_phrases(METEOR_SPEECH)
    spoken = " ".join(plan.text for plan in plans)
    assert "wasn't" in spoken
    assert "you'd" in spoken
    assert "don't" in spoken
    assert "they'll" in spoken
    assert "You've" in spoken or "you've" in spoken
    assert "was not" not in spoken
    assert "do not" not in spoken


def test_punctuation_gets_real_pauses():
    plans = plan_phrases(METEOR_SPEECH)
    sentence_pauses = [
        plan.pause_after_ms
        for plan in plans[:-1]
        if plan.text.rstrip().endswith((".", "!", "?", "...", "…"))
    ]
    assert sentence_pauses
    assert all(pause >= 280 for pause in sentence_pauses)
    periods = [
        plan.pause_after_ms
        for plan in plans[:-1]
        if plan.text.rstrip().endswith(".") and "..." not in plan.text
    ]
    assert periods
    assert all(pause >= 400 for pause in periods)
    ellipsis = next(plan for plan in plans if "..." in plan.text or plan.text.rstrip().endswith("…"))
    period = next(plan for plan in plans if plan.text.rstrip().endswith(".") and "..." not in plan.text)
    assert ellipsis.pause_after_ms > period.pause_after_ms


def test_meant_to_be_beautiful_stays_deep():
    plans = plan_phrases("I was meant to be new. I was meant to be beautiful.")
    assert len(plans) == 2
    assert plans[0].pause_after_ms >= 400
    for plan in plans:
        assert plan.contour != "wound"
        assert plan.pitch_semitones <= 0.05
        assert "pause" in plan.instruct.lower()


def test_delivery_fluctuates_across_phrases():
    plans = plan_phrases(METEOR_SPEECH)
    boom = next(plan for plan in plans if plan.text.lower().startswith("boom"))
    purity = next(plan for plan in plans if "purity" in plan.text.lower())
    ending = next(plan for plan in plans if "stronger" in plan.text.lower())
    the_end = next(plan for plan in plans if plan.text.lower().startswith("the end"))

    assert boom.speed > purity.speed + 0.06
    assert boom.pitch_semitones > the_end.pitch_semitones + 0.4
    assert boom.energy_gain_db > purity.energy_gain_db
    assert the_end.bass_boost > boom.bass_boost
    assert ending.contour == "punch"
    assert len({plan.speed for plan in plans}) >= 4
    assert len({plan.pitch_semitones for plan in plans}) >= 4
    assert max(plan.pitch_semitones for plan in plans) - min(plan.pitch_semitones for plan in plans) >= 0.6
    assert max(plan.energy_gain_db for plan in plans) - min(plan.energy_gain_db for plan in plans) >= 1.0
    assert any("ultron" in plan.instruct.lower() or "metallic" in plan.instruct.lower() for plan in plans)
    assert any("pause" in plan.instruct.lower() for plan in plans)
