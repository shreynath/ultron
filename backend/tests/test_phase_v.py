from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.foundation.phase_v import (
    CapabilityExample,
    EvaluationSummary,
    PhaseVStore,
    PhaseVVerdict,
    create_curriculum,
    create_manifest,
    grade_addition,
    observe_density,
    verdict,
    wilson_interval,
)


def _example(identifier: str, split: str, text: str) -> CapabilityExample:
    return CapabilityExample(
        example_id=identifier,
        capability_id="addition",
        skill="add",
        difficulty=1,
        structure_id=identifier,
        source_id="owned",
        license="CC0-1.0",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        split=split,
        prompt=text,
        answer="5",
    )


def _summary(**updates: float) -> EvaluationSummary:
    values = {
        "direct": 0.0,
        "surface": 0.0,
        "structural": 0.0,
        "counterfactual": 0.0,
        "compositional": 0.0,
        "regression": 1.0,
        "composite_generalization": 0.0,
        "n": 20,
        "wilson_low": 0.0,
        "wilson_high": 0.16,
    }
    values.update(updates)
    return EvaluationSummary(**values)


def test_curriculum_train_test_separation_and_manifest_immutability(
    tmp_path: Path,
) -> None:
    curriculum = create_curriculum(
        capability_id="addition",
        version="v1",
        selection_reason="deterministic graded arithmetic",
        examples=(
            _example("train", "TRAIN", "ADD 2 3"),
            _example("test", "TEST_DIRECT", "ADD 7 8"),
        ),
        semantic_contamination_detection="LIMITED",
    )
    manifest = create_manifest(
        experiment_id="v-b",
        control_experiment_id="v-a",
        changed_variable="target_fraction",
        primary_capability_id="addition",
        curriculum_sha256=curriculum.curriculum_sha256,
        model_config_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        dataset_release_ids=("general", "target"),
        training_tokens=20_000_000,
        batch_size=2,
        context_length=64,
        optimizer="adamw",
        learning_rate=3e-4,
        seed=17,
        requested_target_fraction=0.25,
    )
    store = PhaseVStore(tmp_path)
    store.write_immutable("manifest", manifest)
    with pytest.raises(ValueError, match="already exists"):
        store.write_immutable("manifest", manifest.model_copy(update={"seed": 18}))
    with pytest.raises(ValueError, match="overlap"):
        create_curriculum(
            capability_id="addition",
            version="v2",
            selection_reason="x",
            examples=(
                _example("a", "TRAIN", "same"),
                _example("b", "TEST_DIRECT", "same"),
            ),
        )


def test_grading_density_uncertainty_and_verdicts() -> None:
    assert grade_addition(output="answer: 5", expected="5").answer == 1
    assert grade_addition(output="none", expected="5").syntax == 0
    density = observe_density(requested=0.25, target_tokens=250, total_tokens=1000)
    assert density.within_tolerance
    low, high = wilson_interval(10, 20)
    assert 0 <= low < high <= 1
    baseline = _summary()
    partial = _summary(structural=0.1)
    acquired = _summary(direct=0.2, structural=0.2, composite_generalization=0.2)
    assert (
        verdict(baseline=baseline, trained=partial, loss_delta=-1)
        is PhaseVVerdict.PARTIAL_CAPABILITY_ACQUISITION
    )
    assert (
        verdict(baseline=baseline, trained=acquired, loss_delta=-1, replicated=True)
        is PhaseVVerdict.REPLICATED_CAPABILITY_ACQUISITION
    )
    assert (
        verdict(baseline=baseline, trained=_summary(direct=0.2), loss_delta=-1, replicated=True)
        is PhaseVVerdict.REPLICATED_CAPABILITY_ACQUISITION
    )
