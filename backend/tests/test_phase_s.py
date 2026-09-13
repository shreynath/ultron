"""Phase S only recommends bounded Phase-O interventions; it never trains itself."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.foundation.phase_p import CapabilityComparison
from app.foundation.phase_r import LearningAttribution, LearningEvidenceLevel
from app.foundation.phase_s import (
    AdaptiveAction,
    CheckpointEvidence,
    DifficultyBand,
    InterventionStatus,
    PhaseSEngine,
    SamplingPolicy,
    StopReason,
    adapt_intervention,
    adaptive_difficulty,
    create_early_stop_policy,
    create_intervention,
    create_learning_policy,
    efficiency_for,
    forgetting_for,
    learning_curve_for,
    learning_rate_variants,
    select_checkpoint,
    should_stop,
    stability_for,
    summarize_replication,
)
from app.foundation.training import TrainingResult


def _intervention(
    identifier: str,
    *,
    target_weight: float = 1.0,
    learning_rate: float = 3e-4,
    releases: tuple[str, ...] = ("release-a",),
):
    return create_intervention(
        intervention_id=identifier,
        parent_model_id="baseline",
        parent_checkpoint_sha256="a" * 64,
        target_id="target-recursion",
        dataset_release_ids=releases,
        dataset_release_hashes=tuple("b" * 64 for _ in releases),
        curriculum_id="curriculum-a",
        curriculum_ordering=(
            DifficultyBand.FOUNDATIONAL,
            DifficultyBand.EASY,
            DifficultyBand.MODERATE,
        ),
        mixture_weights={"target": 1.0},
        sampling_policy=SamplingPolicy.TARGET_DENSITY,
        dataset_weights={release: 1.0 / len(releases) for release in releases},
        target_weight=target_weight,
        difficulty_weights={DifficultyBand.FOUNDATIONAL.value: 1.0},
        learning_rate=learning_rate,
        scheduler="linear_warmup_constant",
        optimizer="adamw",
        batch_size=1,
        sequence_length=16,
        gradient_accumulation=1,
        weight_decay=0.01,
        warmup_steps=1,
        max_tokens=192,
        max_steps=12,
        seed=17,
        checkpoint_interval=1,
        early_stop_policy=create_early_stop_policy(
            "stop", capability_evaluation_available=True, maximum_gradient_norm=5
        ),
        status=InterventionStatus.APPROVED_FOR_PHASE_O,
    )


def _attribution() -> LearningAttribution:
    return LearningAttribution(
        attribution_id="learning-r",
        experiment_id="r",
        baseline_run_id="base",
        candidate_run_id="candidate",
        weights_changed=True,
        target_capability_delta=0,
        generalization_delta=0,
        regression_rate=0,
        evidence=LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
        rationale="measured zero",
    )


def test_immutable_intervention_policy_and_evidence_driven_adaptation(
    tmp_path: Path,
) -> None:
    policy = create_learning_policy(
        "s-policy",
        maximum_experiments=2,
        maximum_tokens_per_intervention=192,
        maximum_adaptive_iterations=1,
    )
    first, second = (
        _intervention("s-1", target_weight=1),
        _intervention("s-2", target_weight=2),
    )
    first.verify_integrity()
    engine = PhaseSEngine(tmp_path)
    engine.configure(policy)
    engine.record_intervention(first, policy)
    decision = adapt_intervention(
        previous=first,
        next_intervention=second,
        attribution=_attribution(),
        policy=policy,
        iteration=1,
    )
    assert decision.action is AdaptiveAction.INCREASE_TARGET_DENSITY
    engine.persist("decisions", decision.decision_id, decision)
    with pytest.raises(ValueError, match="configuration changed"):
        engine.record_intervention(
            first.model_copy(update={"learning_rate": 1e-3}), policy
        )
    with pytest.raises(ValueError, match="adaptive iterations"):
        adapt_intervention(
            previous=first,
            next_intervention=second,
            attribution=_attribution(),
            policy=policy,
            iteration=2,
        )


def test_scheduling_checkpoint_stability_retention_and_replication() -> None:
    policy = create_learning_policy("policy", maximum_learning_rate_variants=3)
    assert learning_rate_variants(1e-3, policy) == (5e-4, 1e-3, 2e-3)
    assert (
        adaptive_difficulty(
            target_id="t", current=DifficultyBand.EASY, success_rate=0.95, policy=policy
        ).action
        == "ADVANCE"
    )
    assert (
        adaptive_difficulty(
            target_id="t", current=DifficultyBand.HARD, success_rate=0.1, policy=policy
        ).action
        == "ADD_PREREQUISITES"
    )
    selection = select_checkpoint(
        intervention_id="s",
        maximum_regression_rate=0,
        candidates=(
            CheckpointEvidence(
                checkpoint_id="one",
                checkpoint_sha256="1" * 64,
                target_capability_delta=0.2,
                generalization_delta=0.1,
                regression_rate=0,
                training_stable=True,
                tokens_processed=100,
            ),
            CheckpointEvidence(
                checkpoint_id="two",
                checkpoint_sha256="2" * 64,
                target_capability_delta=0.3,
                generalization_delta=0,
                regression_rate=0.1,
                training_stable=True,
                tokens_processed=200,
            ),
        ),
    )
    assert selection.selected_checkpoint_id == "one"
    comparison = CapabilityComparison(
        suite_id="suite",
        suite_version="v1",
        baseline_run_id="base",
        candidate_run_id="candidate",
        domain_deltas={"PROGRAMMING": 0.1, "REASONING": -0.2},
        regression_rate=0.1,
        target_improved=True,
    )
    forgetting = forgetting_for(
        intervention_id="s", comparison=comparison, tolerance=0.05
    )
    assert forgetting.catastrophic_forgetting_risk and forgetting.rehearsal_recommended
    curve = learning_curve_for(
        intervention_id="s",
        points=(
            CheckpointEvidence(
                checkpoint_id="1",
                checkpoint_sha256="1" * 64,
                target_capability_delta=0.1,
                regression_rate=0,
                training_stable=True,
                tokens_processed=10,
            ),
            CheckpointEvidence(
                checkpoint_id="2",
                checkpoint_sha256="2" * 64,
                target_capability_delta=0,
                regression_rate=0,
                training_stable=True,
                tokens_processed=20,
            ),
            CheckpointEvidence(
                checkpoint_id="3",
                checkpoint_sha256="3" * 64,
                target_capability_delta=0,
                regression_rate=0,
                training_stable=True,
                tokens_processed=30,
            ),
        ),
    )
    assert curve.status == "SATURATING"
    assert (
        summarize_replication(
            intervention_family_id="s", seeds=(1,), target_deltas=(0.1,), required=2
        ).status
        == "INSUFFICIENT_REPLICATION"
    )
    assert (
        efficiency_for(
            intervention_id="s",
            capability_gain=0.2,
            generalization_gain=0.1,
            tokens=100,
            compute_units=50,
        ).capability_gain_per_compute
        == 0.004
    )


def test_stability_and_early_stop_do_not_substitute_loss_for_capability() -> None:
    result = TrainingResult(
        run_id="s",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        steps_completed=1,
        tokens_processed=16,
        training_loss=1,
        validation_loss=3,
        perplexity=1,
        checkpoint_path="/tmp/checkpoint",
        tokens_per_second=1,
        maximum_gradient_norm=None,
    )
    policy = create_early_stop_policy(
        "stop-no-capability",
        capability_evaluation_available=False,
        maximum_validation_loss_increase=0.1,
    )
    stability = stability_for(intervention_id="s", result=result, early_stop=policy)
    assert stability.stable and "gradient_norm" in stability.unavailable_metrics
    assert (
        should_stop(stability=stability, comparison=None, early_stop=policy)
        is StopReason.NONE
    )
    capability_policy = create_early_stop_policy(
        "stop-with-capability",
        capability_evaluation_available=True,
        maximum_validation_loss_increase=0.1,
    )
    assert (
        should_stop(stability=stability, comparison=None, early_stop=capability_policy)
        is StopReason.VALIDATION_DEGRADATION
    )
