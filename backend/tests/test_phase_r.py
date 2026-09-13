"""Phase R evidence stays conservative even when weights and loss change."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    CapabilityItemResult,
    CapabilityRun,
    EvaluationPartition,
    ItemOutcome,
    ItemRole,
    compare_capability_runs,
    create_capability_suite,
)
from app.foundation.phase_r import (
    GeneralizationClass,
    LearningEvidenceLevel,
    PhaseREngine,
    assess_generalization,
    bounded_designs,
    choose_dataset_champion,
    classify_learning_evidence,
    create_experiment,
    create_experiment_policy,
    create_generalization_policy,
    data_value_for,
    dataset_attributions,
    detect_contamination,
    diagnose_failure,
    efficiency_for,
    next_recommendation,
)


def _suite(root: Path):
    items = (
        CapabilityItem(
            task_id="id",
            domain=CapabilityDomain.PROGRAMMING,
            partition=EvaluationPartition.TEST,
            role=ItemRole.TARGET,
            prompt="write a recursive sum",
            expected_answer="sum base case",
            difficulty=3,
            provenance={"source": "owned"},
        ),
        CapabilityItem(
            task_id="struct",
            domain=CapabilityDomain.GENERALIZATION,
            partition=EvaluationPartition.TEST,
            role=ItemRole.GENERALIZATION,
            prompt="trace recursive sum for five",
            expected_answer="five recursion trace",
            difficulty=5,
            provenance={"source": "owned"},
        ),
        CapabilityItem(
            task_id="compose",
            domain=CapabilityDomain.GENERALIZATION,
            partition=EvaluationPartition.TEST,
            role=ItemRole.GENERALIZATION,
            prompt="combine recursion and parity",
            expected_answer="combined answer",
            difficulty=6,
            provenance={"source": "owned"},
        ),
    )
    return create_capability_suite(
        root / "phase-r-suite.json",
        suite_id="phase-r-suite",
        version="v1",
        partition=EvaluationPartition.TEST,
        items=items,
    )


def _run(suite, candidate: str, scores: tuple[float, float, float]) -> CapabilityRun:
    results = tuple(
        CapabilityItemResult(
            task_id=item.task_id,
            domain=item.domain,
            partition=item.partition,
            role=item.role,
            outcome=ItemOutcome.PASS if score else ItemOutcome.FAIL,
            score=score,
            output="",
            detail="fixture",
        )
        for item, score in zip(suite.items, scores, strict=True)
    )
    return CapabilityRun(
        run_id=f"{candidate}-run",
        suite_id=suite.suite_id,
        suite_version=suite.version,
        suite_sha256=suite.suite_sha256,
        checkpoint_id=f"{candidate}-checkpoint",
        checkpoint_path=f"/tmp/{candidate}",
        candidate_id=candidate,
        results=results,
        domain_scores={
            CapabilityDomain.PROGRAMMING.value: scores[0],
            CapabilityDomain.GENERALIZATION.value: sum(scores[1:]) / 2,
        },
    )


def _experiment():
    return create_experiment(
        experiment_id="phase-r-experiment",
        parent_model_id="baseline",
        parent_checkpoint_sha256="a" * 64,
        candidate_model_id="candidate",
        candidate_checkpoint_sha256="b" * 64,
        target_id="target-recursion",
        target_domain=CapabilityDomain.PROGRAMMING.value,
        dataset_release_ids=("data-a", "data-b"),
        dataset_release_hashes=("c" * 64, "d" * 64),
        curriculum_id="curriculum-recursion",
        curriculum_ordering_hash="e" * 64,
        baseline_evaluation_ids=("baseline-run",),
        training_config_sha256="f" * 64,
        evaluation_config_sha256="0" * 64,
        analysis_policy_id="policy",
        analysis_policy_fingerprint="1" * 64,
        seed=7,
        training_tokens=100,
        optimizer_steps=5,
        wall_seconds=3,
        compute_units=50,
        runtime="local-mlx",
        software_version="test",
    )


def test_immutable_experiment_bounded_design_attribution_efficiency_and_persistence(
    tmp_path: Path,
) -> None:
    policy = create_experiment_policy(
        "policy",
        maximum_experiments=3,
        maximum_datasets_per_experiment=3,
        maximum_training_tokens=100,
        maximum_compute_units=50,
    )
    experiment = _experiment()
    experiment.verify_integrity()
    engine = PhaseREngine(tmp_path)
    engine.configure(policy)
    engine.record_experiment(experiment)
    with pytest.raises(ValueError, match="Experiment manifest changed"):
        engine.record_experiment(experiment.model_copy(update={"decision": "CHANGED"}))
    designs = bounded_designs(
        target_id=experiment.target_id,
        dataset_release_ids=("data-a", "data-b", "data-c", "data-d"),
        estimated_tokens=100,
        estimated_compute_units=50,
        policy=policy,
    )
    assert [design.kind for design in designs] == [
        "FULL_MIXTURE",
        "LEAVE_ONE_OUT",
        "LEAVE_ONE_OUT",
    ]
    attribution = dataset_attributions(
        experiment_id=experiment.experiment_id,
        full_mixture_delta=0.3,
        leave_one_out_deltas={"data-a": 0.1, "data-b": None},
    )
    assert attribution[0].observed_marginal_delta == 0.2
    assert attribution[1].status == "INCONCLUSIVE_CONTRIBUTION"
    efficiency = efficiency_for(
        experiment=experiment,
        capability_gain=0.2,
        generalization_gain=0.1,
        newly_solved_items=2,
    )
    assert efficiency.capability_gain_per_token == 0.002
    value = data_value_for(
        dataset_release_id="data-a",
        experiment_id=experiment.experiment_id,
        capability_contribution=-0.1,
        generalization_contribution=0.2,
        target_importance=0.8,
        evidence_confidence=0.8,
        novelty=0.8,
        source_reliability=1,
        contamination_risk=0,
        token_cost=100,
        compute_cost=50,
    )
    assert value.status.value == "HARMFUL"
    champion_value = data_value_for(
        dataset_release_id="data-b",
        experiment_id=experiment.experiment_id,
        capability_contribution=0.2,
        generalization_contribution=0.2,
        target_importance=0.8,
        evidence_confidence=0.8,
        novelty=0.8,
        source_reliability=1,
        contamination_risk=0,
        token_cost=100,
        compute_cost=50,
    )
    champion = choose_dataset_champion(
        values=(value, champion_value), release_hashes={"data-b": "2" * 64}
    )
    assert champion and champion.dataset_release_id == "data-b"


def test_generalization_requires_clean_independent_evidence_and_distinguishes_loss_signal(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    baseline = _run(suite, "baseline", (0, 0, 0))
    candidate = _run(suite, "candidate", (1, 0, 0))
    classes = {
        "id": GeneralizationClass.IN_DISTRIBUTION,
        "struct": GeneralizationClass.STRUCTURAL,
        "compose": GeneralizationClass.COMPOSITIONAL,
    }
    assessment = assess_generalization(
        experiment_id="x",
        baseline=baseline,
        candidate=candidate,
        suite=suite,
        classes=classes,
        policy=create_generalization_policy("g"),
    )
    comparison = compare_capability_runs(baseline, candidate)
    evidence = classify_learning_evidence(
        experiment_id="x",
        baseline=baseline,
        candidate=candidate,
        comparison=comparison,
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=True,
        training_loss_delta=-0.3,
        held_out_loss_delta=-0.1,
        generalization=assessment,
    )
    assert assessment.status == "NO_GENERALIZATION_EVIDENCE"
    assert evidence.evidence is LearningEvidenceLevel.IN_DISTRIBUTION_IMPROVEMENT
    clean_candidate = _run(suite, "clean-candidate", (1, 1, 1))
    clean_assessment = assess_generalization(
        experiment_id="clean",
        baseline=baseline,
        candidate=clean_candidate,
        suite=suite,
        classes=classes,
        policy=create_generalization_policy("g-clean"),
    )
    clean_evidence = classify_learning_evidence(
        experiment_id="clean",
        baseline=baseline,
        candidate=clean_candidate,
        comparison=compare_capability_runs(baseline, clean_candidate),
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=True,
        training_loss_delta=-0.3,
        held_out_loss_delta=-0.1,
        generalization=clean_assessment,
    )
    assert clean_assessment.status == "GENERALIZATION_EVIDENCE"
    assert clean_evidence.evidence is LearningEvidenceLevel.GENERALIZATION_EVIDENCE
    no_gain = classify_learning_evidence(
        experiment_id="loss-only",
        baseline=baseline,
        candidate=baseline,
        comparison=compare_capability_runs(baseline, baseline),
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=True,
        training_loss_delta=-0.4,
        held_out_loss_delta=-0.2,
        generalization=assessment,
    )
    assert no_gain.evidence is LearningEvidenceLevel.TRAINING_SIGNAL_ONLY


def test_contamination_quarantines_evaluation_and_next_recommendation_is_conservative(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path)
    baseline = _run(suite, "baseline", (0, 0, 0))
    candidate = _run(suite, "candidate", (1, 1, 1))
    findings = detect_contamination(
        experiment_id="contaminated",
        suite=suite,
        training_records={
            "record-1": "Instruction: write a recursive sum\nOutput: sum base case"
        },
    )
    assert {finding.method.value for finding in findings} >= {
        "EXACT",
        "NORMALIZED",
        "NGRAM",
    }
    assessment = assess_generalization(
        experiment_id="contaminated",
        baseline=baseline,
        candidate=candidate,
        suite=suite,
        classes={
            "id": GeneralizationClass.IN_DISTRIBUTION,
            "struct": GeneralizationClass.STRUCTURAL,
            "compose": GeneralizationClass.COMPOSITIONAL,
        },
        policy=create_generalization_policy("g2"),
        quarantined_task_ids=tuple({finding.task_id for finding in findings}),
    )
    attribution = classify_learning_evidence(
        experiment_id="contaminated",
        baseline=baseline,
        candidate=candidate,
        comparison=compare_capability_runs(baseline, candidate),
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=True,
        training_loss_delta=-0.1,
        held_out_loss_delta=-0.1,
        generalization=assessment,
    )
    failure = diagnose_failure(experiment_id="contaminated", attribution=attribution)
    recommendation = next_recommendation(
        experiment=_experiment().model_copy(update={"experiment_id": "contaminated"}),
        attribution=attribution,
        failure=failure,
        expected_value=0,
        confidence=0.9,
    )
    assert assessment.status == "QUARANTINED"
    assert attribution.evidence is LearningEvidenceLevel.CONTRADICTORY_EVIDENCE
    assert failure.diagnosis.value == "CONTAMINATION"
    assert recommendation.action.value == "DEFER"
