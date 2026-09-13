from __future__ import annotations

from pathlib import Path

import pytest

from app.foundation.development import EvaluationMode
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    CapabilityItemResult,
    CapabilityRun,
    EvaluationPartition,
    ItemOutcome,
    ItemRole,
    LearningEvidenceStatus,
    PhasePEngine,
    TeacherArtifact,
    compare_capability_runs,
    create_capability_suite,
    learning_evidence,
    load_capability_suite,
    phase_n_evidence,
    teacher_dependency_summary,
)


def _items() -> tuple[CapabilityItem, ...]:
    return (
        CapabilityItem(task_id="reasoning-target", domain=CapabilityDomain.REASONING, partition=EvaluationPartition.TEST, role=ItemRole.TARGET, prompt="What follows?", expected_answer="answer", difficulty=4, provenance={"source": "task-owned fixture"}),
        CapabilityItem(task_id="reasoning-heldout", domain=CapabilityDomain.REASONING, partition=EvaluationPartition.TEST, role=ItemRole.GENERALIZATION, prompt="Infer this rule.", expected_answer="rule", difficulty=6, provenance={"source": "task-owned fixture"}),
        CapabilityItem(task_id="factual-regression", domain=CapabilityDomain.FACTUALITY, partition=EvaluationPartition.TEST, role=ItemRole.REGRESSION, prompt="State the checked fact.", expected_answer="fact", difficulty=3, provenance={"source": "task-owned fixture"}),
    )


def _run(*, candidate_id: str, suite_sha256: str, scores: tuple[float, float, float]) -> CapabilityRun:
    items = _items()
    results = tuple(
        CapabilityItemResult(task_id=item.task_id, domain=item.domain, partition=item.partition, role=item.role, outcome=ItemOutcome.PASS if score else ItemOutcome.FAIL, score=score, output=item.expected_answer if score else "wrong", detail="fixture measurement")
        for item, score in zip(items, scores, strict=True)
    )
    return CapabilityRun(
        run_id=f"{candidate_id}-run", suite_id="capability-fixture", suite_version="v1", suite_sha256=suite_sha256,
        checkpoint_id=f"{candidate_id}-checkpoint", checkpoint_path=f"/local/{candidate_id}", candidate_id=candidate_id,
        mode=EvaluationMode.BASE_MODEL, results=results,
        domain_scores={"REASONING": (scores[0] + scores[1]) / 2, "FACTUALITY": scores[2]},
    )


def test_capability_suite_is_hash_pinned_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    suite = create_capability_suite(path, suite_id="capability-fixture", version="v1", partition=EvaluationPartition.TEST, items=_items())
    assert load_capability_suite(path).suite_sha256 == suite.suite_sha256
    changed = suite.model_copy(update={"items": (suite.items[0].model_copy(update={"expected_answer": "changed"}), *suite.items[1:])})
    path.write_text(changed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity hash"):
        load_capability_suite(path)


def test_comparison_learning_evidence_and_opportunity_are_auditable(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite = create_capability_suite(suite_path, suite_id="capability-fixture", version="v1", partition=EvaluationPartition.TEST, items=_items())
    baseline = _run(candidate_id="baseline", suite_sha256=suite.suite_sha256, scores=(0.0, 0.0, 1.0))
    candidate = _run(candidate_id="candidate", suite_sha256=suite.suite_sha256, scores=(1.0, 1.0, 0.0))

    comparison = compare_capability_runs(baseline, candidate)
    assert comparison.domain_deltas == {"FACTUALITY": -1.0, "REASONING": 1.0}
    assert comparison.regression_rate == pytest.approx(1 / 3)
    evidence = learning_evidence(baseline, candidate)
    assert {item.task_id: item.status for item in evidence} == {
        "reasoning-target": LearningEvidenceStatus.POST_TRAINING_SUCCESS,
        "reasoning-heldout": LearningEvidenceStatus.GENERALIZED,
        "factual-regression": LearningEvidenceStatus.UNKNOWN,
    }
    development_evidence = phase_n_evidence(candidate, comparison)
    assert development_evidence.mode is EvaluationMode.BASE_MODEL
    assert development_evidence.metrics["reasoning_accuracy"] == 1.0
    assert development_evidence.metrics["regression_rate"] == pytest.approx(1 / 3)

    engine = PhasePEngine(tmp_path)
    engine.register_suite(suite)
    _, persisted_evidence, clusters, opportunities = engine.analyze(
        baseline=baseline, candidate=candidate, suite=suite,
        importance={CapabilityDomain.FACTUALITY: 0.8}, expected_improvement={CapabilityDomain.FACTUALITY: 0.25},
        generalization_potential={CapabilityDomain.FACTUALITY: 0.4}, confidence={CapabilityDomain.FACTUALITY: 0.75},
        estimated_compute_tokens={CapabilityDomain.FACTUALITY: 120},
    )
    assert len(persisted_evidence) == 3
    assert clusters[0].task_ids == ("factual-regression",)
    assert opportunities[0].priority > 0
    hypothesis = engine.propose_experiment(experiment_id="phase-p-next", opportunity=opportunities[0], intervention="Add independently validated factuality examples that target the observed cluster.", dataset_ids=("owned-dataset-v2",), maximum_acceptable_regression=0.0)
    assert hypothesis.fingerprint
    assert (tmp_path / "phase_p" / "hypotheses" / "phase-p-next.json").is_file()


def test_teacher_accounting_records_provenance_without_runtime_dependency() -> None:
    artifacts = (
        TeacherArtifact(artifact_id="teacher-1", provider="external-provider", model="teacher-model", version="2026-01", purpose="offline draft only", source_prompt_sha256="a" * 64, output_sha256="b" * 64, validation_status="VALIDATED", output_used_for_training=True),
        TeacherArtifact(artifact_id="teacher-2", provider="external-provider", model="teacher-model", version=None, purpose="rejected draft", source_prompt_sha256="c" * 64, output_sha256="d" * 64, validation_status="REJECTED", runtime_required=False),
    )
    assert teacher_dependency_summary(artifacts) == {
        "artifacts": 2, "training_data_artifacts": 1, "runtime_required": False, "unvalidated": 1,
    }
