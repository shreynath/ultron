from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.foundation.development import (
    CandidatePlan,
    CandidateState,
    DevelopmentAction,
    DevelopmentOrchestrator,
    EvaluationEvidence,
    EvaluationMode,
    PromotionPolicy,
    TrainingCompletion,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _policy(*, automatic_start: bool = True) -> PromotionPolicy:
    return PromotionPolicy(
        policy_id="development-test-v1",
        primary_metrics=("knowledge_accuracy", "reasoning_accuracy"),
        required_modes=(EvaluationMode.BASE_MODEL,),
        max_training_tokens_per_experiment=100,
        max_training_tokens_total=200,
        allow_automatic_training_start=automatic_start,
    )


def _plan(candidate_id: str, *, seed: int) -> CandidatePlan:
    return CandidatePlan(
        candidate_id=candidate_id,
        experiment_id=f"experiment-{candidate_id}",
        model_id="ultron-test",
        model_version="v1",
        dataset_release_id="release-test",
        dataset_manifest_sha256=_digest("dataset"),
        tokenizer_sha256=_digest("tokenizer"),
        training_config={"learning_rate": 0.0001},
        seed=seed,
        hardware={"machine": "test"},
        curriculum_stage="reasoning",
        training_token_budget=100,
        reason="Controlled matched-suite experiment.",
    )


def _completion(candidate_id: str) -> TrainingCompletion:
    return TrainingCompletion(
        candidate_id=candidate_id,
        training_run_id=f"run-{candidate_id}",
        checkpoint_id=f"checkpoint-{candidate_id}",
        checkpoint_sha256=_digest(f"checkpoint-{candidate_id}"),
        tokens_processed=100,
    )


def _evidence(candidate_id: str, *, evaluation_id: str, knowledge: float, reasoning: float, hallucination: float = 0.1) -> EvaluationEvidence:
    return EvaluationEvidence(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        mode=EvaluationMode.BASE_MODEL,
        suite_id="intrinsic-capability",
        suite_version="v1",
        suite_sha256=_digest("suite"),
        report_sha256=_digest(evaluation_id),
        metrics={
            "knowledge_accuracy": knowledge,
            "reasoning_accuracy": reasoning,
            "hallucination_rate": hallucination,
            "regression_rate": 0.0,
        },
        evaluator_version="task-owned-v1",
    )


def _train(orchestrator: DevelopmentOrchestrator, candidate_id: str) -> None:
    assert orchestrator.start_training(candidate_id, lambda plan: f"local-{plan.candidate_id}") == f"local-{candidate_id}"
    orchestrator.record_training_completion(_completion(candidate_id))


def _bootstrap(orchestrator: DevelopmentOrchestrator) -> None:
    orchestrator.plan_candidate(_plan("champion-a", seed=1))
    _train(orchestrator, "champion-a")
    orchestrator.record_evaluation(_evidence("champion-a", evaluation_id="champion-eval", knowledge=0.5, reasoning=0.5))
    promoted = orchestrator.establish_initial_champion("champion-a", operator_id="operator-1")
    assert promoted.decision == "CHAMPION"


def test_development_ledger_requires_policy_and_records_auditable_lifecycle(tmp_path: Path) -> None:
    orchestrator = DevelopmentOrchestrator(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        orchestrator.plan_candidate(_plan("candidate-a", seed=1))

    orchestrator.configure(_policy())
    _bootstrap(orchestrator)
    orchestrator.plan_candidate(_plan("candidate-b", seed=2))
    _train(orchestrator, "candidate-b")
    orchestrator.record_evaluation(_evidence("candidate-b", evaluation_id="candidate-eval", knowledge=0.6, reasoning=0.55, hallucination=0.05))
    validated = orchestrator.validate_candidate("candidate-b")
    assert validated.decision == "VALIDATED"
    assert orchestrator.candidate_states()["candidate-b"] is CandidateState.VALIDATED
    assert orchestrator.status().next_action is DevelopmentAction.HUMAN_PROMOTION

    promoted = orchestrator.promote("candidate-b", operator_id="operator-2")
    assert promoted.decision == "CHAMPION"
    assert orchestrator.status().champion_id == "candidate-b"
    assert (tmp_path / "development" / "plans" / "candidate-b.json").exists()
    assert len(orchestrator.ledger.events()) >= 10


def test_development_policy_refuses_a_zero_capability_improvement_threshold() -> None:
    with pytest.raises(ValueError):
        PromotionPolicy(policy_id="no-zero-improvement", minimum_improvement=0)


def test_development_orchestrator_rejects_regressions_and_duplicate_trials(tmp_path: Path) -> None:
    orchestrator = DevelopmentOrchestrator(tmp_path)
    orchestrator.configure(_policy())
    _bootstrap(orchestrator)

    duplicate = _plan("duplicate", seed=1)
    with pytest.raises(ValueError, match="identical configuration"):
        orchestrator.plan_candidate(duplicate)

    orchestrator.plan_candidate(_plan("candidate-b", seed=2))
    _train(orchestrator, "candidate-b")
    orchestrator.record_evaluation(_evidence("candidate-b", evaluation_id="candidate-eval", knowledge=0.4, reasoning=0.9))
    rejected = orchestrator.validate_candidate("candidate-b")
    assert rejected.decision == "REJECTED"
    assert "regressed" in rejected.rationale
    assert orchestrator.candidate_states()["candidate-b"] is CandidateState.REJECTED


def test_policy_remains_immutable_and_cannot_start_training_without_authorization(tmp_path: Path) -> None:
    orchestrator = DevelopmentOrchestrator(tmp_path)
    orchestrator.configure(_policy(automatic_start=False))
    with pytest.raises(ValueError, match="different content"):
        orchestrator.configure(_policy(automatic_start=True))
    orchestrator.plan_candidate(_plan("candidate-a", seed=1))
    with pytest.raises(PermissionError, match="operator"):
        orchestrator.start_training("candidate-a", lambda _: "unexpected")
    assert orchestrator.record_operator_training_start("candidate-a", operator_id="operator-1", local_job_id="local-candidate-a") == "local-candidate-a"
    assert orchestrator.candidate_states()["candidate-a"] is CandidateState.TRAINING
    assert orchestrator.status().next_action is DevelopmentAction.NO_ACTION


def test_active_training_reserves_its_full_token_budget(tmp_path: Path) -> None:
    orchestrator = DevelopmentOrchestrator(tmp_path)
    policy = _policy().model_copy(update={"max_training_tokens_total": 150, "max_active_experiments": 2})
    orchestrator.configure(policy)
    orchestrator.plan_candidate(_plan("candidate-a", seed=1))
    orchestrator.plan_candidate(_plan("candidate-b", seed=2))
    orchestrator.start_training("candidate-a", lambda _: "local-a")
    with pytest.raises(RuntimeError, match="Total training-token"):
        orchestrator.start_training("candidate-b", lambda _: "local-b")
