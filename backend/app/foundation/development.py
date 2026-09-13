"""Bounded, event-sourced control plane for ULTRON foundation development.

It coordinates local training evidence; it does not change evaluation data,
rewrite checkpoints, or invoke a remote model.
"""
from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash


class CandidateState(str, Enum):
    CANDIDATE = "CANDIDATE"
    TRAINING = "TRAINING"
    RESUMABLE = "RESUMABLE"
    EVALUATING = "EVALUATING"
    VALIDATED = "VALIDATED"
    CHAMPION = "CHAMPION"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class EvaluationMode(str, Enum):
    BASE_MODEL = "BASE_MODEL"
    MODEL_PLUS_MEMORY = "MODEL_PLUS_MEMORY"
    MODEL_PLUS_TOOLS = "MODEL_PLUS_TOOLS"
    FULL_ULTRON = "FULL_ULTRON"


class DevelopmentAction(str, Enum):
    PLAN_TRAINING = "PLAN_TRAINING"
    START_TRAINING = "START_TRAINING"
    COMPLETE_EVALUATION = "COMPLETE_EVALUATION"
    HUMAN_PROMOTION = "HUMAN_PROMOTION"
    STOP_RESOURCE_LIMIT = "STOP_RESOURCE_LIMIT"
    NO_ACTION = "NO_ACTION"


class PromotionPolicy(BaseModel):
    """Immutable boundaries for one development ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    primary_metrics: tuple[str, ...] = (
        "knowledge_accuracy", "reasoning_accuracy", "coding_accuracy",
        "mathematical_accuracy", "factuality", "generalization", "instruction_following",
    )
    required_modes: tuple[EvaluationMode, ...] = (EvaluationMode.BASE_MODEL,)
    maximum_metric_regression: float = Field(default=0.0, ge=0, le=1)
    minimum_improvement: float = Field(default=0.001, gt=0, le=1)
    maximum_hallucination_increase: float = Field(default=0.0, ge=0, le=1)
    maximum_regression_rate_increase: float = Field(default=0.0, ge=0, le=1)
    max_training_tokens_per_experiment: int = Field(default=0, ge=0)
    max_training_tokens_total: int = Field(default=0, ge=0)
    max_active_experiments: int = Field(default=1, ge=1, le=64)
    allow_automatic_training_start: bool = False
    allow_automatic_promotion: bool = False


class CandidatePlan(BaseModel):
    """An immutable, reproducible request to train one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    experiment_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    model_id: str = Field(min_length=1, max_length=256)
    model_version: str = Field(min_length=1, max_length=256)
    dataset_release_id: str = Field(min_length=1, max_length=256)
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokenizer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config: dict[str, Any]
    seed: int
    hardware: dict[str, Any]
    curriculum_stage: str = Field(min_length=1, max_length=128)
    training_token_budget: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=2_000)
    parent_experiment: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def configuration_fingerprint(self) -> str:
        return canonical_hash({
            "model_id": self.model_id, "model_version": self.model_version,
            "dataset_release_id": self.dataset_release_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "training_config": self.training_config, "seed": self.seed,
            "curriculum_stage": self.curriculum_stage,
            "training_token_budget": self.training_token_budget,
        })


class TrainingCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    training_run_id: str = Field(min_length=1, max_length=256)
    checkpoint_id: str = Field(min_length=1, max_length=256)
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tokens_processed: int = Field(ge=0)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recoverable_failure: bool = False
    detail: str = Field(default="", max_length=2_000)


class EvaluationEvidence(BaseModel):
    """Hash-pinned evaluator output; scores are measurements, never claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    evaluation_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    candidate_id: str
    mode: EvaluationMode
    suite_id: str = Field(min_length=1, max_length=256)
    suite_version: str = Field(min_length=1, max_length=128)
    suite_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    metrics: dict[str, float]
    evaluator_version: str = Field(min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any, /) -> None:
        if not self.metrics:
            raise ValueError("Evaluation evidence needs at least one measured metric.")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in self.metrics.values()):
            raise ValueError("Evaluation metrics must be finite values from 0 through 1.")

    @property
    def suite_key(self) -> tuple[EvaluationMode, str, str, str]:
        return (self.mode, self.suite_id, self.suite_version, self.suite_sha256)


class DevelopmentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    decision: str
    rationale: str
    compared_to: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DevelopmentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str
    candidate_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_event_hash: str = "GENESIS"
    event_hash: str = ""

    def sealed(self, previous_event_hash: str) -> DevelopmentEvent:
        raw = self.model_dump(mode="json", exclude={"event_hash", "previous_event_hash"})
        return self.model_copy(update={"previous_event_hash": previous_event_hash, "event_hash": canonical_hash({"previous_event_hash": previous_event_hash, **raw})})


class DevelopmentStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str | None
    champion_id: str | None
    candidates: dict[str, CandidateState]
    training_tokens_recorded: int
    next_action: DevelopmentAction
    reason: str
    event_count: int


class DevelopmentLedger:
    """Append-only hash chain plus immutable policy, plans, and evaluations."""

    def __init__(self, root: Path) -> None:
        self.root = root / "development"
        self.policy_path = self.root / "policy.json"
        self.plans_dir = self.root / "plans"
        self.evaluations_dir = self.root / "evaluations"
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(exist_ok=True)
        self.evaluations_dir.mkdir(exist_ok=True)

    @staticmethod
    def _write_immutable(path: Path, value: BaseModel) -> None:
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"Immutable artifact already exists with different content: {path.name}")
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def configure(self, policy: PromotionPolicy) -> PromotionPolicy:
        with self._lock:
            self._write_immutable(self.policy_path, policy)
        return policy

    def policy(self) -> PromotionPolicy | None:
        return PromotionPolicy.model_validate_json(self.policy_path.read_text(encoding="utf-8")) if self.policy_path.exists() else None

    def events(self) -> tuple[DevelopmentEvent, ...]:
        if not self.events_path.exists():
            return ()
        records = tuple(DevelopmentEvent.model_validate_json(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip())
        previous = "GENESIS"
        for event in records:
            expected = event.sealed(previous)
            if event.previous_event_hash != previous or event.event_hash != expected.event_hash:
                raise ValueError("Development ledger event chain is invalid.")
            previous = event.event_hash
        return records

    def append(self, event_type: str, *, candidate_id: str | None = None, payload: dict[str, Any] | None = None) -> DevelopmentEvent:
        with self._lock:
            events = self.events()
            event = DevelopmentEvent(event_type=event_type, candidate_id=candidate_id, payload=payload or {}).sealed(events[-1].event_hash if events else "GENESIS")
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
            return event

    def write_plan(self, plan: CandidatePlan) -> CandidatePlan:
        with self._lock:
            self._write_immutable(self.plans_dir / f"{plan.candidate_id}.json", plan)
        return plan

    def plan(self, candidate_id: str) -> CandidatePlan:
        path = self.plans_dir / f"{candidate_id}.json"
        if not path.exists():
            raise KeyError(candidate_id)
        return CandidatePlan.model_validate_json(path.read_text(encoding="utf-8"))

    def plans(self) -> tuple[CandidatePlan, ...]:
        return tuple(CandidatePlan.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(self.plans_dir.glob("*.json")))

    def write_evaluation(self, evidence: EvaluationEvidence) -> EvaluationEvidence:
        with self._lock:
            self._write_immutable(self.evaluations_dir / f"{evidence.evaluation_id}.json", evidence)
        return evidence

    def evaluations(self, candidate_id: str | None = None) -> tuple[EvaluationEvidence, ...]:
        evidence = tuple(EvaluationEvidence.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted(self.evaluations_dir.glob("*.json")))
        return tuple(item for item in evidence if candidate_id is None or item.candidate_id == candidate_id)


class DevelopmentOrchestrator:
    """Coordinates bounded local experimentation from persisted evidence.

    It is not a daemon. A service can invoke it repeatedly, but importing it
    cannot launch compute or promote a model.
    """

    def __init__(self, root: Path) -> None:
        self.ledger = DevelopmentLedger(root)

    def configure(self, policy: PromotionPolicy) -> PromotionPolicy:
        return self.ledger.configure(policy)

    def candidate_states(self) -> dict[str, CandidateState]:
        states: dict[str, CandidateState] = {}
        transitions = {
            "CANDIDATE_PLANNED": CandidateState.CANDIDATE,
            "TRAINING_STARTED": CandidateState.TRAINING,
            "TRAINING_COMPLETED": CandidateState.EVALUATING,
            "TRAINING_INTERRUPTED": CandidateState.RESUMABLE,
            "CANDIDATE_VALIDATED": CandidateState.VALIDATED,
            "CANDIDATE_PROMOTED": CandidateState.CHAMPION,
            "CHAMPION_ARCHIVED": CandidateState.ARCHIVED,
            "CANDIDATE_REJECTED": CandidateState.REJECTED,
            "TRAINING_FAILED": CandidateState.FAILED,
        }
        for event in self.ledger.events():
            if event.candidate_id and event.event_type in transitions:
                states[event.candidate_id] = transitions[event.event_type]
        return states

    def _policy(self) -> PromotionPolicy:
        policy = self.ledger.policy()
        if policy is None:
            raise ValueError("Configure an immutable development policy before planning work.")
        return policy

    def plan_candidate(self, plan: CandidatePlan) -> CandidatePlan:
        policy = self._policy()
        if plan.training_token_budget > policy.max_training_tokens_per_experiment:
            raise ValueError("Candidate exceeds the immutable per-experiment training-token limit.")
        if plan.candidate_id in self.candidate_states():
            raise ValueError("Candidate IDs are immutable; choose a new identifier.")
        if plan.configuration_fingerprint in {item.configuration_fingerprint for item in self.ledger.plans()}:
            raise ValueError("An identical configuration is already in the development ledger.")
        self.ledger.write_plan(plan)
        self.ledger.append("CANDIDATE_PLANNED", candidate_id=plan.candidate_id, payload={"experiment_id": plan.experiment_id, "configuration_fingerprint": plan.configuration_fingerprint, "reason": plan.reason})
        return plan

    def _assert_state(self, candidate_id: str, expected: CandidateState) -> CandidatePlan:
        state = self.candidate_states().get(candidate_id)
        if state is not expected:
            raise ValueError(f"Candidate {candidate_id} must be {expected.value}, found {state.value if state else 'UNKNOWN'}.")
        return self.ledger.plan(candidate_id)

    def _recorded_tokens(self) -> int:
        return sum(int(event.payload.get("tokens_processed", 0)) for event in self.ledger.events() if event.event_type == "TRAINING_COMPLETED")

    def _assert_start_within_resources(self, plan: CandidatePlan, policy: PromotionPolicy) -> None:
        states = self.candidate_states()
        active = [candidate_id for candidate_id, state in states.items() if state is CandidateState.TRAINING]
        if len(active) >= policy.max_active_experiments:
            raise RuntimeError("Active-experiment limit reached.")
        reserved = self._recorded_tokens() + sum(self.ledger.plan(candidate_id).training_token_budget for candidate_id in active)
        if reserved + plan.training_token_budget > policy.max_training_tokens_total:
            raise RuntimeError("Total training-token limit would be exceeded.")

    def start_training(self, candidate_id: str, launch: Callable[[CandidatePlan], str]) -> str:
        policy = self._policy()
        plan = self._assert_state(candidate_id, CandidateState.CANDIDATE)
        if not policy.allow_automatic_training_start:
            raise PermissionError("The immutable policy requires an operator to start training outside this orchestrator.")
        self._assert_start_within_resources(plan, policy)
        try:
            local_job_id = launch(plan)
        except Exception as exc:
            self.ledger.append("TRAINING_FAILED", candidate_id=candidate_id, payload={"recoverable": True, "error_type": type(exc).__name__, "detail": str(exc)[:2_000]})
            raise
        if not local_job_id:
            raise ValueError("A training launcher must return a local job identifier.")
        self.ledger.append("TRAINING_STARTED", candidate_id=candidate_id, payload={"local_job_id": local_job_id})
        return local_job_id

    def record_operator_training_start(self, candidate_id: str, *, operator_id: str, local_job_id: str) -> str:
        """Bridge a human-started local job into the persistent lifecycle."""
        plan = self._assert_state(candidate_id, CandidateState.CANDIDATE)
        if not operator_id or not local_job_id:
            raise ValueError("Human-started training needs both an operator and a local job identifier.")
        self._assert_start_within_resources(plan, self._policy())
        self.ledger.append("TRAINING_STARTED", candidate_id=candidate_id, payload={"local_job_id": local_job_id, "operator_id": operator_id})
        return local_job_id

    def record_training_completion(self, completion: TrainingCompletion) -> TrainingCompletion:
        plan = self._assert_state(completion.candidate_id, CandidateState.TRAINING)
        if completion.tokens_processed > plan.training_token_budget:
            raise ValueError("Recorded training tokens exceed the candidate's approved budget.")
        self.ledger.append("TRAINING_COMPLETED", candidate_id=completion.candidate_id, payload=completion.model_dump(mode="json"))
        return completion

    def record_training_interruption(self, candidate_id: str, *, reason: str) -> None:
        """Preserve a recoverable failure without pretending training completed."""
        self._assert_state(candidate_id, CandidateState.TRAINING)
        self.ledger.append("TRAINING_INTERRUPTED", candidate_id=candidate_id, payload={"reason": reason[:2_000]})

    def resume_training(self, candidate_id: str, *, local_job_id: str) -> str:
        """Return an inspected interrupted candidate to the explicit training state."""
        self._assert_state(candidate_id, CandidateState.RESUMABLE)
        if not local_job_id:
            raise ValueError("Resuming training requires a local job identifier.")
        self.ledger.append("TRAINING_STARTED", candidate_id=candidate_id, payload={"local_job_id": local_job_id, "resumed": True})
        return local_job_id

    def record_evaluation(self, evidence: EvaluationEvidence) -> EvaluationEvidence:
        self._assert_state(evidence.candidate_id, CandidateState.EVALUATING)
        if any(item.evaluation_id == evidence.evaluation_id for item in self.ledger.evaluations(evidence.candidate_id)):
            raise ValueError("Evaluation IDs are immutable.")
        self.ledger.write_evaluation(evidence)
        self.ledger.append("EVALUATION_RECORDED", candidate_id=evidence.candidate_id, payload={"evaluation_id": evidence.evaluation_id, "mode": evidence.mode.value, "suite": list(evidence.suite_key)})
        return evidence

    def _champion(self) -> str | None:
        champions = [key for key, value in self.candidate_states().items() if value is CandidateState.CHAMPION]
        if len(champions) > 1:
            raise ValueError("Development ledger has multiple champions; human recovery is required.")
        return champions[0] if champions else None

    @staticmethod
    def _matching(evidence: tuple[EvaluationEvidence, ...], key: tuple[EvaluationMode, str, str, str]) -> EvaluationEvidence | None:
        matches = [item for item in evidence if item.suite_key == key]
        return max(matches, key=lambda item: item.created_at) if matches else None

    def _reject(self, candidate_id: str, rationale: str, compared_to: str | None = None, metrics: dict[str, float] | None = None) -> DevelopmentDecision:
        decision = DevelopmentDecision(candidate_id=candidate_id, decision="REJECTED", rationale=rationale, compared_to=compared_to, metrics=metrics or {})
        self.ledger.append("CANDIDATE_REJECTED", candidate_id=candidate_id, payload=decision.model_dump(mode="json"))
        return decision

    def validate_candidate(self, candidate_id: str) -> DevelopmentDecision:
        policy = self._policy()
        self._assert_state(candidate_id, CandidateState.EVALUATING)
        candidate_evaluations = self.ledger.evaluations(candidate_id)
        missing_modes = [mode.value for mode in policy.required_modes if not any(item.mode is mode for item in candidate_evaluations)]
        if missing_modes:
            return self._reject(candidate_id, f"Missing required evaluation modes: {', '.join(missing_modes)}.")
        champion = self._champion()
        if champion is None:
            return self._reject(candidate_id, "No champion baseline exists; initial champion establishment requires separate human evidence.")
        baseline_evaluations = self.ledger.evaluations(champion)
        deltas: dict[str, float] = {}
        for mode in policy.required_modes:
            candidate = max((item for item in candidate_evaluations if item.mode is mode), key=lambda item: item.created_at)
            baseline = self._matching(baseline_evaluations, candidate.suite_key)
            if baseline is None:
                return self._reject(candidate_id, f"Champion lacks a matching immutable {mode.value} evaluation suite.")
            missing = [metric for metric in policy.primary_metrics if metric not in candidate.metrics or metric not in baseline.metrics]
            if missing:
                return self._reject(candidate_id, f"Missing required metrics on matched suite: {', '.join(missing)}.")
            for metric in policy.primary_metrics:
                delta = candidate.metrics[metric] - baseline.metrics[metric]
                deltas[f"{mode.value}:{metric}"] = delta
                if delta < -policy.maximum_metric_regression:
                    return self._reject(candidate_id, f"{metric} regressed by {abs(delta):.4f}, above the configured tolerance.", champion, deltas)
            for metric, maximum in (("hallucination_rate", policy.maximum_hallucination_increase), ("regression_rate", policy.maximum_regression_rate_increase)):
                if metric in candidate.metrics and metric in baseline.metrics:
                    delta = candidate.metrics[metric] - baseline.metrics[metric]
                    deltas[f"{mode.value}:{metric}"] = delta
                    if delta > maximum:
                        return self._reject(candidate_id, f"{metric} increased by {delta:.4f}, above the configured tolerance.", champion, deltas)
        primary = [value for name, value in deltas.items() if not name.endswith(("hallucination_rate", "regression_rate"))]
        if not any(value >= policy.minimum_improvement for value in primary):
            return self._reject(candidate_id, "Candidate did not meet the configured measurable improvement threshold.", champion, deltas)
        decision = DevelopmentDecision(candidate_id=candidate_id, decision="VALIDATED", rationale="Candidate met the immutable matched-suite policy; promotion remains separate.", compared_to=champion, metrics=deltas)
        self.ledger.append("CANDIDATE_VALIDATED", candidate_id=candidate_id, payload=decision.model_dump(mode="json"))
        return decision

    def establish_initial_champion(self, candidate_id: str, *, operator_id: str) -> DevelopmentDecision:
        """Record an operator-reviewed bootstrap baseline without fabricating comparison data.

        A fresh ledger has no champion to compare against.  Establishing that
        first baseline is therefore intentionally a distinct, identified human
        decision, and still requires every policy-required evaluation mode and
        metric to be present in immutable evidence.
        """
        if not operator_id:
            raise PermissionError("Initial champion establishment requires an identified human operator.")
        policy = self._policy()
        self._assert_state(candidate_id, CandidateState.EVALUATING)
        evidence = self.ledger.evaluations(candidate_id)
        for mode in policy.required_modes:
            reports = [item for item in evidence if item.mode is mode]
            if not reports:
                raise ValueError(f"Initial champion lacks required {mode.value} evidence.")
            latest = max(reports, key=lambda item: item.created_at)
            missing = [metric for metric in policy.primary_metrics if metric not in latest.metrics]
            if missing:
                raise ValueError(f"Initial champion evidence lacks required metrics: {', '.join(missing)}.")
        if self._champion() is not None:
            raise ValueError("A champion already exists; use normal candidate validation.")
        decision = DevelopmentDecision(candidate_id=candidate_id, decision="VALIDATED_INITIAL_BASELINE", rationale="Operator-reviewed initial baseline with required immutable evaluation evidence.")
        self.ledger.append("CANDIDATE_VALIDATED", candidate_id=candidate_id, payload={**decision.model_dump(mode="json"), "operator_id": operator_id})
        return self.promote(candidate_id, operator_id=operator_id)

    def promote(self, candidate_id: str, *, operator_id: str | None = None) -> DevelopmentDecision:
        policy = self._policy()
        self._assert_state(candidate_id, CandidateState.VALIDATED)
        if not policy.allow_automatic_promotion and not operator_id:
            raise PermissionError("Promotion requires an identified human operator under the immutable policy.")
        prior = self._champion()
        decision = DevelopmentDecision(candidate_id=candidate_id, decision="CHAMPION", rationale="Validated candidate promoted with auditable authorization.", compared_to=prior)
        if prior:
            self.ledger.append("CHAMPION_ARCHIVED", candidate_id=prior, payload={"replacement": candidate_id, "operator_id": operator_id or "AUTOMATED_POLICY"})
        self.ledger.append("CANDIDATE_PROMOTED", candidate_id=candidate_id, payload={**decision.model_dump(mode="json"), "operator_id": operator_id or "AUTOMATED_POLICY"})
        return decision

    def status(self) -> DevelopmentStatus:
        policy = self.ledger.policy()
        states, count = self.candidate_states(), len(self.ledger.events())
        if policy is None:
            return DevelopmentStatus(policy_id=None, champion_id=None, candidates=states, training_tokens_recorded=0, next_action=DevelopmentAction.NO_ACTION, reason="No immutable development policy is configured.", event_count=count)
        tokens = self._recorded_tokens()
        if any(state is CandidateState.EVALUATING for state in states.values()):
            action, reason = DevelopmentAction.COMPLETE_EVALUATION, "A completed candidate needs immutable evaluation evidence."
        elif any(state is CandidateState.VALIDATED for state in states.values()):
            action, reason = DevelopmentAction.HUMAN_PROMOTION, "A validated candidate requires identified human promotion under the policy."
        elif tokens >= policy.max_training_tokens_total:
            action, reason = DevelopmentAction.STOP_RESOURCE_LIMIT, "The configured total training-token limit has been reached."
        elif any(state is CandidateState.TRAINING for state in states.values()):
            action, reason = DevelopmentAction.NO_ACTION, "A bounded local training job is currently active."
        elif any(state is CandidateState.RESUMABLE for state in states.values()):
            action, reason = DevelopmentAction.START_TRAINING, "A recoverable candidate has a persisted checkpoint and requires an explicit resume decision."
        elif any(state is CandidateState.CANDIDATE for state in states.values()):
            action = DevelopmentAction.START_TRAINING if policy.allow_automatic_training_start else DevelopmentAction.PLAN_TRAINING
            reason = "A candidate is ready within policy." if policy.allow_automatic_training_start else "A candidate is planned, but policy requires an operator to start training."
        else:
            action, reason = DevelopmentAction.PLAN_TRAINING, "No active candidate exists; select a new intervention from measured weaknesses."
        return DevelopmentStatus(policy_id=policy.policy_id, champion_id=self._champion(), candidates=states, training_tokens_recorded=tokens, next_action=action, reason=reason, event_count=count)
