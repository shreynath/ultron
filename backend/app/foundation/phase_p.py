"""Phase P: immutable, model-only capability evidence and experiment ranking.

The evaluator invokes ULTRON's local checkpoint directly.  It has no provider,
retrieval, memory, or teacher-answer parameter, so it cannot silently use an
external model to produce candidate answers.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import (
    CheckpointManifest,
    CheckpointStore,
    canonical_hash,
)
from app.foundation.data import ProcessedDocument, normalize_text
from app.foundation.development import EvaluationEvidence, EvaluationMode
from app.foundation.generation import GenerationConfig, UltronGenerator
from app.foundation.model import UltronModelConfig
from app.foundation.tokenizer import UltronTokenizer


class CapabilityDomain(str, Enum):
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    FACTUALITY = "FACTUALITY"
    MATHEMATICS = "MATHEMATICS"
    PROGRAMMING = "PROGRAMMING"
    TECHNICAL_KNOWLEDGE = "TECHNICAL_KNOWLEDGE"
    REASONING = "REASONING"
    INSTRUCTION_FOLLOWING = "INSTRUCTION_FOLLOWING"
    LANGUAGE = "LANGUAGE"
    GENERALIZATION = "GENERALIZATION"
    HALLUCINATION = "HALLUCINATION"
    REGRESSION = "REGRESSION"


class EvaluationPartition(str, Enum):
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    REGRESSION = "REGRESSION"


class ItemRole(str, Enum):
    TARGET = "TARGET"
    GENERALIZATION = "GENERALIZATION"
    REGRESSION = "REGRESSION"


class ItemOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class LearningEvidenceStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    BASELINE_FAILURE = "BASELINE_FAILURE"
    TRAINED = "TRAINED"
    POST_TRAINING_SUCCESS = "POST_TRAINING_SUCCESS"
    GENERALIZED = "GENERALIZED"
    REGRESSION_FREE = "REGRESSION_FREE"
    VALIDATED_LEARNING = "VALIDATED_LEARNING"


class HypothesisOutcome(str, Enum):
    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CapabilityItem(BaseModel):
    """A deterministic task-owned capability item; no model supplies its key."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    domain: CapabilityDomain
    partition: EvaluationPartition
    role: ItemRole = ItemRole.TARGET
    prompt: str = Field(min_length=1, max_length=8_000)
    expected_answer: str = Field(min_length=1, max_length=4_000)
    difficulty: int = Field(ge=1, le=10)
    provenance: dict[str, str]
    contamination_status: str = "CLEAR"
    validation_status: str = "VERIFIED"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CapabilitySuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    version: str = Field(min_length=1, max_length=128)
    partition: EvaluationPartition
    items: tuple[CapabilityItem, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    suite_sha256: str

    def model_post_init(self, __context: Any, /) -> None:
        if not self.items or len({item.task_id for item in self.items}) != len(self.items):
            raise ValueError("A capability suite needs uniquely identified items.")
        if any(item.partition is not self.partition for item in self.items):
            raise ValueError("Every capability item must match the immutable suite partition.")
        if any(item.contamination_status != "CLEAR" or item.validation_status != "VERIFIED" for item in self.items):
            raise ValueError("Suites accept only verified, contamination-clear items.")

    def verify_integrity(self) -> None:
        expected = canonical_hash({
            "suite_id": self.suite_id,
            "version": self.version,
            "partition": self.partition.value,
            "items": [item.model_dump(mode="json") for item in self.items],
        })
        if self.suite_sha256 != expected:
            raise ValueError("Capability suite integrity hash changed; create a new immutable version.")


class CapabilityItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    domain: CapabilityDomain
    partition: EvaluationPartition
    role: ItemRole
    outcome: ItemOutcome
    score: float = Field(ge=0, le=1)
    output: str
    detail: str


class CapabilityRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str
    suite_id: str
    suite_version: str
    suite_sha256: str
    checkpoint_id: str
    checkpoint_path: str
    candidate_id: str
    mode: EvaluationMode = EvaluationMode.BASE_MODEL
    runtime_dependency: str = "NONE_LOCAL_ULTRON_CHECKPOINT_ONLY"
    results: tuple[CapabilityItemResult, ...]
    domain_scores: dict[str, float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def model_post_init(self, __context: Any, /) -> None:
        if not self.results:
            raise ValueError("A capability run needs one or more item results.")
        if len({result.task_id for result in self.results}) != len(self.results):
            raise ValueError("A capability run cannot contain duplicate task results.")
        expected_domains = {result.domain.value for result in self.results}
        if set(self.domain_scores) != expected_domains:
            raise ValueError("Capability run domain scores must cover exactly the observed domains.")


class CapabilityComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    suite_version: str
    baseline_run_id: str
    candidate_run_id: str
    domain_deltas: dict[str, float]
    regression_rate: float
    target_improved: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearningEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    task_id: str
    baseline_run_id: str
    candidate_run_id: str
    status: LearningEvidenceStatus
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FailureCluster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cluster_id: str
    candidate_run_id: str
    domain: CapabilityDomain
    difficulty: int
    task_ids: tuple[str, ...]
    diagnosis: str


class LearningOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opportunity_id: str
    source_run_id: str
    domain: CapabilityDomain
    task_ids: tuple[str, ...]
    importance: float = Field(gt=0, le=1)
    frequency: float = Field(gt=0, le=1)
    expected_improvement: float = Field(ge=0, le=1)
    generalization_potential: float = Field(gt=0, le=1)
    confidence: float = Field(gt=0, le=1)
    estimated_compute_tokens: int = Field(gt=0)
    priority: float
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExperimentHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    opportunity_id: str
    target_domain: CapabilityDomain
    intervention: str
    dataset_ids: tuple[str, ...]
    expected_improvement: str
    maximum_acceptable_regression: float = Field(ge=0, le=1)
    estimated_tokens: int = Field(gt=0)
    rationale: str
    fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExperimentOutcome(BaseModel):
    """Measured result of a hypothesis, not a claim that training succeeded."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    outcome_id: str
    experiment_id: str
    hypothesis_fingerprint: str
    target_domain: CapabilityDomain
    intervention: str
    dataset_ids: tuple[str, ...]
    candidate_run_id: str
    comparison_id: str
    target_delta: float = Field(ge=-1, le=1)
    regression_rate: float = Field(ge=0, le=1)
    tokens_used: int = Field(ge=0)
    status: HypothesisOutcome
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DevelopmentTrace(BaseModel):
    """An immutable explanation of why one bounded cycle took its action."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    trace_id: str
    cycle_id: str
    candidate_id: str
    baseline_run_id: str
    candidate_run_id: str
    comparison_id: str
    selected_opportunity_id: str
    hypothesis_id: str
    dataset_ids: tuple[str, ...]
    training_tokens: int = Field(ge=0)
    decision: str
    rationale: str
    next_experiment_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetEvaluationIsolation(BaseModel):
    """Conservative exact-overlap check between training release and eval truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    suite_id: str
    suite_version: str
    checked_items: int
    contaminated_task_ids: tuple[str, ...]
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TeacherArtifact(BaseModel):
    """Provenance/accounting only; Phase P never calls a teacher itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str
    provider: str
    model: str
    version: str | None
    purpose: str
    source_prompt_sha256: str
    output_sha256: str
    validation_status: str
    output_used_for_training: bool = False
    runtime_required: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PhasePStore:
    _CATEGORIES = frozenset({"suites", "runs", "comparisons", "learning", "clusters", "opportunities", "hypotheses", "outcomes", "traces", "isolation", "teachers"})

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_p"
        for name in self._CATEGORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase P artifact category: {category}")
        path = self.root / category / f"{identifier}.json"
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(f"Immutable Phase P artifact already exists: {identifier}")
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def reads(self, category: str, model: type[BaseModel]) -> tuple[BaseModel, ...]:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase P artifact category: {category}")
        return tuple(model.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted((self.root / category).glob("*.json")))


def create_capability_suite(path: Path, *, suite_id: str, version: str, partition: EvaluationPartition, items: tuple[CapabilityItem, ...]) -> CapabilitySuite:
    if path.exists():
        raise ValueError("Capability suites are immutable; use a new suite/version path.")
    raw = [item.model_dump(mode="json") for item in items]
    suite = CapabilitySuite(suite_id=suite_id, version=version, partition=partition, items=items, suite_sha256=canonical_hash({"suite_id": suite_id, "version": version, "partition": partition.value, "items": raw}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(suite.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return suite


def load_capability_suite(path: Path) -> CapabilitySuite:
    suite = CapabilitySuite.model_validate_json(path.read_text(encoding="utf-8"))
    suite.verify_integrity()
    return suite


def _normal(value: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in value.lower()).split())


class ModelOnlyCapabilityEvaluator:
    """Evaluates only ULTRON checkpoint output using deterministic scoring."""

    def evaluate(self, *, checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig, suite: CapabilitySuite, candidate_id: str, max_new_tokens: int = 32) -> CapabilityRun:
        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("Capability evaluation requires a complete proprietary checkpoint.")
        generator = UltronGenerator(checkpoint, tokenizer, model_config)
        results: list[CapabilityItemResult] = []
        for item in suite.items:
            try:
                generated = generator.generate(item.prompt, GenerationConfig(max_new_tokens=max_new_tokens, temperature=0))
                output = generated.text
                if generated.status.value == "FAILED":
                    outcome, score, detail = ItemOutcome.ERROR, 0.0, f"Local generation failed: {generated.error or 'unspecified error'}"
                else:
                    passed = _normal(output) == _normal(item.expected_answer)
                    outcome, score = (ItemOutcome.PASS, 1.0) if passed else (ItemOutcome.FAIL, 0.0)
                    detail = "Deterministic normalized exact match over local ULTRON output."
            except (OSError, RuntimeError, ValueError) as exc:
                output, outcome, score = "", ItemOutcome.ERROR, 0.0
                detail = f"Local generation could not score this item: {type(exc).__name__}: {exc}"
            results.append(CapabilityItemResult(task_id=item.task_id, domain=item.domain, partition=item.partition, role=item.role, outcome=outcome, score=score, output=output, detail=detail))
        scores: dict[str, float] = {}
        for domain in CapabilityDomain:
            matching = [result.score for result in results if result.domain is domain]
            if matching:
                scores[domain.value] = round(sum(matching) / len(matching), 6)
        return CapabilityRun(run_id=f"{candidate_id}-{suite.suite_id}-{suite.version}-{manifest.checkpoint_id}", suite_id=suite.suite_id, suite_version=suite.version, suite_sha256=suite.suite_sha256, checkpoint_id=manifest.checkpoint_id, checkpoint_path=str(checkpoint.resolve()), candidate_id=candidate_id, results=tuple(results), domain_scores=scores)


def compare_capability_runs(baseline: CapabilityRun, candidate: CapabilityRun) -> CapabilityComparison:
    if (baseline.suite_id, baseline.suite_version, baseline.suite_sha256) != (candidate.suite_id, candidate.suite_version, candidate.suite_sha256):
        raise ValueError("Only results from the exact same immutable suite can be compared.")
    base = {item.task_id: item for item in baseline.results}
    current = {item.task_id: item for item in candidate.results}
    if base.keys() != current.keys():
        raise ValueError("Baseline and candidate item results differ.")
    domains = sorted(set(baseline.domain_scores) | set(candidate.domain_scores))
    deltas = {domain: round(candidate.domain_scores.get(domain, 0) - baseline.domain_scores.get(domain, 0), 6) for domain in domains}
    regressed = sum(current[key].score < base[key].score for key in base)
    target_improved = any(current[key].score > base[key].score and current[key].role is ItemRole.TARGET for key in base)
    return CapabilityComparison(suite_id=baseline.suite_id, suite_version=baseline.suite_version, baseline_run_id=baseline.run_id, candidate_run_id=candidate.run_id, domain_deltas=deltas, regression_rate=round(regressed / max(1, len(base)), 6), target_improved=target_improved)


def phase_n_evidence(run: CapabilityRun, comparison: CapabilityComparison) -> EvaluationEvidence:
    mapping = {
        CapabilityDomain.GENERAL_KNOWLEDGE: "knowledge_accuracy", CapabilityDomain.FACTUALITY: "factuality",
        CapabilityDomain.MATHEMATICS: "mathematical_accuracy", CapabilityDomain.PROGRAMMING: "coding_accuracy",
        CapabilityDomain.REASONING: "reasoning_accuracy", CapabilityDomain.INSTRUCTION_FOLLOWING: "instruction_following",
        CapabilityDomain.GENERALIZATION: "generalization",
    }
    metrics = {metric: run.domain_scores[domain.value] for domain, metric in mapping.items() if domain.value in run.domain_scores}
    metrics["regression_rate"] = comparison.regression_rate
    return EvaluationEvidence(evaluation_id=f"{run.run_id}-phase-n", candidate_id=run.candidate_id, mode=EvaluationMode.BASE_MODEL, suite_id=run.suite_id, suite_version=run.suite_version, suite_sha256=run.suite_sha256, report_sha256=canonical_hash(run.model_dump(mode="json")), metrics=metrics, evaluator_version="PhaseP.ModelOnlyCapabilityEvaluator.v1")


def learning_evidence(baseline: CapabilityRun, candidate: CapabilityRun) -> tuple[LearningEvidence, ...]:
    comparison = compare_capability_runs(baseline, candidate)
    base = {item.task_id: item for item in baseline.results}
    output: list[LearningEvidence] = []
    for item in candidate.results:
        before = base[item.task_id]
        if before.score > 0:
            status, rationale = LearningEvidenceStatus.UNKNOWN, "Baseline did not fail this item."
        elif item.score == 0:
            status, rationale = LearningEvidenceStatus.BASELINE_FAILURE, "Baseline and candidate both failed."
        elif item.role is ItemRole.GENERALIZATION and comparison.regression_rate == 0:
            status, rationale = LearningEvidenceStatus.VALIDATED_LEARNING, "Baseline failure, held-out generalization success, and no item regression."
        elif item.role is ItemRole.GENERALIZATION:
            status, rationale = LearningEvidenceStatus.GENERALIZED, "Baseline failure and held-out generalization success; regression check did not pass."
        elif comparison.regression_rate == 0:
            status, rationale = LearningEvidenceStatus.REGRESSION_FREE, "Baseline failure, post-training success, and no item regression."
        else:
            status, rationale = LearningEvidenceStatus.POST_TRAINING_SUCCESS, "Baseline failure and post-training success; regression remains."
        output.append(LearningEvidence(evidence_id=f"learning-{candidate.run_id}-{item.task_id}", task_id=item.task_id, baseline_run_id=baseline.run_id, candidate_run_id=candidate.run_id, status=status, rationale=rationale))
    return tuple(output)


def cluster_failures(run: CapabilityRun, suite: CapabilitySuite) -> tuple[FailureCluster, ...]:
    by_task = {item.task_id: item for item in suite.items}
    grouped: dict[tuple[CapabilityDomain, int], list[str]] = {}
    for result in run.results:
        if result.outcome is not ItemOutcome.PASS:
            item = by_task[result.task_id]
            grouped.setdefault((item.domain, item.difficulty), []).append(item.task_id)
    return tuple(FailureCluster(cluster_id=f"cluster-{run.run_id}-{domain.value.lower()}-{difficulty}", candidate_run_id=run.run_id, domain=domain, difficulty=difficulty, task_ids=tuple(task_ids), diagnosis="Deterministic task failure cluster; it identifies an intervention target but does not assert a causal mechanism.") for (domain, difficulty), task_ids in sorted(grouped.items(), key=lambda item: (item[0][0].value, item[0][1])))


def prioritize_learning_opportunities(run: CapabilityRun, clusters: tuple[FailureCluster, ...], *, importance: dict[CapabilityDomain, float], expected_improvement: dict[CapabilityDomain, float], generalization_potential: dict[CapabilityDomain, float], confidence: dict[CapabilityDomain, float], estimated_compute_tokens: dict[CapabilityDomain, int]) -> tuple[LearningOpportunity, ...]:
    total = max(1, len(run.results))
    opportunities: list[LearningOpportunity] = []
    for cluster in clusters:
        domain = cluster.domain
        values = (importance.get(domain, 0.5), expected_improvement.get(domain, 0.0), generalization_potential.get(domain, 0.5), confidence.get(domain, 0.1), estimated_compute_tokens.get(domain, 1))
        if not (0 < values[0] <= 1 and 0 <= values[1] <= 1 and 0 < values[2] <= 1 and 0 < values[3] <= 1 and values[4] > 0):
            raise ValueError("Opportunity inputs need normalized importance, expected improvement, generalization potential, confidence, and positive compute cost.")
        frequency = len(cluster.task_ids) / total
        priority = values[0] * frequency * values[1] * values[2] * values[3] / values[4]
        opportunities.append(LearningOpportunity(opportunity_id=f"opportunity-{cluster.cluster_id}", source_run_id=run.run_id, domain=domain, task_ids=cluster.task_ids, importance=values[0], frequency=frequency, expected_improvement=values[1], generalization_potential=values[2], confidence=values[3], estimated_compute_tokens=values[4], priority=priority, rationale="importance × observed failure frequency × expected improvement × generalization potential × confidence ÷ estimated tokens"))
    return tuple(sorted(opportunities, key=lambda item: (-item.priority, item.opportunity_id)))


def experiment_hypothesis(*, experiment_id: str, opportunity: LearningOpportunity, intervention: str, dataset_ids: tuple[str, ...], maximum_acceptable_regression: float) -> ExperimentHypothesis:
    if not dataset_ids:
        raise ValueError("A hypothesis requires one or more immutable eligible dataset IDs.")
    source = {"opportunity_id": opportunity.opportunity_id, "intervention": intervention, "dataset_ids": dataset_ids, "maximum_acceptable_regression": maximum_acceptable_regression, "estimated_tokens": opportunity.estimated_compute_tokens}
    return ExperimentHypothesis(experiment_id=experiment_id, opportunity_id=opportunity.opportunity_id, target_domain=opportunity.domain, intervention=intervention, dataset_ids=dataset_ids, expected_improvement=f"Expected {opportunity.expected_improvement:.1%} improvement in {opportunity.domain.value} under a measured model-only suite.", maximum_acceptable_regression=maximum_acceptable_regression, estimated_tokens=opportunity.estimated_compute_tokens, rationale=opportunity.rationale, fingerprint=canonical_hash(source))


def teacher_dependency_summary(artifacts: tuple[TeacherArtifact, ...]) -> dict[str, int | bool]:
    return {"artifacts": len(artifacts), "training_data_artifacts": sum(item.output_used_for_training for item in artifacts), "runtime_required": any(item.runtime_required for item in artifacts), "unvalidated": sum(item.validation_status != "VALIDATED" for item in artifacts)}


def validate_dataset_isolation(*, release_dir: Path, suite: CapabilitySuite) -> DatasetEvaluationIsolation:
    """Quarantine direct eval prompt/key overlap before a release reaches training.

    This deterministic guard is intentionally conservative: it catches exact
    normalized prompt or answer phrases in the immutable training split.  It
    does not pretend to solve semantic-contamination detection.
    """
    manifest = release_dir / "manifest.json"
    if not manifest.is_file() or not (release_dir / "train.jsonl").is_file():
        raise ValueError("Isolation checks require an immutable release manifest and train split.")
    release_id = str(json.loads(manifest.read_text(encoding="utf-8"))["release_id"])
    train = "\n".join(
        _normal(normalize_text(ProcessedDocument.model_validate_json(line).content))
        for line in (release_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    contaminated = tuple(
        item.task_id
        for item in suite.items
        if _normal(item.prompt) in train or _normal(item.expected_answer) in train
    )
    return DatasetEvaluationIsolation(
        release_id=release_id, suite_id=suite.suite_id, suite_version=suite.version,
        checked_items=len(suite.items), contaminated_task_ids=contaminated,
        status="QUARANTINED" if contaminated else "CLEAR",
    )


class PhasePEngine:
    """Persists model-only capability evidence and bounded experiment proposals.

    It deliberately does not launch training, procure data, invoke a teacher,
    or promote a checkpoint.  Those actions remain governed by the Phase N/O
    control plane and its explicit resource policy.
    """

    def __init__(self, root: Path) -> None:
        self.store = PhasePStore(root)

    def register_suite(self, suite: CapabilitySuite) -> CapabilitySuite:
        suite.verify_integrity()
        self.store.write_immutable("suites", f"{suite.suite_id}-{suite.version}", suite)
        return suite

    def evaluate_model_only(self, *, checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig, suite: CapabilitySuite, candidate_id: str, max_new_tokens: int = 32) -> CapabilityRun:
        self.register_suite(suite)
        run = ModelOnlyCapabilityEvaluator().evaluate(checkpoint=checkpoint, tokenizer=tokenizer, model_config=model_config, suite=suite, candidate_id=candidate_id, max_new_tokens=max_new_tokens)
        self.store.write_immutable("runs", run.run_id, run)
        return run

    def analyze(self, *, baseline: CapabilityRun, candidate: CapabilityRun, suite: CapabilitySuite, importance: dict[CapabilityDomain, float], expected_improvement: dict[CapabilityDomain, float], generalization_potential: dict[CapabilityDomain, float], confidence: dict[CapabilityDomain, float], estimated_compute_tokens: dict[CapabilityDomain, int]) -> tuple[CapabilityComparison, tuple[LearningEvidence, ...], tuple[FailureCluster, ...], tuple[LearningOpportunity, ...]]:
        comparison = compare_capability_runs(baseline, candidate)
        existing = next(
            (
                item
                for item in self.store.reads("comparisons", CapabilityComparison)
                if item.baseline_run_id == comparison.baseline_run_id
                and item.candidate_run_id == comparison.candidate_run_id
                and item.suite_id == comparison.suite_id
                and item.suite_version == comparison.suite_version
            ),
            None,
        )
        if existing is not None:
            comparison = existing
        evidence = learning_evidence(baseline, candidate)
        clusters = cluster_failures(candidate, suite)
        opportunities = prioritize_learning_opportunities(candidate, clusters, importance=importance, expected_improvement=expected_improvement, generalization_potential=generalization_potential, confidence=confidence, estimated_compute_tokens=estimated_compute_tokens)
        self.store.write_immutable("comparisons", f"comparison-{candidate.run_id}", comparison)
        for item in evidence:
            self.store.write_immutable("learning", item.evidence_id, item)
        for item in clusters:
            self.store.write_immutable("clusters", item.cluster_id, item)
        for item in opportunities:
            self.store.write_immutable("opportunities", item.opportunity_id, item)
        return comparison, evidence, clusters, opportunities

    def validate_dataset_isolation(self, *, release_dir: Path, suite: CapabilitySuite) -> DatasetEvaluationIsolation:
        self.register_suite(suite)
        isolation = validate_dataset_isolation(release_dir=release_dir, suite=suite)
        self.store.write_immutable("isolation", f"{isolation.release_id}-{suite.suite_id}-{suite.version}", isolation)
        if isolation.status != "CLEAR":
            raise ValueError(f"Dataset quarantined due to direct evaluation overlap: {', '.join(isolation.contaminated_task_ids)}")
        return isolation

    def propose_experiment(self, *, experiment_id: str, opportunity: LearningOpportunity, intervention: str, dataset_ids: tuple[str, ...], maximum_acceptable_regression: float) -> ExperimentHypothesis:
        hypothesis = experiment_hypothesis(experiment_id=experiment_id, opportunity=opportunity, intervention=intervention, dataset_ids=dataset_ids, maximum_acceptable_regression=maximum_acceptable_regression)
        self.store.write_immutable("hypotheses", hypothesis.experiment_id, hypothesis)
        return hypothesis

    def record_outcome(self, *, hypothesis: ExperimentHypothesis, candidate_run: CapabilityRun, comparison: CapabilityComparison, tokens_used: int) -> ExperimentOutcome:
        target_delta = comparison.domain_deltas.get(hypothesis.target_domain.value, 0.0)
        if target_delta >= 0 and comparison.regression_rate <= hypothesis.maximum_acceptable_regression:
            status = HypothesisOutcome.SUPPORTED if target_delta > 0 else HypothesisOutcome.INCONCLUSIVE
        elif target_delta > 0:
            status = HypothesisOutcome.PARTIALLY_SUPPORTED
        else:
            status = HypothesisOutcome.REFUTED
        outcome = ExperimentOutcome(
            outcome_id=f"outcome-{hypothesis.experiment_id}-{candidate_run.run_id}", experiment_id=hypothesis.experiment_id,
            hypothesis_fingerprint=hypothesis.fingerprint, target_domain=hypothesis.target_domain,
            intervention=hypothesis.intervention, dataset_ids=hypothesis.dataset_ids,
            candidate_run_id=candidate_run.run_id, comparison_id=f"comparison-{candidate_run.run_id}",
            target_delta=target_delta, regression_rate=comparison.regression_rate, tokens_used=tokens_used,
            status=status,
            rationale="Outcome is derived from the matched immutable model-only comparison and configured regression limit.",
        )
        self.store.write_immutable("outcomes", outcome.outcome_id, outcome)
        return outcome

    def select_next_experiment(self, *, opportunities: tuple[LearningOpportunity, ...], dataset_ids: tuple[str, ...], intervention_by_domain: dict[CapabilityDomain, str], maximum_acceptable_regression: float, exploration_fraction: float = 0.3) -> ExperimentHypothesis | None:
        """Choose a non-refuted opportunity using persisted outcome evidence.

        The deterministic tie-breaker makes restart behavior reproducible.  A
        future policy may choose an exploration slot; this foundational selector
        records the configured fraction but does not simulate random success.
        """
        if not 0 <= exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be between zero and one.")
        outcomes = tuple(self.store.reads("outcomes", ExperimentOutcome))
        ranked: list[tuple[float, LearningOpportunity, str]] = []
        for opportunity in opportunities:
            intervention = intervention_by_domain.get(opportunity.domain)
            if not intervention:
                continue
            matching = [item for item in outcomes if item.target_domain is opportunity.domain and item.intervention == intervention and item.dataset_ids == dataset_ids]
            if any(item.status is HypothesisOutcome.REFUTED for item in matching):
                continue
            observed_efficiency = sum(max(0.0, item.target_delta) / max(1, item.tokens_used) for item in matching) / len(matching) if matching else 0.0
            ranked.append((opportunity.priority * (1 + observed_efficiency), opportunity, intervention))
        if not ranked:
            return None
        _, opportunity, intervention = min(ranked, key=lambda item: (-item[0], item[1].opportunity_id, item[2]))
        return self.propose_experiment(experiment_id=f"next-{opportunity.opportunity_id}"[:128], opportunity=opportunity, intervention=intervention, dataset_ids=dataset_ids, maximum_acceptable_regression=maximum_acceptable_regression)

    def write_trace(self, trace: DevelopmentTrace) -> DevelopmentTrace:
        self.store.write_immutable("traces", trace.trace_id, trace)
        return trace
