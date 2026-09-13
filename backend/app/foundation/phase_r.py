"""Phase R: immutable learning-attribution and generalization evidence.

Phase R is an analysis and experiment-design layer over Phase O training,
Phase P model-only evaluations, and Phase Q immutable data releases.  It does
not execute data, call a teacher, or promote a model.  In particular, it keeps
"weights changed" distinct from evidence that a measured capability changed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash
from app.foundation.data import normalize_text
from app.foundation.phase_p import (
    CapabilityComparison,
    CapabilityRun,
    CapabilitySuite,
)


class ExperimentStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    ANALYZED = "ANALYZED"


class GeneralizationClass(str, Enum):
    IN_DISTRIBUTION = "IN_DISTRIBUTION"
    STRUCTURAL = "STRUCTURAL"
    COMPOSITIONAL = "COMPOSITIONAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    ADVERSARIAL = "ADVERSARIAL"


class OverlapMethod(str, Enum):
    EXACT = "EXACT"
    NORMALIZED = "NORMALIZED"
    NGRAM = "NGRAM"
    STRUCTURAL = "STRUCTURAL"


class LearningEvidenceLevel(str, Enum):
    NO_LEARNING_EVIDENCE = "NO_LEARNING_EVIDENCE"
    TRAINING_SIGNAL_ONLY = "TRAINING_SIGNAL_ONLY"
    IN_DISTRIBUTION_IMPROVEMENT = "IN_DISTRIBUTION_IMPROVEMENT"
    GENERALIZATION_EVIDENCE = "GENERALIZATION_EVIDENCE"
    STRONG_LEARNING_EVIDENCE = "STRONG_LEARNING_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FailureDiagnosis(str, Enum):
    NO_WEIGHT_CHANGE = "NO_WEIGHT_CHANGE"
    TRAINING_INSTABILITY = "TRAINING_INSTABILITY"
    NO_HELD_OUT_GAIN = "NO_HELD_OUT_GAIN"
    NO_CAPABILITY_GAIN = "NO_CAPABILITY_GAIN"
    MEMORIZATION = "MEMORIZATION"
    CONTAMINATION = "CONTAMINATION"
    REGRESSION = "REGRESSION"
    DATA_QUALITY = "DATA_QUALITY"
    CURRICULUM_FAILURE = "CURRICULUM_FAILURE"
    INSUFFICIENT_COMPUTE = "INSUFFICIENT_COMPUTE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    EVALUATION_LIMITATION = "EVALUATION_LIMITATION"


class DatasetValueStatus(str, Enum):
    CHAMPION = "CHAMPION"
    USEFUL = "USEFUL"
    NEUTRAL = "NEUTRAL"
    LOW_VALUE = "LOW_VALUE"
    HARMFUL = "HARMFUL"
    QUARANTINED = "QUARANTINED"
    RETIRED = "RETIRED"
    DOMINATED = "DOMINATED"


class RecommendationAction(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REPLICATE = "REPLICATE"
    ACQUIRE_MORE_DATA = "ACQUIRE_MORE_DATA"
    CHANGE_CURRICULUM = "CHANGE_CURRICULUM"
    DEFER = "DEFER"


class ExperimentPolicy(BaseModel):
    """Bounded automatic selection policy; Phase N remains the execution gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    version: str = "phase-r-v1"
    maximum_experiments: int = Field(default=4, ge=1)
    maximum_datasets_per_experiment: int = Field(default=3, ge=1)
    maximum_training_tokens: int = Field(default=1_000_000, ge=1)
    maximum_compute_units: float = Field(default=1_000_000_000.0, gt=0)
    maximum_wall_seconds: float = Field(default=86_400, gt=0)
    maximum_evaluation_items: int = Field(default=10_000, ge=1)
    maximum_concurrent_jobs: int = Field(default=1, ge=1)
    minimum_generalization_delta: float = Field(default=0.01, ge=0, le=1)
    maximum_regression_rate: float = Field(default=0.0, ge=0, le=1)
    require_replications_for_strong_evidence: int = Field(default=2, ge=2)
    fingerprint: str

    def verify_integrity(self) -> None:
        expected = canonical_hash(self.model_dump(mode="json", exclude={"fingerprint"}))
        if expected != self.fingerprint:
            raise ValueError(
                "Experiment policy fingerprint changed; create a new immutable policy."
            )


def create_experiment_policy(policy_id: str, **values: object) -> ExperimentPolicy:
    unsigned = ExperimentPolicy(policy_id=policy_id, **values, fingerprint="")
    return unsigned.model_copy(
        update={
            "fingerprint": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"fingerprint"})
            )
        }
    )


class LearningExperiment(BaseModel):
    """Immutable manifest that joins one controlled intervention to O/P/Q facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    parent_model_id: str
    parent_checkpoint_sha256: str
    candidate_model_id: str
    candidate_checkpoint_sha256: str | None = None
    target_id: str
    target_domain: str
    dataset_release_ids: tuple[str, ...]
    dataset_release_hashes: tuple[str, ...]
    curriculum_id: str | None = None
    curriculum_ordering_hash: str | None = None
    baseline_evaluation_ids: tuple[str, ...]
    candidate_evaluation_ids: tuple[str, ...] = ()
    training_config_sha256: str
    evaluation_config_sha256: str
    analysis_policy_id: str
    analysis_policy_fingerprint: str
    seed: int
    training_tokens: int = Field(ge=0)
    optimizer_steps: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)
    compute_units: float = Field(ge=0)
    runtime: str
    software_version: str
    git_commit: str = "UNKNOWN"
    distributed_world_size: int = Field(default=1, ge=1)
    status: ExperimentStatus = ExperimentStatus.PLANNED
    decision: str = "PENDING_PHASE_N_GATE"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str

    def model_post_init(self, __context: object, /) -> None:
        if not self.dataset_release_ids or len(self.dataset_release_ids) != len(
            self.dataset_release_hashes
        ):
            raise ValueError(
                "An experiment requires aligned immutable dataset release IDs and hashes."
            )
        if not self.baseline_evaluation_ids:
            raise ValueError(
                "An experiment requires a measured baseline evaluation ID."
            )

    def verify_integrity(self) -> None:
        payload = self.model_dump(
            mode="json", exclude={"manifest_sha256", "created_at"}
        )
        if canonical_hash(payload) != self.manifest_sha256:
            raise ValueError("Experiment manifest changed; create a new experiment ID.")


def create_experiment(**values: object) -> LearningExperiment:
    raw = dict(values)
    raw.setdefault("status", ExperimentStatus.PLANNED)
    raw.setdefault("decision", "PENDING_PHASE_N_GATE")
    unsigned = LearningExperiment(**raw, manifest_sha256="")
    return unsigned.model_copy(
        update={
            "manifest_sha256": canonical_hash(
                unsigned.model_dump(
                    mode="json", exclude={"manifest_sha256", "created_at"}
                )
            )
        }
    )


class GeneralizationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    version: str = "phase-r-generalization-v1"
    minimum_class_items: int = Field(default=1, ge=1)
    minimum_delta: float = Field(default=0.01, ge=0, le=1)
    require_all_nonempty_classes: bool = True
    fingerprint: str


def create_generalization_policy(
    policy_id: str, **values: object
) -> GeneralizationPolicy:
    unsigned = GeneralizationPolicy(policy_id=policy_id, **values, fingerprint="")
    return unsigned.model_copy(
        update={
            "fingerprint": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"fingerprint"})
            )
        }
    )


class GeneralizationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    assessment_id: str
    experiment_id: str
    policy_id: str
    policy_fingerprint: str
    baseline_run_ids: tuple[str, ...]
    candidate_run_ids: tuple[str, ...]
    suite_hashes: tuple[str, ...]
    class_counts: dict[str, int]
    class_deltas: dict[str, float | None]
    quarantined_task_ids: tuple[str, ...] = ()
    generalization_score: float | None
    status: str
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContaminationFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    finding_id: str
    experiment_id: str
    task_id: str
    training_record_id: str
    method: OverlapMethod
    score: float = Field(ge=0, le=1)
    detector_version: str
    explanation: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearningAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attribution_id: str
    experiment_id: str
    baseline_run_id: str
    candidate_run_id: str
    weights_changed: bool
    training_loss_delta: float | None = None
    held_out_loss_delta: float | None = None
    target_capability_delta: float = Field(ge=-1, le=1)
    generalization_delta: float | None = Field(default=None, ge=-1, le=1)
    regression_rate: float = Field(ge=0, le=1)
    contaminated_task_ids: tuple[str, ...] = ()
    evidence: LearningEvidenceLevel
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attribution_id: str
    experiment_id: str
    dataset_release_id: str
    full_mixture_delta: float
    without_dataset_delta: float | None
    observed_marginal_delta: float | None
    status: str
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EfficiencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    efficiency_id: str
    experiment_id: str
    capability_gain_per_token: float | None
    generalization_gain_per_token: float | None
    capability_gain_per_step: float | None
    capability_gain_per_compute: float | None
    tokens_per_newly_solved_item: float | None
    compute_per_newly_solved_item: float | None
    training_tokens: int
    optimizer_steps: int
    compute_units: float
    newly_solved_items: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DataValueRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value_id: str
    dataset_release_id: str
    experiment_id: str
    capability_contribution: float = Field(ge=-1, le=1)
    generalization_contribution: float = Field(ge=-1, le=1)
    target_importance: float = Field(gt=0, le=1)
    evidence_confidence: float = Field(ge=0, le=1)
    novelty: float = Field(ge=0, le=1)
    source_reliability: float = Field(ge=0, le=1)
    contamination_risk: float = Field(ge=0, le=1)
    token_cost: int = Field(ge=0)
    compute_cost: float = Field(ge=0)
    value_score: float
    status: DatasetValueStatus
    formula_version: str = "phase-r-data-value-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExperimentFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    failure_id: str
    experiment_id: str
    diagnosis: FailureDiagnosis
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NextExperimentRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recommendation_id: str
    experiment_id: str
    target_id: str
    observed: str
    likely_cause: str
    action: RecommendationAction
    data_requirement: str
    preferred_source: str
    expected_value: float
    confidence: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChampionDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    champion_id: str
    dataset_release_id: str
    previous_champion_id: str | None
    experiment_ids: tuple[str, ...]
    release_hashes: tuple[str, ...]
    evidence: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearningCurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    training_tokens: int = Field(ge=0)
    capability_score: float = Field(ge=0, le=1)
    generalization_score: float | None = Field(default=None, ge=0, le=1)


class ExperimentDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    design_id: str
    target_id: str
    kind: str
    dataset_release_ids: tuple[str, ...]
    estimated_tokens: int
    estimated_compute_units: float
    rationale: str


class PhaseRStore:
    _CATEGORIES = frozenset(
        {
            "policies",
            "experiments",
            "contamination",
            "generalization",
            "learning",
            "dataset_attribution",
            "efficiency",
            "data_value",
            "failures",
            "recommendations",
            "champions",
            "curves",
            "designs",
        }
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_r"
        for category in self._CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase R artifact category: {category}")
        path = self.root / category / f"{identifier}.json"
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    f"Immutable Phase R artifact already exists: {identifier}"
                )
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def reads(self, category: str, model: type[BaseModel]) -> tuple[BaseModel, ...]:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase R artifact category: {category}")
        return tuple(
            model.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / category).glob("*.json"))
        )


def _normalized(value: str) -> str:
    return normalize_text(value).lower()


def _ngrams(value: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", _normalized(value))
    return {
        tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))
    }


def _structure(value: str) -> str:
    """A transparent number-insensitive template detector, not semantics.

    Replacing every word with ``w`` made unrelated ordinary sentences collide.
    This deliberately narrow structural pass instead catches repeated templates
    whose variable is numeric (for example, exercise 7 versus exercise 12).
    """
    return re.sub(r"\b\d+\b", "#", _normalized(value))


def detect_contamination(
    *, experiment_id: str, suite: CapabilitySuite, training_records: dict[str, str]
) -> tuple[ContaminationFinding, ...]:
    """Quarantine direct, normalized, n-gram and template overlap deterministically.

    This does not claim semantic-contamination coverage; detector provenance is
    retained so later stronger detectors can create new findings.
    """
    findings: list[ContaminationFinding] = []
    for item in suite.items:
        prompt, answer = item.prompt, item.expected_answer
        for record_id, training in training_records.items():
            checks = (
                (
                    OverlapMethod.EXACT,
                    prompt in training or answer in training,
                    1.0,
                    "Evaluation prompt or key occurs verbatim in training record.",
                ),
                (
                    OverlapMethod.NORMALIZED,
                    _normalized(prompt) in _normalized(training)
                    or _normalized(answer) in _normalized(training),
                    1.0,
                    "Normalized evaluation prompt or key occurs in training record.",
                ),
                (
                    OverlapMethod.NGRAM,
                    bool((_ngrams(prompt) | _ngrams(answer)) & _ngrams(training)),
                    0.5,
                    "At least one normalized 3-gram overlaps; conservative review required.",
                ),
                (
                    OverlapMethod.STRUCTURAL,
                    len(_structure(prompt).split()) >= 4
                    and _structure(prompt) in _structure(training),
                    0.25,
                    "Lexical prompt template overlaps; this is not a semantic detector.",
                ),
            )
            for method, matched, score, explanation in checks:
                if matched:
                    findings.append(
                        ContaminationFinding(
                            finding_id=f"contamination-{experiment_id}-{item.task_id}-{record_id}-{method.value.lower()}"[
                                :128
                            ],
                            experiment_id=experiment_id,
                            task_id=item.task_id,
                            training_record_id=record_id,
                            method=method,
                            score=score,
                            detector_version="PhaseR.Overlap.v1",
                            explanation=explanation,
                        )
                    )
    return tuple(findings)


def assess_generalization(
    *,
    experiment_id: str,
    baseline: CapabilityRun,
    candidate: CapabilityRun,
    suite: CapabilitySuite,
    classes: dict[str, GeneralizationClass],
    policy: GeneralizationPolicy,
    quarantined_task_ids: Iterable[str] = (),
) -> GeneralizationAssessment:
    if (baseline.suite_id, baseline.suite_version, baseline.suite_sha256) != (
        candidate.suite_id,
        candidate.suite_version,
        candidate.suite_sha256,
    ):
        raise ValueError(
            "Generalization requires baseline and candidate results from the exact same immutable suite."
        )
    if suite.suite_sha256 != baseline.suite_sha256:
        raise ValueError("Generalization suite does not match model-only results.")
    policy_fingerprint = policy.fingerprint
    if (
        canonical_hash(policy.model_dump(mode="json", exclude={"fingerprint"}))
        != policy_fingerprint
    ):
        raise ValueError("Generalization policy fingerprint does not verify.")
    quarantine = set(quarantined_task_ids)
    before, after = (
        {item.task_id: item for item in run.results} for run in (baseline, candidate)
    )
    unknown = set(classes) - set(before)
    if unknown:
        raise ValueError(f"Unknown generalization task IDs: {sorted(unknown)}")
    counts: dict[str, int] = {kind.value: 0 for kind in GeneralizationClass}
    sums: dict[str, float] = {kind.value: 0.0 for kind in GeneralizationClass}
    for task_id, kind in classes.items():
        if task_id in quarantine:
            continue
        counts[kind.value] += 1
        sums[kind.value] += after[task_id].score - before[task_id].score
    deltas = {
        kind.value: (
            round(sums[kind.value] / counts[kind.value], 6)
            if counts[kind.value]
            else None
        )
        for kind in GeneralizationClass
    }
    non_id = [
        kind
        for kind in GeneralizationClass
        if kind is not GeneralizationClass.IN_DISTRIBUTION
        and counts[kind.value] >= policy.minimum_class_items
    ]
    values = [deltas[kind.value] for kind in non_id if deltas[kind.value] is not None]
    score = round(sum(values) / len(values), 6) if values else None
    missing = [
        kind.value
        for kind in GeneralizationClass
        if kind is not GeneralizationClass.IN_DISTRIBUTION
        and counts[kind.value]
        and counts[kind.value] < policy.minimum_class_items
    ]
    if quarantine:
        status, rationale = (
            "QUARANTINED",
            "Contaminated items were excluded; this result cannot receive generalization credit.",
        )
    elif not values or (policy.require_all_nonempty_classes and missing):
        status, rationale = (
            "INSUFFICIENT_EVIDENCE",
            "No adequately sized independent generalization class was available.",
        )
    elif all(value >= policy.minimum_delta for value in values):
        status, rationale = (
            "GENERALIZATION_EVIDENCE",
            "Every evaluated independent generalization class met the immutable delta threshold.",
        )
    else:
        status, rationale = (
            "NO_GENERALIZATION_EVIDENCE",
            "At least one evaluated independent generalization class did not meet the immutable delta threshold.",
        )
    return GeneralizationAssessment(
        assessment_id=f"generalization-{experiment_id}",
        experiment_id=experiment_id,
        policy_id=policy.policy_id,
        policy_fingerprint=policy_fingerprint,
        baseline_run_ids=(baseline.run_id,),
        candidate_run_ids=(candidate.run_id,),
        suite_hashes=(suite.suite_sha256,),
        class_counts=counts,
        class_deltas=deltas,
        quarantined_task_ids=tuple(sorted(quarantine)),
        generalization_score=score,
        status=status,
        rationale=rationale,
    )


def classify_learning_evidence(
    *,
    experiment_id: str,
    baseline: CapabilityRun,
    candidate: CapabilityRun,
    comparison: CapabilityComparison,
    target_domain: str,
    weights_changed: bool,
    training_loss_delta: float | None,
    held_out_loss_delta: float | None,
    generalization: GeneralizationAssessment,
    replications: int = 1,
) -> LearningAttribution:
    target_delta = comparison.domain_deltas.get(target_domain, 0.0)
    generalization_delta = generalization.generalization_score
    contamination = generalization.quarantined_task_ids
    if contamination:
        evidence, rationale = (
            LearningEvidenceLevel.CONTRADICTORY_EVIDENCE,
            "Evaluation contamination was detected; no capability or generalization claim is allowed.",
        )
    elif comparison.regression_rate > 0:
        evidence, rationale = (
            LearningEvidenceLevel.CONTRADICTORY_EVIDENCE,
            "Observed improvements coexist with item regressions; Phase P regression policy remains controlling.",
        )
    elif not weights_changed:
        evidence, rationale = (
            LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
            "Checkpoint hashes did not change.",
        )
    elif target_delta <= 0 and (
        generalization_delta is None or generalization_delta <= 0
    ):
        if training_loss_delta is not None and training_loss_delta < 0:
            evidence, rationale = (
                LearningEvidenceLevel.TRAINING_SIGNAL_ONLY,
                "Training loss decreased but no held-out capability or generalization delta was measured.",
            )
        else:
            evidence, rationale = (
                LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
                "Weights changed but no measured held-out capability delta was observed.",
            )
    elif generalization.status != "GENERALIZATION_EVIDENCE":
        evidence, rationale = (
            LearningEvidenceLevel.IN_DISTRIBUTION_IMPROVEMENT,
            "Held-out in-distribution capability increased, but independent generalization evidence did not meet policy.",
        )
    elif replications < 2:
        evidence, rationale = (
            LearningEvidenceLevel.GENERALIZATION_EVIDENCE,
            "One clean controlled run improved independent generalization; replication remains insufficient.",
        )
    else:
        evidence, rationale = (
            LearningEvidenceLevel.STRONG_LEARNING_EVIDENCE,
            "Replicated clean runs met independent generalization and regression criteria.",
        )
    return LearningAttribution(
        attribution_id=f"learning-{experiment_id}",
        experiment_id=experiment_id,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        weights_changed=weights_changed,
        training_loss_delta=training_loss_delta,
        held_out_loss_delta=held_out_loss_delta,
        target_capability_delta=target_delta,
        generalization_delta=generalization_delta,
        regression_rate=comparison.regression_rate,
        contaminated_task_ids=contamination,
        evidence=evidence,
        rationale=rationale,
    )


def dataset_attributions(
    *,
    experiment_id: str,
    full_mixture_delta: float,
    leave_one_out_deltas: dict[str, float | None],
) -> tuple[DatasetAttribution, ...]:
    """Report observed leave-one-out differences, never causal certainty."""
    results: list[DatasetAttribution] = []
    for dataset_id, without in sorted(leave_one_out_deltas.items()):
        marginal = None if without is None else round(full_mixture_delta - without, 6)
        status = (
            "INCONCLUSIVE_CONTRIBUTION" if without is None else "OBSERVED_CONTRIBUTION"
        )
        rationale = (
            "No bounded leave-one-out result exists."
            if without is None
            else "Observed full-mixture delta minus the corresponding leave-one-out delta; this is not causal certainty."
        )
        results.append(
            DatasetAttribution(
                attribution_id=f"dataset-attribution-{experiment_id}-{dataset_id}"[
                    :128
                ],
                experiment_id=experiment_id,
                dataset_release_id=dataset_id,
                full_mixture_delta=full_mixture_delta,
                without_dataset_delta=without,
                observed_marginal_delta=marginal,
                status=status,
                rationale=rationale,
            )
        )
    return tuple(results)


def efficiency_for(
    *,
    experiment: LearningExperiment,
    capability_gain: float,
    generalization_gain: float | None,
    newly_solved_items: int,
) -> EfficiencyRecord:
    tokens, steps, compute = (
        experiment.training_tokens,
        experiment.optimizer_steps,
        experiment.compute_units,
    )
    return EfficiencyRecord(
        efficiency_id=f"efficiency-{experiment.experiment_id}",
        experiment_id=experiment.experiment_id,
        capability_gain_per_token=round(capability_gain / tokens, 12)
        if tokens
        else None,
        generalization_gain_per_token=round(generalization_gain / tokens, 12)
        if tokens and generalization_gain is not None
        else None,
        capability_gain_per_step=round(capability_gain / steps, 12) if steps else None,
        capability_gain_per_compute=round(capability_gain / compute, 12)
        if compute
        else None,
        tokens_per_newly_solved_item=round(tokens / newly_solved_items, 6)
        if newly_solved_items
        else None,
        compute_per_newly_solved_item=round(compute / newly_solved_items, 6)
        if newly_solved_items
        else None,
        training_tokens=tokens,
        optimizer_steps=steps,
        compute_units=compute,
        newly_solved_items=newly_solved_items,
    )


def data_value_for(
    *,
    dataset_release_id: str,
    experiment_id: str,
    capability_contribution: float,
    generalization_contribution: float,
    target_importance: float,
    evidence_confidence: float,
    novelty: float,
    source_reliability: float,
    contamination_risk: float,
    token_cost: int,
    compute_cost: float,
) -> DataValueRecord:
    # Transparent, versioned heuristic. Negative contribution remains visible.
    benefit = (
        capability_contribution
        * max(0.0, generalization_contribution)
        * target_importance
        * evidence_confidence
        * novelty
        * source_reliability
        * (1 - contamination_risk)
    )
    cost = max(1.0, float(token_cost) + compute_cost)
    score = benefit / cost
    if contamination_risk > 0:
        status = DatasetValueStatus.QUARANTINED
    elif capability_contribution < 0:
        status = DatasetValueStatus.HARMFUL
    elif score > 0:
        status = DatasetValueStatus.USEFUL
    else:
        status = DatasetValueStatus.NEUTRAL
    return DataValueRecord(
        value_id=f"data-value-{experiment_id}-{dataset_release_id}"[:128],
        dataset_release_id=dataset_release_id,
        experiment_id=experiment_id,
        capability_contribution=capability_contribution,
        generalization_contribution=generalization_contribution,
        target_importance=target_importance,
        evidence_confidence=evidence_confidence,
        novelty=novelty,
        source_reliability=source_reliability,
        contamination_risk=contamination_risk,
        token_cost=token_cost,
        compute_cost=compute_cost,
        value_score=score,
        status=status,
    )


def choose_dataset_champion(
    *,
    values: tuple[DataValueRecord, ...],
    release_hashes: dict[str, str],
    previous_champion: ChampionDataset | None = None,
) -> ChampionDataset | None:
    """Select the best clean observed value without erasing prior champions."""
    eligible = [
        value
        for value in values
        if value.status in {DatasetValueStatus.USEFUL, DatasetValueStatus.CHAMPION}
    ]
    if not eligible:
        return None
    winner = min(
        eligible, key=lambda value: (-value.value_score, value.dataset_release_id)
    )
    release_hash = release_hashes.get(winner.dataset_release_id)
    if not release_hash:
        raise ValueError("Champion selection requires the immutable release hash.")
    return ChampionDataset(
        champion_id=f"champion-{winner.experiment_id}-{winner.dataset_release_id}"[
            :128
        ],
        dataset_release_id=winner.dataset_release_id,
        previous_champion_id=previous_champion.champion_id
        if previous_champion
        else None,
        experiment_ids=(winner.experiment_id,),
        release_hashes=(release_hash,),
        evidence=(
            "Highest transparent clean data-value score among supplied measured "
            "records; this is a ranking, not a causal claim."
        ),
    )


def diagnose_failure(
    *,
    experiment_id: str,
    attribution: LearningAttribution,
    training_stable: bool = True,
) -> ExperimentFailure:
    if attribution.contaminated_task_ids:
        diagnosis, rationale = (
            FailureDiagnosis.CONTAMINATION,
            "Contaminated evaluation items were quarantined.",
        )
    elif not attribution.weights_changed:
        diagnosis, rationale = (
            FailureDiagnosis.NO_WEIGHT_CHANGE,
            "Candidate checkpoint hash equals parent checkpoint hash.",
        )
    elif not training_stable:
        diagnosis, rationale = (
            FailureDiagnosis.TRAINING_INSTABILITY,
            "Training result reported instability.",
        )
    elif attribution.regression_rate > 0:
        diagnosis, rationale = (
            FailureDiagnosis.REGRESSION,
            "Phase P detected regressed scored items.",
        )
    elif attribution.target_capability_delta <= 0:
        diagnosis, rationale = (
            FailureDiagnosis.NO_CAPABILITY_GAIN,
            "No target capability gain was measured.",
        )
    elif (
        attribution.generalization_delta is None
        or attribution.generalization_delta <= 0
    ):
        diagnosis, rationale = (
            FailureDiagnosis.NO_HELD_OUT_GAIN,
            "No independent generalization gain was measured.",
        )
    else:
        diagnosis, rationale = (
            FailureDiagnosis.EVALUATION_LIMITATION,
            "No failure diagnosis is warranted by the measured evidence.",
        )
    return ExperimentFailure(
        failure_id=f"failure-{experiment_id}",
        experiment_id=experiment_id,
        diagnosis=diagnosis,
        rationale=rationale,
    )


def next_recommendation(
    *,
    experiment: LearningExperiment,
    attribution: LearningAttribution,
    failure: ExperimentFailure,
    expected_value: float,
    confidence: float,
    preferred_source: str = "validated source-derived Phase Q set",
) -> NextExperimentRecommendation:
    if failure.diagnosis is FailureDiagnosis.CONTAMINATION:
        action, requirement, cause = (
            RecommendationAction.DEFER,
            "Create a non-overlapping evaluation suite before another training run.",
            "evaluation contamination",
        )
    elif attribution.evidence in {
        LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
        LearningEvidenceLevel.TRAINING_SIGNAL_ONLY,
    }:
        action, requirement, cause = (
            RecommendationAction.ACQUIRE_MORE_DATA,
            "Add structurally diverse, validated examples targeted at the measured weakness.",
            "insufficient measured transfer",
        )
    elif attribution.evidence is LearningEvidenceLevel.GENERALIZATION_EVIDENCE:
        action, requirement, cause = (
            RecommendationAction.REPLICATE,
            "Repeat the same immutable intervention under a new seed before stronger attribution.",
            "insufficient replication",
        )
    elif attribution.evidence is LearningEvidenceLevel.STRONG_LEARNING_EVIDENCE:
        action, requirement, cause = (
            RecommendationAction.ACCEPT,
            "Route only as a recommendation through the existing Phase N human promotion gate.",
            "replicated clean evidence",
        )
    else:
        action, requirement, cause = (
            RecommendationAction.CHANGE_CURRICULUM,
            "Test bounded ordered, randomized, or reversed curriculum variants.",
            "limited generalization evidence",
        )
    observed = f"target_delta={attribution.target_capability_delta:.6f}; generalization_delta={attribution.generalization_delta}; regression_rate={attribution.regression_rate:.6f}; evidence={attribution.evidence.value}"
    return NextExperimentRecommendation(
        recommendation_id=f"recommendation-{experiment.experiment_id}",
        experiment_id=experiment.experiment_id,
        target_id=experiment.target_id,
        observed=observed,
        likely_cause=cause,
        action=action,
        data_requirement=requirement,
        preferred_source=preferred_source,
        expected_value=expected_value,
        confidence=confidence,
    )


def bounded_designs(
    *,
    target_id: str,
    dataset_release_ids: tuple[str, ...],
    estimated_tokens: int,
    estimated_compute_units: float,
    policy: ExperimentPolicy,
) -> tuple[ExperimentDesign, ...]:
    """Choose full/single/leave-one-out designs without combinatorial expansion."""
    policy.verify_integrity()
    chosen = tuple(dataset_release_ids[: policy.maximum_datasets_per_experiment])
    if (
        not chosen
        or estimated_tokens > policy.maximum_training_tokens
        or estimated_compute_units > policy.maximum_compute_units
    ):
        return ()
    designs = [
        ExperimentDesign(
            design_id=f"design-{target_id}-full"[:128],
            target_id=target_id,
            kind="FULL_MIXTURE",
            dataset_release_ids=chosen,
            estimated_tokens=estimated_tokens,
            estimated_compute_units=estimated_compute_units,
            rationale="Bounded full mixture baseline.",
        )
    ]
    if len(chosen) == 1:
        return tuple(designs)
    for dataset_id in chosen:
        if len(designs) >= policy.maximum_experiments:
            break
        remainder = tuple(value for value in chosen if value != dataset_id)
        designs.append(
            ExperimentDesign(
                design_id=f"design-{target_id}-without-{dataset_id}"[:128],
                target_id=target_id,
                kind="LEAVE_ONE_OUT",
                dataset_release_ids=remainder,
                estimated_tokens=estimated_tokens,
                estimated_compute_units=estimated_compute_units,
                rationale=f"Bounded leave-one-out for {dataset_id}; observed, not causal, attribution.",
            )
        )
    return tuple(designs)


class PhaseREngine:
    """Persists immutable Phase R evidence; it cannot launch or promote training."""

    def __init__(self, root: Path) -> None:
        self.store = PhaseRStore(root)

    def configure(self, policy: ExperimentPolicy) -> ExperimentPolicy:
        policy.verify_integrity()
        self.store.write_immutable("policies", policy.policy_id, policy)
        return policy

    def record_experiment(self, experiment: LearningExperiment) -> LearningExperiment:
        experiment.verify_integrity()
        self.store.write_immutable("experiments", experiment.experiment_id, experiment)
        return experiment

    def persist(self, category: str, identifier: str, value: BaseModel) -> BaseModel:
        self.store.write_immutable(category, identifier, value)
        return value
