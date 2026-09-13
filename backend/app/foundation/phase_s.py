"""Phase S: bounded, evidence-driven optimization specifications for Phase O.

This module never trains a model itself.  It makes immutable intervention and
adaptation decisions which Phase O may execute only through its existing Phase
N-gated job path.  Scores remain Phase P/R evidence, not optimizer output.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash
from app.foundation.phase_p import CapabilityComparison
from app.foundation.phase_r import LearningAttribution, LearningEvidenceLevel
from app.foundation.training import TrainingResult


class InterventionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED_FOR_PHASE_O = "APPROVED_FOR_PHASE_O"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    PAUSED_FOR_REVIEW = "PAUSED_FOR_REVIEW"


class DifficultyBand(str, Enum):
    FOUNDATIONAL = "FOUNDATIONAL"
    EASY = "EASY"
    MODERATE = "MODERATE"
    HARD = "HARD"
    COMPOSITIONAL = "COMPOSITIONAL"
    ADVERSARIAL = "ADVERSARIAL"


class SamplingPolicy(str, Enum):
    RELEASE_ORDER = "RELEASE_ORDER"
    TARGET_DENSITY = "TARGET_DENSITY"
    FAILURE_REPLAY = "FAILURE_REPLAY"
    RETENTION_REHEARSAL = "RETENTION_REHEARSAL"


class AdaptiveAction(str, Enum):
    INCREASE_TARGET_DENSITY = "INCREASE_TARGET_DENSITY"
    INCREASE_STRUCTURAL_DIVERSITY = "INCREASE_STRUCTURAL_DIVERSITY"
    ADD_PREREQUISITES = "ADD_PREREQUISITES"
    CHANGE_CURRICULUM = "CHANGE_CURRICULUM"
    ADJUST_LEARNING_RATE = "ADJUST_LEARNING_RATE"
    ADD_RETENTION_REHEARSAL = "ADD_RETENTION_REHEARSAL"
    REPLICATE = "REPLICATE"
    PAUSE_FOR_REVIEW = "PAUSE_FOR_REVIEW"
    ARCHITECTURE_LIMITATION_SUSPECTED = "ARCHITECTURE_LIMITATION_SUSPECTED"


class StopReason(str, Enum):
    NONE = "NONE"
    TRAINING_INSTABILITY = "TRAINING_INSTABILITY"
    VALIDATION_DEGRADATION = "VALIDATION_DEGRADATION"
    CAPABILITY_REGRESSION = "CAPABILITY_REGRESSION"
    NO_MEASURABLE_IMPROVEMENT = "NO_MEASURABLE_IMPROVEMENT"
    RESOURCE_CEILING = "RESOURCE_CEILING"


class LearningPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    version: str = "phase-s-v1"
    maximum_adaptive_iterations: int = Field(default=3, ge=1)
    maximum_experiments: int = Field(default=4, ge=1)
    maximum_tokens_total: int = Field(default=1_000_000, ge=1)
    maximum_tokens_per_intervention: int = Field(default=500_000, ge=1)
    maximum_concurrent_jobs: int = Field(default=1, ge=1)
    maximum_learning_rate_variants: int = Field(default=3, ge=1)
    maximum_curriculum_variants: int = Field(default=3, ge=1)
    maximum_dataset_mixtures: int = Field(default=3, ge=1)
    maximum_regression_rate: float = Field(default=0.0, ge=0, le=1)
    minimum_generalization_delta: float = Field(default=0.01, ge=0, le=1)
    no_capability_patience: int = Field(default=1, ge=1)
    advance_difficulty_success_rate: float = Field(default=0.9, ge=0, le=1)
    retain_difficulty_min_success_rate: float = Field(default=0.4, ge=0, le=1)
    prerequisite_success_rate: float = Field(default=0.2, ge=0, le=1)
    require_human_authorization: bool = True
    fingerprint: str

    def verify_integrity(self) -> None:
        if not (
            self.prerequisite_success_rate
            <= self.retain_difficulty_min_success_rate
            <= self.advance_difficulty_success_rate
        ):
            raise ValueError(
                "Difficulty thresholds must be ordered prerequisite ≤ retain ≤ advance."
            )
        if (
            canonical_hash(self.model_dump(mode="json", exclude={"fingerprint"}))
            != self.fingerprint
        ):
            raise ValueError(
                "Learning policy fingerprint changed; create a new immutable policy."
            )


def create_learning_policy(policy_id: str, **values: object) -> LearningPolicy:
    unsigned = LearningPolicy(policy_id=policy_id, **values, fingerprint="")
    return unsigned.model_copy(
        update={
            "fingerprint": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"fingerprint"})
            )
        }
    )


class EarlyStopPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    capability_evaluation_available: bool = False
    patience_evaluations: int = Field(default=2, ge=1)
    maximum_validation_loss_increase: float = Field(default=0.1, ge=0)
    maximum_gradient_norm: float | None = Field(default=None, gt=0)
    stop_on_capability_regression: bool = True
    fingerprint: str


def create_early_stop_policy(policy_id: str, **values: object) -> EarlyStopPolicy:
    unsigned = EarlyStopPolicy(policy_id=policy_id, **values, fingerprint="")
    return unsigned.model_copy(
        update={
            "fingerprint": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"fingerprint"})
            )
        }
    )


class TrainingIntervention(BaseModel):
    """Immutable Phase-S specification which maps onto an existing O config."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_id: str
    parent_model_id: str
    parent_checkpoint_sha256: str
    target_id: str
    dataset_release_ids: tuple[str, ...]
    dataset_release_hashes: tuple[str, ...]
    curriculum_id: str | None
    curriculum_ordering: tuple[DifficultyBand, ...]
    mixture_weights: dict[str, float]
    sampling_policy: SamplingPolicy
    dataset_weights: dict[str, float] = Field(default_factory=dict)
    target_weight: float = Field(default=1.0, gt=0)
    difficulty_weights: dict[str, float] = Field(default_factory=dict)
    novelty_weight: float = Field(default=1.0, gt=0)
    failure_repair_weight: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(gt=0)
    scheduler: str
    optimizer: str
    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    weight_decay: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    seed: int
    checkpoint_interval: int = Field(ge=1)
    early_stop_policy: EarlyStopPolicy
    status: InterventionStatus = InterventionStatus.PROPOSED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    configuration_sha256: str

    def model_post_init(self, __context: object, /) -> None:
        if not self.dataset_release_ids or len(self.dataset_release_ids) != len(
            self.dataset_release_hashes
        ):
            raise ValueError(
                "Interventions require aligned immutable release IDs and hashes."
            )
        if not self.curriculum_ordering:
            raise ValueError(
                "An intervention requires an explicit difficulty ordering."
            )
        if not self.mixture_weights or any(
            value <= 0 for value in self.mixture_weights.values()
        ):
            raise ValueError("Mixture weights must be explicit and positive.")
        if not math.isclose(sum(self.mixture_weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Mixture weights must sum to exactly one.")

    def verify_integrity(self) -> None:
        payload = self.model_dump(
            mode="json", exclude={"configuration_sha256", "created_at"}
        )
        if canonical_hash(payload) != self.configuration_sha256:
            raise ValueError(
                "Intervention configuration changed; create a new intervention ID."
            )


def create_intervention(**values: object) -> TrainingIntervention:
    unsigned = TrainingIntervention(**values, configuration_sha256="")
    return unsigned.model_copy(
        update={
            "configuration_sha256": canonical_hash(
                unsigned.model_dump(
                    mode="json", exclude={"configuration_sha256", "created_at"}
                )
            )
        }
    )


class AdaptiveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decision_id: str
    policy_id: str
    policy_fingerprint: str
    previous_intervention_id: str
    observed_attribution_id: str
    observed_evidence: LearningEvidenceLevel
    decision_rule: str
    action: AdaptiveAction
    new_intervention_id: str
    rationale: str
    iteration: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReplayProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_id: str
    original_exposure: int = Field(ge=0)
    replay_exposure: int = Field(ge=0)
    unique_examples: int = Field(ge=0)
    unique_templates: int = Field(ge=0)
    replay_fraction: float = Field(ge=0, le=1)
    structurally_diverse: bool
    rationale: str

    def model_post_init(self, __context: object, /) -> None:
        if self.replay_exposure and not self.structurally_diverse:
            raise ValueError("Targeted replay requires structurally diverse examples.")


class DifficultyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_id: str
    observed_success_rate: float = Field(ge=0, le=1)
    current_band: DifficultyBand
    next_band: DifficultyBand
    action: str
    policy_id: str
    rationale: str


class CheckpointEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint_id: str
    checkpoint_sha256: str
    target_capability_delta: float = Field(ge=-1, le=1)
    generalization_delta: float | None = Field(default=None, ge=-1, le=1)
    regression_rate: float = Field(ge=0, le=1)
    training_stable: bool
    tokens_processed: int = Field(ge=0)


class CheckpointSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    selection_id: str
    intervention_id: str
    selected_checkpoint_id: str | None
    candidates: tuple[CheckpointEvidence, ...]
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StabilityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_id: str
    stable: bool
    flags: tuple[str, ...]
    training_loss: float | None
    validation_loss: float | None
    maximum_gradient_norm: float | None
    unavailable_metrics: tuple[str, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EfficiencyComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_id: str
    capability_gain_per_token: float | None
    generalization_gain_per_token: float | None
    capability_gain_per_compute: float | None
    generalization_gain_per_compute: float | None
    tokens: int
    compute_units: float | None


class ForgettingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_id: str
    regressed_domains: dict[str, float]
    catastrophic_forgetting_risk: bool
    rehearsal_recommended: bool
    rationale: str


class ReplicationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_family_id: str
    seeds: tuple[int, ...]
    target_deltas: tuple[float, ...]
    mean: float | None
    median: float | None
    variance: float | None
    status: str


class LearningCurveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intervention_id: str
    points: tuple[CheckpointEvidence, ...]
    status: str
    rationale: str


class NextTrainingRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    recommendation_id: str
    intervention_id: str
    action: AdaptiveAction
    rationale: str
    confidence: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PhaseSStore:
    _CATEGORIES = frozenset(
        {
            "policies",
            "interventions",
            "decisions",
            "replay",
            "difficulty",
            "checkpoints",
            "stability",
            "efficiency",
            "forgetting",
            "replication",
            "curves",
            "recommendations",
        }
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_s"
        for category in self._CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase S artifact category: {category}")
        path = self.root / category / f"{identifier}.json"
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    f"Immutable Phase S artifact already exists: {identifier}"
                )
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def reads(self, category: str, model: type[BaseModel]) -> tuple[BaseModel, ...]:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase S artifact category: {category}")
        return tuple(
            model.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / category).glob("*.json"))
        )


def learning_rate_variants(base: float, policy: LearningPolicy) -> tuple[float, ...]:
    policy.verify_integrity()
    variants = (base * 0.5, base, base * 2.0)
    return variants[: policy.maximum_learning_rate_variants]


def adaptive_difficulty(
    *,
    target_id: str,
    current: DifficultyBand,
    success_rate: float,
    policy: LearningPolicy,
) -> DifficultyDecision:
    policy.verify_integrity()
    ordered = tuple(DifficultyBand)
    index = ordered.index(current)
    if (
        success_rate >= policy.advance_difficulty_success_rate
        and index < len(ordered) - 1
    ):
        next_band, action = ordered[index + 1], "ADVANCE"
    elif success_rate < policy.prerequisite_success_rate:
        next_band, action = DifficultyBand.FOUNDATIONAL, "ADD_PREREQUISITES"
    else:
        next_band, action = current, "RETAIN"
    return DifficultyDecision(
        target_id=target_id,
        observed_success_rate=success_rate,
        current_band=current,
        next_band=next_band,
        action=action,
        policy_id=policy.policy_id,
        rationale="Aggregated success-rate thresholds from immutable learning policy.",
    )


def adapt_intervention(
    *,
    previous: TrainingIntervention,
    next_intervention: TrainingIntervention,
    attribution: LearningAttribution,
    policy: LearningPolicy,
    iteration: int,
) -> AdaptiveDecision:
    """Audit a bounded evidence-to-configuration adaptation.

    The caller materializes the proposed immutable next intervention first;
    this verifies it is an explicit configuration change, never a hidden
    optimizer mutation.
    """
    policy.verify_integrity()
    if iteration > policy.maximum_adaptive_iterations:
        raise ValueError("Learning policy maximum adaptive iterations reached.")
    if previous.target_id != next_intervention.target_id:
        raise ValueError("Adaptive interventions must retain the same declared target.")
    if (
        attribution.evidence
        in {
            LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
            LearningEvidenceLevel.TRAINING_SIGNAL_ONLY,
        }
        and next_intervention.target_weight > previous.target_weight
    ):
        action, rule = (
            AdaptiveAction.INCREASE_TARGET_DENSITY,
            "No measured learning and increased target sampling weight.",
        )
    elif (
        attribution.evidence
        in {
            LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
            LearningEvidenceLevel.TRAINING_SIGNAL_ONLY,
        }
        and next_intervention.dataset_release_ids != previous.dataset_release_ids
    ):
        action, rule = (
            AdaptiveAction.INCREASE_STRUCTURAL_DIVERSITY,
            "No measured learning and a new validated release was added.",
        )
    elif next_intervention.learning_rate != previous.learning_rate:
        action, rule = (
            AdaptiveAction.ADJUST_LEARNING_RATE,
            "Learning rate changed under the bounded policy.",
        )
    elif attribution.regression_rate > policy.maximum_regression_rate:
        action, rule = (
            AdaptiveAction.ADD_RETENTION_REHEARSAL,
            "Regression exceeds policy threshold; recommend retention mixture.",
        )
    else:
        action, rule = (
            AdaptiveAction.PAUSE_FOR_REVIEW,
            "No permitted evidence-based configuration change was supplied.",
        )
    return AdaptiveDecision(
        decision_id=f"adaptive-{previous.intervention_id}-to-{next_intervention.intervention_id}"[
            :128
        ],
        policy_id=policy.policy_id,
        policy_fingerprint=policy.fingerprint,
        previous_intervention_id=previous.intervention_id,
        observed_attribution_id=attribution.attribution_id,
        observed_evidence=attribution.evidence,
        decision_rule=rule,
        action=action,
        new_intervention_id=next_intervention.intervention_id,
        rationale="The decision references immutable Phase R evidence and an explicit immutable configuration delta.",
        iteration=iteration,
    )


def stability_for(
    *, intervention_id: str, result: TrainingResult, early_stop: EarlyStopPolicy
) -> StabilityAssessment:
    flags: list[str] = []
    for name, value in (
        ("training_loss", result.training_loss),
        ("validation_loss", result.validation_loss),
    ):
        if not math.isfinite(value):
            flags.append(f"NONFINITE_{name.upper()}")
    if (
        early_stop.maximum_gradient_norm is not None
        and result.maximum_gradient_norm is not None
        and result.maximum_gradient_norm > early_stop.maximum_gradient_norm
    ):
        flags.append("GRADIENT_NORM_EXCEEDED")
    if result.early_stop_reason:
        flags.append(f"TRAINER_STOP_{result.early_stop_reason}")
    unavailable = tuple(
        name
        for name, value in (("gradient_norm", result.maximum_gradient_norm),)
        if value is None
    )
    return StabilityAssessment(
        intervention_id=intervention_id,
        stable=not flags,
        flags=tuple(flags),
        training_loss=result.training_loss,
        validation_loss=result.validation_loss,
        maximum_gradient_norm=result.maximum_gradient_norm,
        unavailable_metrics=unavailable,
    )


def should_stop(
    *,
    stability: StabilityAssessment,
    comparison: CapabilityComparison | None,
    early_stop: EarlyStopPolicy,
    resource_ceiling_reached: bool = False,
) -> StopReason:
    if resource_ceiling_reached:
        return StopReason.RESOURCE_CEILING
    if not stability.stable:
        return StopReason.TRAINING_INSTABILITY
    if (
        comparison
        and early_stop.stop_on_capability_regression
        and comparison.regression_rate > 0
    ):
        return StopReason.CAPABILITY_REGRESSION
    # Loss alone cannot terminate a run when no capability observation exists.
    if (
        early_stop.capability_evaluation_available
        and stability.validation_loss is not None
        and stability.training_loss is not None
        and stability.validation_loss
        > stability.training_loss + early_stop.maximum_validation_loss_increase
    ):
        return StopReason.VALIDATION_DEGRADATION
    return StopReason.NONE


def select_checkpoint(
    *,
    intervention_id: str,
    candidates: tuple[CheckpointEvidence, ...],
    maximum_regression_rate: float,
) -> CheckpointSelection:
    eligible = tuple(
        item
        for item in candidates
        if item.training_stable and item.regression_rate <= maximum_regression_rate
    )
    selected = (
        min(
            eligible,
            key=lambda item: (
                -item.target_capability_delta,
                -(
                    item.generalization_delta
                    if item.generalization_delta is not None
                    else -1.0
                ),
                item.tokens_processed,
                item.checkpoint_id,
            ),
        )
        if eligible
        else None
    )
    rationale = (
        "Selected by target capability, then generalization, regression safety, stability, and fewer tokens."
        if selected
        else "No checkpoint met stability and regression boundaries."
    )
    return CheckpointSelection(
        selection_id=f"checkpoint-selection-{intervention_id}",
        intervention_id=intervention_id,
        selected_checkpoint_id=selected.checkpoint_id if selected else None,
        candidates=candidates,
        rationale=rationale,
    )


def efficiency_for(
    *,
    intervention_id: str,
    capability_gain: float,
    generalization_gain: float | None,
    tokens: int,
    compute_units: float | None,
) -> EfficiencyComparison:
    return EfficiencyComparison(
        intervention_id=intervention_id,
        capability_gain_per_token=capability_gain / tokens if tokens else None,
        generalization_gain_per_token=generalization_gain / tokens
        if tokens and generalization_gain is not None
        else None,
        capability_gain_per_compute=capability_gain / compute_units
        if compute_units
        else None,
        generalization_gain_per_compute=generalization_gain / compute_units
        if compute_units and generalization_gain is not None
        else None,
        tokens=tokens,
        compute_units=compute_units,
    )


def forgetting_for(
    *, intervention_id: str, comparison: CapabilityComparison, tolerance: float = 0.05
) -> ForgettingAssessment:
    regressed = {
        domain: delta
        for domain, delta in comparison.domain_deltas.items()
        if delta < -tolerance
    }
    return ForgettingAssessment(
        intervention_id=intervention_id,
        regressed_domains=regressed,
        catastrophic_forgetting_risk=bool(regressed),
        rehearsal_recommended=bool(regressed),
        rationale="Phase P domain deltas below the explicit retention tolerance require a rehearsal recommendation."
        if regressed
        else "No domain crossed the explicit retention tolerance.",
    )


def summarize_replication(
    *,
    intervention_family_id: str,
    seeds: tuple[int, ...],
    target_deltas: tuple[float, ...],
    required: int,
) -> ReplicationSummary:
    if len(seeds) != len(target_deltas) or len(set(seeds)) != len(seeds):
        raise ValueError("Replication requires one unique seed per measured delta.")
    if len(target_deltas) < required:
        return ReplicationSummary(
            intervention_family_id=intervention_family_id,
            seeds=seeds,
            target_deltas=target_deltas,
            mean=None,
            median=None,
            variance=None,
            status="INSUFFICIENT_REPLICATION",
        )
    ordered = sorted(target_deltas)
    median = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    )
    mean = sum(target_deltas) / len(target_deltas)
    variance = sum((value - mean) ** 2 for value in target_deltas) / len(target_deltas)
    return ReplicationSummary(
        intervention_family_id=intervention_family_id,
        seeds=seeds,
        target_deltas=target_deltas,
        mean=mean,
        median=median,
        variance=variance,
        status="MEASURED_REPLICATION",
    )


def learning_curve_for(
    *, intervention_id: str, points: tuple[CheckpointEvidence, ...]
) -> LearningCurveAssessment:
    ordered = tuple(sorted(points, key=lambda point: point.tokens_processed))
    if len(ordered) < 3:
        return LearningCurveAssessment(
            intervention_id=intervention_id,
            points=ordered,
            status="INSUFFICIENT_CURVE",
            rationale="Fewer than three measured checkpoints; no saturation inference.",
        )
    gains = [point.target_capability_delta for point in ordered]
    saturation = gains[-1] <= gains[-2] <= gains[-3] and gains[-1] <= 0
    return LearningCurveAssessment(
        intervention_id=intervention_id,
        points=ordered,
        status="SATURATING" if saturation else "MEASURED_NO_SATURATION",
        rationale="Measured checkpoint deltas show diminishing non-positive return."
        if saturation
        else "Measured checkpoints do not satisfy the conservative saturation rule.",
    )


def recommend_next_training(
    *,
    intervention_id: str,
    attribution: LearningAttribution,
    repeated_no_learning: bool,
) -> NextTrainingRecommendation:
    if repeated_no_learning:
        action, rationale = (
            AdaptiveAction.PAUSE_FOR_REVIEW,
            "Multiple bounded interventions produced no measured capability gain; collect stronger data or revisit the development-model boundary before spending more budget.",
        )
    elif attribution.regression_rate > 0:
        action, rationale = (
            AdaptiveAction.ADD_RETENTION_REHEARSAL,
            "Regression evidence requires a measured rehearsal mixture before further target escalation.",
        )
    elif attribution.evidence is LearningEvidenceLevel.GENERALIZATION_EVIDENCE:
        action, rationale = (
            AdaptiveAction.REPLICATE,
            "Independent generalization evidence should be replicated before a stronger conclusion.",
        )
    else:
        action, rationale = (
            AdaptiveAction.INCREASE_STRUCTURAL_DIVERSITY,
            "Insufficient clean evidence suggests a new structurally diverse, validated release.",
        )
    return NextTrainingRecommendation(
        recommendation_id=f"next-training-{intervention_id}",
        intervention_id=intervention_id,
        action=action,
        rationale=rationale,
        confidence=0.7,
    )


class PhaseSEngine:
    """Immutable Phase-S optimizer evidence; deliberately does not launch jobs."""

    def __init__(self, root: Path) -> None:
        self.store = PhaseSStore(root)

    def configure(self, policy: LearningPolicy) -> LearningPolicy:
        policy.verify_integrity()
        self.store.write_immutable("policies", policy.policy_id, policy)
        return policy

    def record_intervention(
        self, intervention: TrainingIntervention, policy: LearningPolicy
    ) -> TrainingIntervention:
        policy.verify_integrity()
        intervention.verify_integrity()
        if intervention.max_tokens > policy.maximum_tokens_per_intervention:
            raise ValueError(
                "Intervention exceeds the immutable per-intervention token boundary."
            )
        if (
            len(self.store.reads("interventions", TrainingIntervention))
            >= policy.maximum_experiments
        ):
            raise ValueError("Learning policy maximum experiment count reached.")
        self.store.write_immutable(
            "interventions", intervention.intervention_id, intervention
        )
        return intervention

    def persist(self, category: str, identifier: str, value: BaseModel) -> BaseModel:
        self.store.write_immutable(category, identifier, value)
        return value
