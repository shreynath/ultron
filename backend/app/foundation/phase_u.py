"""Phase U: auditable substantial proprietary-learning campaigns.

Phase U deliberately reuses Phase O's job lifecycle and FoundationTrainer.  It
adds the data-accounting, native mixture sequence selection, resource gate, and
verdict artifacts needed to distinguish a million-token-scale campaign from a
small pipeline smoke test.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.foundation.checkpoints import canonical_hash
from app.foundation.data import PackedShardSequenceDataset
from app.foundation.phase_r import LearningAttribution, LearningEvidenceLevel
from app.foundation.phase_t import (
    DeterministicMixtureSampler,
    MixtureExposure,
    MixtureSample,
    ModelArchitectureRecord,
    exposure_for,
)

PHASE_T_CANDIDATE_TOKENS = 2_048
PHASE_U_MINIMUM_EFFECTIVE_TOKENS = 20_480_000


class LearningVerdict(str, Enum):
    TRAINING_SIGNAL_ONLY = "TRAINING_SIGNAL_ONLY"
    CAPABILITY_ACQUISITION = "CAPABILITY_ACQUISITION"
    GENERALIZED_CAPABILITY_ACQUISITION = "GENERALIZED_CAPABILITY_ACQUISITION"
    REPLICATED_CAPABILITY_ACQUISITION = "REPLICATED_CAPABILITY_ACQUISITION"
    NO_LEARNING_EVIDENCE = "NO_LEARNING_EVIDENCE"
    REGRESSION = "REGRESSION"
    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"
    PHASE_U_RESOURCE_BLOCKED = "PHASE_U_RESOURCE_BLOCKED"


class SourceProvenance(BaseModel):
    """The required licensing facts for a source included in a U release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    source_uri: str
    license: str
    retrieval_timestamp: datetime
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    license_evidence: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    document_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def _licensed(self) -> SourceProvenance:
        if self.license.upper() in {"", "UNSPECIFIED", "UNKNOWN"}:
            raise ValueError("Phase U sources require an explicit license.")
        return self


class CorpusAccounting(BaseModel):
    """Token accounting that rejects duplicate/repeated budget inflation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_release_id: str
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_documents: int = Field(ge=0)
    normalized_documents: int = Field(ge=0)
    unique_documents: int = Field(ge=0)
    unique_records: int = Field(ge=0)
    raw_tokens: int = Field(ge=0)
    deduplicated_tokens: int = Field(ge=0)
    train_tokens: int = Field(ge=0)
    validation_tokens: int = Field(ge=0)
    test_tokens: int = Field(ge=0)
    teacher_generated_tokens: int = Field(ge=0)
    source_derived_tokens: int = Field(ge=0)
    synthetic_tokens: int = Field(ge=0)
    repeated_tokens: int = Field(ge=0)
    contaminated_tokens: int = Field(ge=0)
    quarantined_tokens: int = Field(ge=0)
    effective_unique_training_tokens: int = Field(ge=0)
    training_token_multiplier_vs_phase_t: float = Field(ge=0)
    duplicate_rate: float = Field(ge=0, le=1)
    contamination_rate: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _consistent(self) -> CorpusAccounting:
        if not (
            self.raw_documents >= self.normalized_documents >= self.unique_documents
        ):
            raise ValueError(
                "Document accounting must be monotonic after normalization."
            )
        if self.deduplicated_tokens > self.raw_tokens:
            raise ValueError("Deduplicated tokens cannot exceed raw tokens.")
        expected = max(
            0,
            self.train_tokens
            - self.repeated_tokens
            - self.contaminated_tokens
            - self.quarantined_tokens,
        )
        if self.effective_unique_training_tokens != expected:
            raise ValueError(
                "Effective training tokens must exclude repeated, contaminated, and quarantined tokens."
            )
        if self.effective_unique_training_tokens > self.deduplicated_tokens:
            raise ValueError(
                "Effective training tokens exceed deduplicated corpus tokens."
            )
        expected_multiplier = (
            self.effective_unique_training_tokens / PHASE_T_CANDIDATE_TOKENS
        )
        if abs(self.training_token_multiplier_vs_phase_t - expected_multiplier) > 1e-9:
            raise ValueError(
                "Phase T multiplier must be calculated from effective tokens."
            )
        return self

    @property
    def meets_phase_u_minimum(self) -> bool:
        return self.effective_unique_training_tokens >= PHASE_U_MINIMUM_EFFECTIVE_TOKENS


def account_corpus(**values: object) -> CorpusAccounting:
    raw = dict(values)
    train = int(raw["train_tokens"])
    raw.setdefault(
        "effective_unique_training_tokens",
        train
        - int(raw.get("repeated_tokens", 0))
        - int(raw.get("contaminated_tokens", 0))
        - int(raw.get("quarantined_tokens", 0)),
    )
    raw.setdefault(
        "training_token_multiplier_vs_phase_t",
        int(raw["effective_unique_training_tokens"]) / PHASE_T_CANDIDATE_TOKENS,
    )
    return CorpusAccounting(**raw)


class MixtureConfiguration(BaseModel):
    """Immutable policy used by ``NativeMixtureSequenceDataset`` at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mixture_id: str
    release_weights: dict[str, float]
    seed: int
    tolerance: float = Field(default=0.02, gt=0, le=1)
    sampling_algorithm: str = "SEEDED_WEIGHTED_RELEASE_THEN_PERMUTATION_NO_REPLACEMENT"
    sampling_version: str = "phase-u-v1"
    configuration_sha256: str

    @model_validator(mode="after")
    def _weights(self) -> MixtureConfiguration:
        if not self.release_weights or any(
            value <= 0 for value in self.release_weights.values()
        ):
            raise ValueError("Mixture weights must be explicit and positive.")
        if abs(sum(self.release_weights.values()) - 1.0) > 1e-6:
            raise ValueError("Mixture weights must sum to one.")
        expected = canonical_hash(
            self.model_dump(mode="json", exclude={"configuration_sha256"})
        )
        if self.configuration_sha256 and expected != self.configuration_sha256:
            raise ValueError("Mixture configuration changed; create a new mixture ID.")
        return self


def create_mixture_configuration(
    mixture_id: str, *, release_weights: dict[str, float], seed: int, **values: object
) -> MixtureConfiguration:
    unsigned = MixtureConfiguration(
        mixture_id=mixture_id,
        release_weights=release_weights,
        seed=seed,
        configuration_sha256="",
        **values,
    )
    return unsigned.model_copy(
        update={
            "configuration_sha256": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"configuration_sha256"})
            )
        }
    )


class NativeMixtureSequenceDataset(Sequence[list[int] | tuple[list[int], list[int]]]):
    """Actual data-loader selection, not post-hoc mixture accounting.

    The complete sampled plan is retained as a checkpoint-replayable data
    position.  It references immutable packed releases lazily, so it does not
    materialize all token arrays in memory.
    """

    def __init__(
        self,
        *,
        datasets: dict[str, PackedShardSequenceDataset],
        configuration: MixtureConfiguration,
        samples: int,
        tokens_per_sample: int,
    ) -> None:
        if set(datasets) != set(configuration.release_weights):
            raise ValueError(
                "Packed releases must exactly match mixture configuration."
            )
        if samples < 1 or tokens_per_sample < 1:
            raise ValueError("Mixture dataset needs a positive sample and token count.")
        self.datasets = datasets
        self.configuration = configuration
        self.tokens_per_sample = tokens_per_sample
        sampler = DeterministicMixtureSampler(
            sequences_by_release={
                key: tuple(range(len(value))) for key, value in datasets.items()
            },
            weights=configuration.release_weights,
            seed=configuration.seed,
        )
        release_plan = sampler.plan(samples)
        permutations = {
            release_id: self._permutation(len(dataset), configuration.seed, release_id)
            for release_id, dataset in datasets.items()
        }
        seen: dict[str, int] = {release_id: 0 for release_id in datasets}
        plan: list[MixtureSample] = []
        for selected in release_plan:
            sequence_number = seen[selected.release_id]
            permutation = permutations[selected.release_id]
            plan.append(
                MixtureSample(
                    selected.release_id,
                    permutation[sequence_number % len(permutation)],
                )
            )
            seen[selected.release_id] += 1
        self.plan = tuple(plan)
        self.repeated_samples = sum(
            max(0, count - len(datasets[release_id]))
            for release_id, count in seen.items()
        )
        self.exposure: MixtureExposure = exposure_for(
            weights=configuration.release_weights,
            samples=self.plan,
            tokens_per_sample=tokens_per_sample,
            seed=configuration.seed,
            tolerance=configuration.tolerance,
        )

    @staticmethod
    def _permutation(length: int, seed: int, release_id: str) -> tuple[int, ...]:
        if length < 1:
            raise ValueError("Mixture releases need one or more packed sequences.")
        values = list(range(length))
        random.Random(f"{seed}:{release_id}").shuffle(values)
        return tuple(values)

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(
        self, index: int | slice
    ) -> (
        list[int]
        | tuple[list[int], list[int]]
        | list[list[int] | tuple[list[int], list[int]]]
    ):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        sample = self.plan[index]
        return self.datasets[sample.release_id][sample.sequence_index]

    def sampler_state(self, position: int) -> dict[str, object]:
        if not 0 <= position <= len(self):
            raise ValueError("Sampler position falls outside the immutable plan.")
        return {
            "mixture_configuration_sha256": self.configuration.configuration_sha256,
            "sampling_algorithm": self.configuration.sampling_algorithm,
            "sampling_version": self.configuration.sampling_version,
            "seed": self.configuration.seed,
            "position": position,
            "planned_samples": len(self),
            "repeated_samples": self.repeated_samples,
            "observed_exposure": self.exposure.model_dump(mode="json"),
        }


class TrainingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_tokens: int = Field(ge=PHASE_U_MINIMUM_EFFECTIVE_TOKENS)
    effective_unique_training_tokens: int = Field(ge=0)
    tokens_per_optimizer_step: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    multiplier_vs_phase_t: float = Field(ge=10_000)

    @model_validator(mode="after")
    def _budget(self) -> TrainingBudget:
        if self.effective_unique_training_tokens < self.requested_tokens:
            raise ValueError(
                "Campaign budget exceeds effective unique training tokens."
            )
        if (
            self.optimizer_steps * self.tokens_per_optimizer_step
            < self.requested_tokens
        ):
            raise ValueError(
                "Optimizer steps do not supply the requested token budget."
            )
        return self


def training_budget(
    *,
    accounting: CorpusAccounting,
    batch_size: int,
    context_length: int,
    gradient_accumulation_steps: int = 1,
    requested_tokens: int = PHASE_U_MINIMUM_EFFECTIVE_TOKENS,
) -> TrainingBudget:
    if requested_tokens < PHASE_U_MINIMUM_EFFECTIVE_TOKENS:
        raise ValueError(
            "Phase U cannot be configured below the 10,000× token minimum."
        )
    tokens_per_step = batch_size * context_length * gradient_accumulation_steps
    steps = (requested_tokens + tokens_per_step - 1) // tokens_per_step
    return TrainingBudget(
        requested_tokens=requested_tokens,
        effective_unique_training_tokens=accounting.effective_unique_training_tokens,
        tokens_per_optimizer_step=tokens_per_step,
        optimizer_steps=steps,
        multiplier_vs_phase_t=requested_tokens / PHASE_T_CANDIDATE_TOKENS,
    )


class ResourceAccounting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    measured_tokens_per_second: float = Field(gt=0)
    tokens_target: int = Field(gt=0)
    estimated_training_seconds: float = Field(gt=0)
    maximum_wall_seconds: float | None = Field(default=None, gt=0)
    disk_usage_bytes: int = Field(ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    status: str
    rationale: str


def assess_resources(
    *,
    tokens_per_second: float,
    tokens_target: int,
    maximum_wall_seconds: float | None,
    disk_usage_bytes: int,
    peak_memory_bytes: int | None,
) -> ResourceAccounting:
    estimated = tokens_target / tokens_per_second
    blocked = maximum_wall_seconds is not None and estimated > maximum_wall_seconds
    return ResourceAccounting(
        measured_tokens_per_second=tokens_per_second,
        tokens_target=tokens_target,
        estimated_training_seconds=estimated,
        maximum_wall_seconds=maximum_wall_seconds,
        disk_usage_bytes=disk_usage_bytes,
        peak_memory_bytes=peak_memory_bytes,
        status="PHASE_U_RESOURCE_BLOCKED" if blocked else "READY_FOR_FULL_CAMPAIGN",
        rationale=(
            "Measured throughput projects beyond the explicit campaign wall-clock boundary."
            if blocked
            else "Measured throughput supports the explicit full-token campaign boundary."
        ),
    )


class PhaseUExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    architecture_id: str
    parameter_count: int = Field(gt=0)
    architecture_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    dataset_release_id: str
    dataset_sha256: str
    mixture_sha256: str
    training_config_sha256: str
    initialization_seed: int
    initialization_method: str = "RANDOM"
    requested_tokens: int = Field(ge=PHASE_U_MINIMUM_EFFECTIVE_TOKENS)
    tokens_per_optimizer_step: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    checkpoint_steps: tuple[int, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str

    @model_validator(mode="after")
    def _integrity(self) -> PhaseUExperiment:
        if self.initialization_method != "RANDOM":
            raise ValueError("Phase U accepts only random proprietary initialization.")
        if (
            not self.checkpoint_steps
            or self.checkpoint_steps[-1] != self.optimizer_steps
        ):
            raise ValueError("The training curve must retain the final checkpoint.")
        expected = canonical_hash(
            self.model_dump(mode="json", exclude={"manifest_sha256", "created_at"})
        )
        if self.manifest_sha256 and expected != self.manifest_sha256:
            raise ValueError("Phase U experiment changed; choose a new experiment ID.")
        return self


def create_experiment(
    *,
    experiment_id: str,
    architecture: ModelArchitectureRecord,
    tokenizer_id: str,
    dataset_release_id: str,
    dataset_sha256: str,
    mixture: MixtureConfiguration,
    training_config_sha256: str,
    initialization_seed: int,
    budget: TrainingBudget,
) -> PhaseUExperiment:
    fractions = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
    checkpoints = tuple(
        sorted(
            {
                max(
                    1,
                    min(budget.optimizer_steps, round(budget.optimizer_steps * value)),
                )
                for value in fractions
            }
        )
    )
    unsigned = PhaseUExperiment(
        experiment_id=experiment_id,
        architecture_id=architecture.model_id,
        parameter_count=architecture.parameter_count,
        architecture_sha256=architecture.configuration_sha256,
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=architecture.tokenizer_sha256,
        dataset_release_id=dataset_release_id,
        dataset_sha256=dataset_sha256,
        mixture_sha256=mixture.configuration_sha256,
        training_config_sha256=training_config_sha256,
        initialization_seed=initialization_seed,
        requested_tokens=budget.requested_tokens,
        tokens_per_optimizer_step=budget.tokens_per_optimizer_step,
        optimizer_steps=budget.optimizer_steps,
        checkpoint_steps=checkpoints,
        manifest_sha256="",
    )
    return unsigned.model_copy(
        update={
            "manifest_sha256": canonical_hash(
                unsigned.model_dump(
                    mode="json", exclude={"manifest_sha256", "created_at"}
                )
            )
        }
    )


class PhaseUResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    corpus: CorpusAccounting
    mixture: MixtureExposure
    resources: ResourceAccounting
    attribution_id: str | None = None
    phase_s_recommendation_id: str | None = None
    verdict: LearningVerdict
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def learning_verdict(
    *,
    attribution: LearningAttribution | None,
    resources: ResourceAccounting,
    replicated: bool = False,
    valid: bool = True,
) -> LearningVerdict:
    if not valid:
        return LearningVerdict.INVALID_EXPERIMENT
    if resources.status == "PHASE_U_RESOURCE_BLOCKED":
        return LearningVerdict.PHASE_U_RESOURCE_BLOCKED
    if attribution is None:
        return LearningVerdict.INVALID_EXPERIMENT
    if attribution.regression_rate > 0:
        return LearningVerdict.REGRESSION
    if attribution.evidence is LearningEvidenceLevel.TRAINING_SIGNAL_ONLY:
        return LearningVerdict.TRAINING_SIGNAL_ONLY
    if attribution.evidence is LearningEvidenceLevel.NO_LEARNING_EVIDENCE:
        return LearningVerdict.NO_LEARNING_EVIDENCE
    if (
        attribution.evidence is LearningEvidenceLevel.STRONG_LEARNING_EVIDENCE
        and replicated
    ):
        return LearningVerdict.REPLICATED_CAPABILITY_ACQUISITION
    if attribution.evidence is LearningEvidenceLevel.GENERALIZATION_EVIDENCE:
        return LearningVerdict.GENERALIZED_CAPABILITY_ACQUISITION
    if attribution.target_capability_delta > 0:
        return LearningVerdict.CAPABILITY_ACQUISITION
    return LearningVerdict.NO_LEARNING_EVIDENCE


class PhaseUStore:
    _CATEGORIES = frozenset(
        {"sources", "corpora", "mixtures", "experiments", "resources", "results"}
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_u"
        for category in self._CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase U artifact category: {category}")
        path = self.root / category / f"{identifier}.json"
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    f"Immutable Phase U artifact already exists: {identifier}"
                )
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)


class PhaseUEngine:
    def __init__(self, root: Path) -> None:
        self.store = PhaseUStore(root)

    def record_source(self, source: SourceProvenance) -> SourceProvenance:
        self.store.write_immutable("sources", source.source_id, source)
        return source

    def record_corpus(self, accounting: CorpusAccounting) -> CorpusAccounting:
        self.store.write_immutable("corpora", accounting.dataset_release_id, accounting)
        return accounting

    def record_mixture(self, mixture: MixtureConfiguration) -> MixtureConfiguration:
        self.store.write_immutable("mixtures", mixture.mixture_id, mixture)
        return mixture

    def record_experiment(self, experiment: PhaseUExperiment) -> PhaseUExperiment:
        self.store.write_immutable("experiments", experiment.experiment_id, experiment)
        return experiment

    def record_result(self, result: PhaseUResult) -> PhaseUResult:
        self.store.write_immutable("results", result.experiment_id, result)
        return result
