"""Phase T: immutable proprietary-model scaling evidence.

Phase T registers and assesses larger *from-scratch* ULTRON architectures. It
does not provide a second trainer: Phase O and FoundationTrainer still own all
weight updates and checkpoints.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.phase_r import LearningAttribution
from app.foundation.tokenizer import UltronTokenizer


class CampaignStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PAUSED_FOR_REVIEW = "PAUSED_FOR_REVIEW"
    REJECTED = "REJECTED"


class ScalingDiagnosis(str, Enum):
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    DATA_LOW_QUALITY = "DATA_LOW_QUALITY"
    TRAINING_BUDGET_TOO_SMALL = "TRAINING_BUDGET_TOO_SMALL"
    OPTIMIZATION_FAILURE = "OPTIMIZATION_FAILURE"
    MODEL_CAPACITY_LIMIT = "MODEL_CAPACITY_LIMIT"
    TOKENIZER_LIMIT = "TOKENIZER_LIMIT"
    EVALUATION_LIMIT = "EVALUATION_LIMIT"
    GENERALIZATION_FAILURE = "GENERALIZATION_FAILURE"
    REGRESSION = "REGRESSION"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    NO_CAPABILITY_EVIDENCE = "NO_CAPABILITY_EVIDENCE"


class ModelArchitectureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    architecture: str = "ULTRON_DECODER_ONLY_TRANSFORMER"
    architecture_version: str
    parameter_count: int = Field(gt=0)
    embedding_dimension: int = Field(gt=0)
    number_of_layers: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    kv_heads: int | str
    ffn_dimension: int = Field(gt=0)
    vocabulary_size: int = Field(gt=0)
    context_length: int = Field(gt=0)
    normalization: str
    activation: str
    positional_encoding: str
    attention_type: str = "CAUSAL_MULTI_HEAD_ATTENTION"
    dtype: str
    initializer: str
    tokenizer_id: str
    tokenizer_sha256: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    configuration_sha256: str

    def verify_integrity(self) -> None:
        if (
            canonical_hash(
                self.model_dump(
                    mode="json", exclude={"configuration_sha256", "created_at"}
                )
            )
            != self.configuration_sha256
        ):
            raise ValueError(
                "Model architecture record changed; register a new model ID."
            )


def register_architecture(
    *,
    model_config: UltronModelConfig,
    tokenizer: UltronTokenizer,
    instantiate: bool = True,
) -> ModelArchitectureRecord:
    if (
        model_config.initialization != "RANDOM"
        or model_config.pretrained_weights != "NONE"
        or model_config.parent_model != "NONE"
    ):
        raise ValueError(
            "Phase T accepts only randomly initialized proprietary ULTRON architectures."
        )
    count = model_config.parameter_estimate()
    if instantiate:
        from mlx.utils import tree_flatten

        model = build_model(model_config)
        actual = sum(int(value.size) for _, value in tree_flatten(model.parameters()))
        if actual != count:
            raise ValueError(
                f"Architecture formula mismatch: expected {count}, constructed {actual}."
            )
        count = actual
    raw = {
        "model_id": model_config.model_name,
        "architecture_version": model_config.version,
        "parameter_count": count,
        "embedding_dimension": model_config.hidden_size,
        "number_of_layers": model_config.num_layers,
        "attention_heads": model_config.num_heads,
        "kv_heads": model_config.num_kv_heads
        or "NOT_APPLICABLE_EQUAL_TO_ATTENTION_HEADS",
        "ffn_dimension": model_config.intermediate_size,
        "vocabulary_size": model_config.vocab_size,
        "context_length": model_config.max_sequence_length,
        "normalization": model_config.normalization,
        "activation": model_config.activation,
        "positional_encoding": model_config.positional_encoding,
        "dtype": model_config.dtype,
        "initializer": model_config.initialization,
        "tokenizer_id": tokenizer.model_path.stem,
        "tokenizer_sha256": tokenizer.sha256,
    }
    unsigned = ModelArchitectureRecord(**raw, configuration_sha256="")
    return unsigned.model_copy(
        update={
            "configuration_sha256": canonical_hash(
                unsigned.model_dump(
                    mode="json", exclude={"configuration_sha256", "created_at"}
                )
            )
        }
    )


class TokenizerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tokenizer_id: str
    version: str
    sha256: str
    vocabulary_size: int
    special_tokens: tuple[str, ...]
    normalization: str = "SENTENCEPIECE_DEFAULT_NFKC"
    training_corpus_release: str
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    tokens_per_character: float
    tokens_per_word: float
    tokens_per_byte: float
    sequence_length_p50: int
    sequence_length_p95: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def profile_tokenizer(
    *, tokenizer: UltronTokenizer, texts: tuple[str, ...], training_corpus_release: str
) -> TokenizerProfile:
    if not texts:
        raise ValueError(
            "Tokenizer profiling requires representative owned/released text."
        )
    counts = [len(tokenizer.encode(text)) for text in texts]
    joined = "\n".join(texts)
    words = len(joined.split())
    ordered = sorted(counts)
    return TokenizerProfile(
        tokenizer_id=tokenizer.model_path.stem,
        version=tokenizer.version,
        sha256=tokenizer.sha256,
        vocabulary_size=tokenizer.vocab_size,
        special_tokens=(),
        training_corpus_release=training_corpus_release,
        character_count=len(joined),
        word_count=words,
        byte_count=len(joined.encode("utf-8")),
        token_count=sum(counts),
        tokens_per_character=sum(counts) / max(1, len(joined)),
        tokens_per_word=sum(counts) / max(1, words),
        tokens_per_byte=sum(counts) / max(1, len(joined.encode("utf-8"))),
        sequence_length_p50=ordered[len(ordered) // 2],
        sequence_length_p95=ordered[
            min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
        ],
    )


class HardwareFeasibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_unified_memory_bytes: int | None
    parameter_bytes: int
    optimizer_state_bytes_estimate: int
    gradient_bytes_estimate: int
    activation_bytes_estimate: int
    checkpoint_bytes_estimate: int
    estimated_working_set_bytes: int
    maximum_batch_size: int
    context_length: int
    status: str
    rationale: str


def feasibility_for(
    *,
    model: UltronModelConfig,
    batch_size: int,
    dtype_bytes: int = 4,
    unified_memory_bytes: int | None = None,
) -> HardwareFeasibility:
    # MLX implementation currently holds f32 parameters, gradients, and Adam m/v.
    if dtype_bytes <= 0 or batch_size < 1:
        raise ValueError("dtype bytes and batch size must be positive.")
    parameters = model.parameter_estimate()
    total = unified_memory_bytes
    parameter_bytes = parameters * dtype_bytes
    optimizer_bytes, gradient_bytes = parameters * dtype_bytes * 2, parameter_bytes
    activation_bytes = (
        batch_size
        * model.max_sequence_length
        * model.hidden_size
        * model.num_layers
        * dtype_bytes
        * 8
    )
    checkpoint_bytes = parameter_bytes + optimizer_bytes
    working = parameter_bytes + optimizer_bytes + gradient_bytes + activation_bytes
    feasible = total is None or working <= total * 0.5
    return HardwareFeasibility(
        total_unified_memory_bytes=total,
        parameter_bytes=parameter_bytes,
        optimizer_state_bytes_estimate=optimizer_bytes,
        gradient_bytes_estimate=gradient_bytes,
        activation_bytes_estimate=activation_bytes,
        checkpoint_bytes_estimate=checkpoint_bytes,
        estimated_working_set_bytes=working,
        maximum_batch_size=batch_size,
        context_length=model.max_sequence_length,
        status="FEASIBLE_CONSERVATIVE_ESTIMATE"
        if feasible
        else "EXCEEDS_CONSERVATIVE_MEMORY_BUDGET",
        rationale="Estimate includes f32 parameters, Adam m/v, gradients, and an 8× activation multiplier; it is not a measured allocation.",
    )


@dataclass(frozen=True)
class MixtureSample:
    release_id: str
    sequence_index: int


class DeterministicMixtureSampler:
    """Seeded weighted sampler; it selects releases before sequence selection."""

    def __init__(
        self,
        *,
        sequences_by_release: dict[str, tuple[object, ...]],
        weights: dict[str, float],
        seed: int,
    ) -> None:
        if (
            set(sequences_by_release) != set(weights)
            or not weights
            or any(value <= 0 for value in weights.values())
        ):
            raise ValueError(
                "Mixture releases and positive explicit weights must match."
            )
        self.sequences_by_release, self.weights, self.seed = (
            sequences_by_release,
            weights,
            seed,
        )
        if any(not sequences for sequences in sequences_by_release.values()):
            raise ValueError(
                "Mixture sampling requires a non-empty sequence set per release."
            )

    def plan(self, count: int) -> tuple[MixtureSample, ...]:
        import random

        rng, ids = random.Random(self.seed), tuple(sorted(self.weights))
        cumulative, total = [], sum(self.weights.values())
        running = 0.0
        for identifier in ids:
            running += self.weights[identifier] / total
            cumulative.append((running, identifier))
        output: list[MixtureSample] = []
        for _ in range(count):
            draw = rng.random()
            release = next(
                identifier for threshold, identifier in cumulative if draw <= threshold
            )
            output.append(
                MixtureSample(
                    release, rng.randrange(len(self.sequences_by_release[release]))
                )
            )
        return tuple(output)


class MixtureExposure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sampling_policy_hash: str
    sampling_seed: int
    planned_weights: dict[str, float]
    actual_token_counts: dict[str, int]
    actual_proportions: dict[str, float]
    total_tokens: int
    materially_diverged: bool


def exposure_for(
    *,
    weights: dict[str, float],
    samples: tuple[MixtureSample, ...],
    tokens_per_sample: int,
    seed: int,
    tolerance: float = 0.05,
) -> MixtureExposure:
    counts = Counter(sample.release_id for sample in samples)
    total_tokens = len(samples) * tokens_per_sample
    actual = {
        identifier: counts[identifier] / max(1, len(samples))
        for identifier in sorted(weights)
    }
    normalized = {
        identifier: value / sum(weights.values())
        for identifier, value in weights.items()
    }
    return MixtureExposure(
        sampling_policy_hash=canonical_hash({"weights": normalized, "seed": seed}),
        sampling_seed=seed,
        planned_weights=normalized,
        actual_token_counts={
            identifier: counts[identifier] * tokens_per_sample
            for identifier in sorted(weights)
        },
        actual_proportions=actual,
        total_tokens=total_tokens,
        materially_diverged=any(
            abs(actual[key] - normalized[key]) > tolerance for key in normalized
        ),
    )


class ScalingCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    campaign_id: str
    architecture_id: str
    architecture_hash: str
    tokenizer_sha256: str
    dataset_release_ids: tuple[str, ...]
    dataset_release_hashes: tuple[str, ...]
    mixture_policy_hash: str
    requested_tokens: int = Field(gt=0)
    maximum_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    max_wall_seconds: float | None = Field(default=None, gt=0)
    max_memory_bytes: int | None = Field(default=None, gt=0)
    max_disk_bytes: int | None = Field(default=None, gt=0)
    max_checkpoints: int = Field(gt=0)
    seed: int
    distributed_world_size: int = Field(default=1, ge=1)
    hpc_distribution: str = "NOT_IMPLEMENTED"
    status: CampaignStatus = CampaignStatus.PLANNED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    manifest_sha256: str

    def verify_integrity(self) -> None:
        if (
            len(self.dataset_release_ids) != len(self.dataset_release_hashes)
            or not self.dataset_release_ids
        ):
            raise ValueError(
                "Campaign requires aligned immutable dataset release identity."
            )
        if (
            canonical_hash(
                self.model_dump(mode="json", exclude={"manifest_sha256", "created_at"})
            )
            != self.manifest_sha256
        ):
            raise ValueError("Scaling campaign changed; create a new campaign ID.")


def create_campaign(**values: object) -> ScalingCampaign:
    unsigned = ScalingCampaign(**values, manifest_sha256="")
    return unsigned.model_copy(
        update={
            "manifest_sha256": canonical_hash(
                unsigned.model_dump(
                    mode="json", exclude={"manifest_sha256", "created_at"}
                )
            )
        }
    )


class ScalingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str
    campaign_id: str
    parameter_count: int
    training_tokens: int
    tokens_per_parameter: float
    baseline_validation_loss: float | None
    candidate_validation_loss: float | None
    validation_loss_delta: float | None
    baseline_capability: float | None
    candidate_capability: float | None
    capability_delta: float
    generalization_delta: float | None
    regression_rate: float
    weights_changed: bool
    throughput_tokens_per_second: float | None
    wall_seconds: float | None
    compute_units: float | None
    conclusion: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScalingChampion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    champion_id: str
    campaign_id: str
    previous_champion_id: str | None
    evidence_id: str
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def scaling_conclusion(*, attribution: LearningAttribution) -> str:
    if attribution.evidence.value in {
        "GENERALIZATION_EVIDENCE",
        "STRONG_LEARNING_EVIDENCE",
    }:
        return "GENERALIZED_LEARNING_EVIDENCE"
    if attribution.target_capability_delta > 0:
        return "CAPABILITY_IMPROVEMENT_WITHOUT_GENERALIZATION_EVIDENCE"
    if (
        attribution.training_loss_delta is not None
        and attribution.training_loss_delta < 0
    ):
        return "TRAINING_SIGNAL_ONLY"
    return "NO_LEARNING_EVIDENCE"


class PhaseTStore:
    _CATEGORIES = frozenset(
        {
            "architectures",
            "tokenizers",
            "feasibility",
            "mixtures",
            "campaigns",
            "evidence",
            "champions",
            "diagnoses",
        }
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_t"
        for category in self._CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase T artifact category: {category}")
        path, payload = (
            self.root / category / f"{identifier}.json",
            value.model_dump_json(indent=2) + "\n",
        )
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    f"Immutable Phase T artifact already exists: {identifier}"
                )
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)


class PhaseTEngine:
    def __init__(self, root: Path) -> None:
        self.store = PhaseTStore(root)

    def register_architecture(
        self, record: ModelArchitectureRecord
    ) -> ModelArchitectureRecord:
        record.verify_integrity()
        self.store.write_immutable("architectures", record.model_id, record)
        return record

    def record_campaign(self, campaign: ScalingCampaign) -> ScalingCampaign:
        campaign.verify_integrity()
        self.store.write_immutable("campaigns", campaign.campaign_id, campaign)
        return campaign

    def persist(self, category: str, identifier: str, value: BaseModel) -> BaseModel:
        self.store.write_immutable(category, identifier, value)
        return value
