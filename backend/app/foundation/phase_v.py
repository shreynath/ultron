"""Phase V controlled capability-acquisition laboratory artifacts."""

from __future__ import annotations

import math
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.foundation.checkpoints import canonical_hash

PHASE_V_MINIMUM_TOKENS = 20_000_000


class PhaseVVerdict(str, Enum):
    NO_LEARNING_EVIDENCE = "NO_LEARNING_EVIDENCE"
    TRAINING_SIGNAL_ONLY = "TRAINING_SIGNAL_ONLY"
    PARTIAL_CAPABILITY_ACQUISITION = "PARTIAL_CAPABILITY_ACQUISITION"
    CAPABILITY_ACQUISITION = "CAPABILITY_ACQUISITION"
    GENERALIZED_CAPABILITY_ACQUISITION = "GENERALIZED_CAPABILITY_ACQUISITION"
    REPLICATED_CAPABILITY_ACQUISITION = "REPLICATED_CAPABILITY_ACQUISITION"
    REGRESSION = "REGRESSION"
    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


class CapabilityExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    example_id: str
    capability_id: str
    skill: str
    difficulty: int = Field(ge=0, le=5)
    structure_id: str
    source_id: str
    license: str
    content_hash: str
    split: str
    prompt: str
    answer: str

    @model_validator(mode="after")
    def _valid(self) -> CapabilityExample:
        if self.split not in {
            "TRAIN",
            "VALIDATION",
            "TEST_DIRECT",
            "TEST_SURFACE",
            "TEST_STRUCTURAL",
            "TEST_COUNTERFACTUAL",
            "TEST_COMPOSITIONAL",
            "REGRESSION",
        }:
            raise ValueError("Unknown Phase V split.")
        if self.license.upper() in {"", "UNSPECIFIED"}:
            raise ValueError("Every Phase V example needs explicit license provenance.")
        return self


class CapabilityCurriculum(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    capability_id: str
    version: str
    selection_reason: str
    examples: tuple[CapabilityExample, ...]
    semantic_contamination_detection: str = "LIMITED"
    curriculum_sha256: str

    @model_validator(mode="after")
    def _separation(self) -> CapabilityCurriculum:
        if len({example.example_id for example in self.examples}) != len(self.examples):
            raise ValueError("Curriculum example IDs must be unique.")
        train = {
            example.content_hash
            for example in self.examples
            if example.split == "TRAIN"
        }
        evaluation = {
            example.content_hash
            for example in self.examples
            if example.split != "TRAIN"
        }
        if train & evaluation:
            raise ValueError("Train/evaluation content hashes overlap.")
        expected = canonical_hash(
            self.model_dump(mode="json", exclude={"curriculum_sha256"})
        )
        if self.curriculum_sha256 and expected != self.curriculum_sha256:
            raise ValueError("Curriculum changed; create a new version.")
        return self


def create_curriculum(**values: object) -> CapabilityCurriculum:
    unsigned = CapabilityCurriculum(**values, curriculum_sha256="")
    return unsigned.model_copy(
        update={
            "curriculum_sha256": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"curriculum_sha256"})
            )
        }
    )


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    control_experiment_id: str | None
    changed_variable: str
    primary_capability_id: str
    curriculum_sha256: str
    model_config_sha256: str
    tokenizer_sha256: str
    dataset_release_ids: tuple[str, ...]
    training_tokens: int = Field(ge=PHASE_V_MINIMUM_TOKENS)
    batch_size: int = Field(gt=0)
    context_length: int = Field(gt=0)
    optimizer: str
    learning_rate: float = Field(gt=0)
    seed: int
    requested_target_fraction: float = Field(ge=0, le=1)
    manifest_sha256: str

    @model_validator(mode="after")
    def _manifest(self) -> ExperimentManifest:
        if not self.dataset_release_ids:
            raise ValueError("A Phase V experiment needs immutable dataset releases.")
        expected = canonical_hash(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 and expected != self.manifest_sha256:
            raise ValueError("Experiment manifest changed; choose a new ID.")
        return self


def create_manifest(**values: object) -> ExperimentManifest:
    unsigned = ExperimentManifest(**values, manifest_sha256="")
    return unsigned.model_copy(
        update={
            "manifest_sha256": canonical_hash(
                unsigned.model_dump(mode="json", exclude={"manifest_sha256"})
            )
        }
    )


class GradedScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    syntax: float = Field(ge=0, le=1)
    concept: float = Field(ge=0, le=1)
    reasoning: float = Field(ge=0, le=1)
    answer: float = Field(ge=0, le=1)


def grade_addition(*, output: str, expected: str) -> GradedScore:
    """Deterministic, transparent partial credit for an integer answer task."""
    normalized = "".join(
        character for character in output if character.isdigit() or character == "-"
    )
    syntax = float(bool(normalized))
    answer = float(normalized == expected)
    return GradedScore(syntax=syntax, concept=answer, reasoning=answer, answer=answer)


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    direct: float = Field(ge=0, le=1)
    surface: float = Field(ge=0, le=1)
    structural: float = Field(ge=0, le=1)
    counterfactual: float = Field(ge=0, le=1)
    compositional: float = Field(ge=0, le=1)
    regression: float = Field(ge=0, le=1)
    composite_generalization: float = Field(ge=0, le=1)
    n: int = Field(ge=1)
    wilson_low: float = Field(ge=0, le=1)
    wilson_high: float = Field(ge=0, le=1)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError(
            "Wilson interval requires bounded successes and positive total."
        )
    p, denominator = successes / total, 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


class DensityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_target_fraction: float = Field(ge=0, le=1)
    observed_target_fraction: float = Field(ge=0, le=1)
    target_tokens: int = Field(ge=0)
    total_tokens: int = Field(gt=0)
    within_tolerance: bool


def observe_density(
    *, requested: float, target_tokens: int, total_tokens: int, tolerance: float = 0.02
) -> DensityObservation:
    observed = target_tokens / total_tokens
    return DensityObservation(
        requested_target_fraction=requested,
        observed_target_fraction=observed,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        within_tolerance=abs(observed - requested) <= tolerance,
    )


def verdict(
    *,
    baseline: EvaluationSummary,
    trained: EvaluationSummary,
    loss_delta: float | None,
    replicated: bool = False,
    resource_blocked: bool = False,
) -> PhaseVVerdict:
    if resource_blocked:
        return PhaseVVerdict.RESOURCE_BLOCKED
    direct_gain = trained.direct - baseline.direct
    generalization_gain = (
        trained.composite_generalization - baseline.composite_generalization
    )
    if trained.regression < baseline.regression:
        return PhaseVVerdict.REGRESSION
    if direct_gain > 0 and generalization_gain > 0:
        return (
            PhaseVVerdict.REPLICATED_CAPABILITY_ACQUISITION
            if replicated
            else PhaseVVerdict.GENERALIZED_CAPABILITY_ACQUISITION
        )
    if direct_gain > 0:
        return (
            PhaseVVerdict.REPLICATED_CAPABILITY_ACQUISITION
            if replicated
            else PhaseVVerdict.CAPABILITY_ACQUISITION
        )
    if any(
        getattr(trained, name) > getattr(baseline, name)
        for name in ("surface", "structural", "counterfactual", "compositional")
    ):
        return PhaseVVerdict.PARTIAL_CAPABILITY_ACQUISITION
    return (
        PhaseVVerdict.TRAINING_SIGNAL_ONLY
        if loss_delta is not None and loss_delta < 0
        else PhaseVVerdict.NO_LEARNING_EVIDENCE
    )


class PhaseVStore:
    def __init__(self, root: Path) -> None:
        self.root = root / "phase_v"
        self.root.mkdir(parents=True, exist_ok=True)

    def write_immutable(self, identifier: str, value: BaseModel) -> None:
        path, payload = (
            self.root / f"{identifier}.json",
            value.model_dump_json(indent=2) + "\n",
        )
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                "Immutable Phase V artifact already exists with different content."
            )
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
