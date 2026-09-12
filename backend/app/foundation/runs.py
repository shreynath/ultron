"""Append-safe local registry for auditable foundation-model training runs."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INVALID = "INVALID"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class CorpusTier(str, Enum):
    """The scale classification of the immutable corpus used by a run."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    DEVELOPMENT = "DEVELOPMENT"
    FORMAL = "FORMAL"


class ExperimentTier(str, Enum):
    INFRASTRUCTURE = "INFRASTRUCTURE"
    DEVELOPMENT = "DEVELOPMENT"
    FORMAL = "FORMAL"


class EvidenceClass(str, Enum):
    INFRASTRUCTURE_ONLY = "INFRASTRUCTURE_ONLY"
    DEVELOPMENT_SCALE = "DEVELOPMENT_SCALE"
    FORMAL_PHASE_L = "FORMAL_PHASE_L"
    HARDWARE_ONLY = "HARDWARE_ONLY"


class FormalGateStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    READY = "READY"
    SIZE_FAILURE = "SIZE_FAILURE"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    BALANCE_FAILURE = "BALANCE_FAILURE"


class DevelopmentStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    READY = "READY"
    BELOW_TARGET = "BELOW_TARGET"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    BALANCE_FAILURE = "BALANCE_FAILURE"


class TrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_id: str
    model_version: str
    parent_checkpoint: str | None = None
    dataset_release: str
    tokenizer_version: str
    training_config: dict[str, Any]
    # These fields are deliberately first-class registry data, rather than
    # prose in a report.  A reporting consumer can therefore never silently
    # upgrade a development run into a formal Phase L result.
    corpus_id: str = "unreleased-infrastructure-text"
    corpus_tier: CorpusTier = CorpusTier.INFRASTRUCTURE
    token_count: int = 0
    evidence_class: EvidenceClass = EvidenceClass.INFRASTRUCTURE_ONLY
    experiment_tier: ExperimentTier = ExperimentTier.INFRASTRUCTURE
    formal_gate_status: FormalGateStatus = FormalGateStatus.NOT_APPLICABLE
    development_status: DevelopmentStatus = DevelopmentStatus.NOT_APPLICABLE
    model_parameters: int = 0
    training_tokens: int = 0
    validation_tokens: int = 0
    test_tokens: int = 0
    hardware: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    steps: int = 0
    tokens: int = 0
    train_loss: float | None = None
    validation_loss: float | None = None
    perplexity: float | None = None
    tokens_per_second: float | None = None
    peak_memory: int | None = None
    peak_swap: int | None = None
    status: RunStatus = RunStatus.CREATED

    def model_post_init(self, __context: Any, /) -> None:
        expected = {
            ExperimentTier.INFRASTRUCTURE: EvidenceClass.INFRASTRUCTURE_ONLY,
            ExperimentTier.DEVELOPMENT: EvidenceClass.DEVELOPMENT_SCALE,
            ExperimentTier.FORMAL: EvidenceClass.FORMAL_PHASE_L,
        }[self.experiment_tier]
        if self.evidence_class is not expected:
            raise ValueError("experiment_tier and evidence_class must agree.")
        if self.corpus_tier.value != self.experiment_tier.value:
            raise ValueError("corpus_tier and experiment_tier must agree for a training run.")
        if self.experiment_tier is ExperimentTier.FORMAL and self.formal_gate_status is not FormalGateStatus.READY:
            raise ValueError("FORMAL_PHASE_L runs require a READY formal corpus gate.")


class TrainingRunRegistry:
    def __init__(self, root: Path) -> None:
        self.path = root / "training" / "runs.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}\n", encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, records: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def create(self, run: TrainingRun) -> TrainingRun:
        records = self._read()
        if run.run_id in records:
            raise ValueError(f"Training run already exists: {run.run_id}")
        records[run.run_id] = run.model_dump(mode="json")
        self._write(records)
        return run

    def update(self, run_id: str, **changes: Any) -> TrainingRun:
        records = self._read()
        if run_id not in records:
            raise KeyError(run_id)
        run = TrainingRun.model_validate(records[run_id]).model_copy(update=changes)
        records[run_id] = run.model_dump(mode="json")
        self._write(records)
        return run

    def get(self, run_id: str) -> TrainingRun:
        return TrainingRun.model_validate(self._read()[run_id])
