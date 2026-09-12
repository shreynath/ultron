"""Immutable, hypothesis-led Phase I experiment records and candidate builders."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.data import DocumentCategory, ProcessedDocument
from app.foundation.model import UltronModelConfig


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    created_at: datetime
    hypothesis: str = Field(min_length=1)
    variables: dict[str, Any]
    controls: dict[str, Any]
    model_version: str
    dataset_version: str
    tokenizer_version: str
    training_config: dict[str, Any]
    seed: int
    hardware: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    conclusion: str = "PENDING"
    status: str = "PLANNED"


class ExperimentRegistry:
    """Append-only records; failed experiments are evidence, never cleanup work."""

    def __init__(self, root: Path, *, registry_id: str = "phase-i") -> None:
        if not registry_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in registry_id):
            raise ValueError("registry_id must contain lowercase letters, digits, or hyphens.")
        self.path = root / "experiments" / f"{registry_id}.jsonl"

    def record(self, value: ExperimentRecord) -> ExperimentRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = [json.loads(line)["experiment_id"] for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if value.experiment_id in existing:
                raise ValueError(f"Experiment ID already recorded: {value.experiment_id}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(value.model_dump_json() + "\n")
        return value

    def records(self) -> tuple[ExperimentRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(ExperimentRecord.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())


def parameter_matched_variants(base: UltronModelConfig) -> dict[str, UltronModelConfig]:
    """Return single-variable depth/width/FFN candidates around one baseline.

    The candidates are approximations, not a claim of exact parameter matching;
    callers must record the resulting parameter counts before comparison.
    """
    return {
        "baseline": base,
        "deeper": base.model_copy(update={"version": f"{base.version}-deeper", "num_layers": base.num_layers + 2, "hidden_size": max(16, base.hidden_size - 48), "intermediate_size": max(16, base.intermediate_size - 160)}),
        "wider": base.model_copy(update={"version": f"{base.version}-wider", "num_layers": max(1, base.num_layers - 1), "hidden_size": base.hidden_size + 48, "intermediate_size": base.intermediate_size + 160}),
        "ffn-expanded": base.model_copy(update={"version": f"{base.version}-ffn", "intermediate_size": base.intermediate_size + max(32, base.hidden_size)}),
    }


def mixture_selection(
    documents: list[ProcessedDocument], mixture: dict[DocumentCategory, float], *, salt: str,
) -> list[ProcessedDocument]:
    """Deterministically downsample categories; it never fabricates instruction data.

    Percentages are targets only.  Missing categories remain visible through the
    result, which is preferable to duplicating text or silently filling a gap.
    """
    if not mixture or any(weight < 0 for weight in mixture.values()):
        raise ValueError("Mixture weights must be non-negative and non-empty.")
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("At least one mixture weight must be positive.")
    by_category: dict[DocumentCategory, list[ProcessedDocument]] = {}
    for document in documents:
        by_category.setdefault(document.classification, []).append(document)
    available = sum(len(items) for items in by_category.values())
    selected: list[ProcessedDocument] = []
    for category, weight in mixture.items():
        target = round(available * weight / total)
        candidates = sorted(by_category.get(category, []), key=lambda item: hashlib.sha256(f"{salt}:{item.content_hash}".encode()).hexdigest())
        selected.extend(candidates[:target])
    return selected
