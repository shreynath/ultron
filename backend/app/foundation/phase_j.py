"""Phase J evidence records and measurements for scaled from-scratch training."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import (
    CheckpointManifest,
    CheckpointStore,
    canonical_hash,
)
from app.foundation.data import DatasetRelease, ProcessedDocument, TokenizedRelease
from app.foundation.generation import (
    GenerationConfig,
    UltronGenerator,
    degeneration_metrics,
)
from app.foundation.model import UltronModelConfig
from app.foundation.tokenizer import UltronTokenizer


class PhaseJRunRecord(BaseModel):
    """Complete append-only run metadata; failures use the same schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    hypothesis: str
    model_version: str
    tokenizer_version: str
    dataset_version: str
    training_config: dict[str, Any]
    seed: int
    hardware: dict[str, Any]
    token_budget: int
    actual_tokens: int
    duration_seconds: float
    throughput_tokens_per_second: float | None
    peak_memory_bytes: int | None
    swap_delta_bytes: int | None
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    generation_metrics: dict[str, Any] = Field(default_factory=dict)
    capability_metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    result: str
    conclusion: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PhaseJRegistry:
    def __init__(self, root: Path) -> None:
        self.path = root / "experiments" / "phase-j.jsonl"

    def record(self, record: PhaseJRunRecord) -> PhaseJRunRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = {json.loads(line)["experiment_id"] for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()} if self.path.exists() else set()
        if record.experiment_id in existing:
            raise ValueError(f"Phase J experiment already exists: {record.experiment_id}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record

    def records(self) -> tuple[PhaseJRunRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(PhaseJRunRecord.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())


class PhaseJDataReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    tokenizer_version: str
    tokenizer_sha256: str
    documents: int
    source_families: int
    raw_characters: int
    raw_tokens: int
    training_tokens: int
    characters_per_token: float
    tokens_per_document: float
    categories: dict[str, dict[str, int | float]]
    source_family_isolation: dict[str, Any]
    duplicate_statistics: dict[str, int | float]
    filtered_statistics: dict[str, int]
    provenance_coverage: float
    license_coverage: float
    quality_distribution: dict[str, int]
    instruction_percent: float
    code_percent: float
    math_percent: float
    source_sha256: str


def phase_j_data_report(release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer) -> PhaseJDataReport:
    """Report composition from immutable release and packed-token manifests."""
    release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenized = TokenizedRelease.model_validate_json((tokenized_dir / "manifest.json").read_text(encoding="utf-8"))
    if tokenized.tokenizer_sha256 != tokenizer.sha256:
        raise ValueError("Tokenized release does not match the requested tokenizer.")
    documents = [ProcessedDocument.model_validate_json(line) for line in (release_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    categories: dict[str, dict[str, int | float]] = {}
    for category in sorted({document.classification.value for document in documents}):
        selected = [document for document in documents if document.classification.value == category]
        raw_characters = sum(len(document.content) for document in selected)
        category_tokens = sum(len(tokenizer.encode(document.content)) + 1 for document in selected)
        categories[category] = {"documents": len(selected), "raw_characters": raw_characters, "raw_tokens": category_tokens, "percent_documents": round(len(selected) / max(1, len(documents)), 6)}
    quality_distribution: dict[str, int] = {"high_ge_0.75": 0, "medium_0.5_to_0.749": 0, "low_lt_0.5": 0}
    for document in documents:
        quality_distribution["high_ge_0.75" if document.quality_score >= 0.75 else ("medium_0.5_to_0.749" if document.quality_score >= 0.5 else "low_lt_0.5")] += 1
    licensed = sum(document.provenance.license not in {"", "UNSPECIFIED"} and document.provenance.license_evidence is not None for document in documents)
    category_docs = release.category_breakdown
    return PhaseJDataReport(
        release_id=release.release_id, tokenizer_version=tokenizer.version, tokenizer_sha256=tokenizer.sha256,
        documents=len(documents), source_families=len({document.provenance.source_group or document.provenance.source_identifier for document in documents}),
        raw_characters=sum(len(document.content) for document in documents), raw_tokens=tokenized.token_count,
        training_tokens=sum(shard.token_count for shard in tokenized.shards if shard.split == "train"),
        characters_per_token=round(tokenized.raw_character_count / max(1, tokenized.token_count), 6), tokens_per_document=round(tokenized.token_count / max(1, len(documents)), 6),
        categories=categories, source_family_isolation=release.leakage_checks, duplicate_statistics=release.duplicate_statistics,
        filtered_statistics=release.filtered_statistics, provenance_coverage=release.provenance_coverage,
        license_coverage=round(licensed / max(1, len(documents)), 6), quality_distribution=quality_distribution,
        instruction_percent=round(category_docs.get("INSTRUCTION", 0) / max(1, release.document_count), 6),
        code_percent=round(category_docs.get("CODE", 0) / max(1, release.document_count), 6),
        math_percent=round(category_docs.get("MATHEMATICS", 0) / max(1, release.document_count), 6),
        source_sha256=hashlib.sha256((release_dir / "documents.jsonl").read_bytes()).hexdigest(),
    )


def verify_phase_j_lineage(checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig) -> CheckpointManifest:
    manifest = CheckpointStore.inspect(checkpoint)
    if not isinstance(manifest, CheckpointManifest):
        raise TypeError("Phase J requires a complete ULTRON checkpoint manifest.")
    if manifest.lineage.get("origin") != "ULTRON_FROM_SCRATCH" or manifest.lineage.get("initialization") != "RANDOM" or manifest.lineage.get("pretrained_weights") != "NONE":
        raise ValueError("Checkpoint does not have valid random-init ULTRON lineage.")
    if manifest.tokenizer_sha256 != tokenizer.sha256 or manifest.tokenizer_version != tokenizer.version:
        raise ValueError("Checkpoint tokenizer lineage does not match.")
    if manifest.architecture_config_sha256 != canonical_hash(model_config.model_dump(mode="json")):
        raise ValueError("Checkpoint model configuration does not match.")
    return manifest


def checkpoint_generation_curve(
    checkpoints: list[Path], tokenizer: UltronTokenizer, model_config: UltronModelConfig, prompts: tuple[str, ...],
    *, config: GenerationConfig | None = None,
) -> tuple[dict[str, Any], ...]:
    """Evaluate unmodified generator output at each supplied checkpoint."""
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        generator = UltronGenerator(checkpoint, tokenizer, model_config)
        per_prompt = []
        for prompt in prompts:
            result = generator.generate(prompt, config or GenerationConfig(max_new_tokens=64, temperature=0))
            per_prompt.append({"prompt": prompt, "result": result.model_dump(mode="json"), "metrics": degeneration_metrics(result)})
        values = [item["metrics"] for item in per_prompt]
        records.append({"checkpoint": str(checkpoint), "checkpoint_id": generator.manifest.checkpoint_id, "tokens_consumed": generator.manifest.tokens_consumed, "mode": "MODEL_ONLY_RAW_GENERATION", "records": per_prompt, "aggregate": {"repetition_2gram": round(sum(float(item["repetition_2gram"]) for item in values) / len(values), 6), "eos_rate": round(sum(float(item["eos_rate"]) for item in values) / len(values), 6), "entropy_bits": round(sum(float(item["entropy_bits"] or 0) for item in values) / len(values), 6)}})
    return tuple(records)
