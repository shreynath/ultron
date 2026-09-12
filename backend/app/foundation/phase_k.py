"""Phase K scaling, corpus-audit, and hardware-frontier evidence tools.

The module deliberately separates allocation-free planning from instantiated
measurements.  A multi-hundred-million parameter candidate can be preflighted
without risking the workstation; its parameter count is only called measured
after a real ULTRON decoder has been instantiated and inspected.
"""
from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.data import DatasetRelease, ProcessedDocument, TokenizedRelease
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.scaling import (
    ResourcePreflight,
    ResourceRisk,
    resource_preflight,
)
from app.foundation.tokenizer import UltronTokenizer


class PhaseKStatus(str, Enum):
    PLANNED = "PLANNED"
    PREFLIGHTED = "PREFLIGHTED"
    PILOT = "PILOT"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    RESOURCE_INFEASIBLE = "RESOURCE_INFEASIBLE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    RESEARCH_BEST = "RESEARCH_BEST"
    PROMOTED = "PROMOTED"


class FailureClass(str, Enum):
    DATA_FAILURE = "DATA_FAILURE"
    TOKENIZER_FAILURE = "TOKENIZER_FAILURE"
    ARCHITECTURE_FAILURE = "ARCHITECTURE_FAILURE"
    OPTIMIZATION_FAILURE = "OPTIMIZATION_FAILURE"
    MEMORY_FAILURE = "MEMORY_FAILURE"
    THERMAL_FAILURE = "THERMAL_FAILURE"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    RESUME_FAILURE = "RESUME_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    CAPABILITY_FAILURE = "CAPABILITY_FAILURE"
    RESOURCE_INFEASIBLE = "RESOURCE_INFEASIBLE"


class ExecutionRecommendation(str, Enum):
    RUN = "RUN"
    RUN_WITH_SWAP = "RUN_WITH_SWAP"
    RUN_WITH_OPTIMIZATION = "RUN_WITH_OPTIMIZATION"
    HIGH_COST_RUN = "HIGH_COST_RUN"
    LIKELY_OOM = "LIKELY_OOM"


class SwapClass(str, Enum):
    NO_SWAP = "NO_SWAP"
    LOW_SWAP = "LOW_SWAP"
    MODERATE_SWAP = "MODERATE_SWAP"
    HIGH_SWAP = "HIGH_SWAP"
    SEVERE_SWAP = "SEVERE_SWAP"


class ExecutionTarget(str, Enum):
    LOCAL_M4_PRO = "LOCAL_M4_PRO"
    LOCAL_LARGER_APPLE_SILICON = "LOCAL_LARGER_APPLE_SILICON"
    LOCAL_GPU = "LOCAL_GPU"
    HPC_SINGLE_GPU = "HPC_SINGLE_GPU"
    HPC_MULTI_GPU = "HPC_MULTI_GPU"
    HPC_DISTRIBUTED = "HPC_DISTRIBUTED"


class DistributedTrainingConfig(BaseModel):
    """Portable planning schema. Only local single-process execution exists."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target: ExecutionTarget = ExecutionTarget.LOCAL_M4_PRO
    node_count: int = Field(default=1, ge=1)
    gpu_count: int = Field(default=0, ge=0)
    distributed_mode: str = "UNIMPLEMENTED"
    data_parallel: bool = False
    tensor_parallel: int = Field(default=1, ge=1)
    pipeline_parallel: int = Field(default=1, ge=1)
    sharding: str = "UNIMPLEMENTED"
    checkpoint_sharding: str = "UNIMPLEMENTED"


class ParameterBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: str
    trainable_parameters: int
    non_trainable_parameters: int = 0
    embedding_parameters: int
    attention_parameters: int
    ffn_parameters: int
    normalization_parameters: int
    output_head_parameters: int
    total_parameters: int


class PhaseKPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: str
    parameter_breakdown: ParameterBreakdown
    resource: ResourcePreflight
    available_disk_bytes: int | None
    minimum_checkpoint_free_bytes: int
    estimated_checkpoint_count: int
    estimated_total_checkpoint_bytes: int
    approximate_training_flops: float | None
    estimated_wall_seconds: float | None
    estimated_swap_requirement_bytes: int | None
    confidence: str
    recommendation: ExecutionRecommendation
    status: PhaseKStatus
    rationale: str


class PhaseKExperimentRecord(BaseModel):
    """Append-only fair-scaling record, including unsuccessful candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    experiment_class: str
    hypothesis: str
    status: PhaseKStatus
    model_scale: str
    model: dict[str, Any]
    parameter_count: int
    token_budget: int
    corpus_release: str
    corpus_hash: str
    tokenizer_version: str
    tokenizer_sha256: str
    training_config: dict[str, Any]
    seed: int
    hardware: dict[str, Any]
    estimated_memory: dict[str, Any] = Field(default_factory=dict)
    actual_memory: dict[str, Any] = Field(default_factory=dict)
    throughput: dict[str, Any] = Field(default_factory=dict)
    wall_time_seconds: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    generation_metrics: dict[str, Any] = Field(default_factory=dict)
    capability_metrics: dict[str, Any] = Field(default_factory=dict)
    failure_class: FailureClass | None = None
    conclusion: str
    artifacts: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PhaseKRegistry:
    def __init__(self, root: Path) -> None:
        self.path = root / "experiments" / "phase-k.jsonl"

    def records(self) -> tuple[PhaseKExperimentRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(PhaseKExperimentRecord.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def record(self, record: PhaseKExperimentRecord) -> PhaseKExperimentRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if record.experiment_id in {item.experiment_id for item in self.records()}:
            raise ValueError(f"Phase K experiment already exists: {record.experiment_id}")
        with self.path.open("a", encoding="utf-8") as output:
            output.write(record.model_dump_json() + "\n")
        return record


class PhaseKCorpusReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    documents: int
    tokenizer_tokens: int
    splits: dict[str, int]
    domains: dict[str, dict[str, int | float]]
    categories: dict[str, dict[str, int | float]]
    largest_domain_percent: float
    smallest_major_domain_percent: float | None
    source_families: int
    provenance_coverage: float
    license_coverage: float
    duplicate_statistics: dict[str, int | float]
    filtered_statistics: dict[str, int]
    leakage_checks: dict[str, Any]
    general_foundation_eligible: bool
    warnings: tuple[str, ...]
    source_sha256: str


def _bucket(name: str) -> str:
    if name.startswith(("token_embedding", "position_embedding")):
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".up." in name or ".down." in name:
        return "ffn"
    if "norm" in name:
        return "normalization"
    if name.startswith("output."):
        return "output"
    raise ValueError(f"Unclassified ULTRON parameter: {name}")


def parameter_breakdown(config: UltronModelConfig, *, instantiate: bool = True) -> ParameterBreakdown:
    """Count all ULTRON parameter tensors, normally from an instantiated model."""
    if instantiate:
        from mlx.utils import tree_flatten

        values = tree_flatten(build_model(config).parameters())
        buckets: Counter[str] = Counter()
        for name, value in values:
            buckets[_bucket(name)] += int(value.size)
        total = sum(buckets.values())
        return ParameterBreakdown(
            method="INSTANTIATED_MLX_PARAMETER_TREE",
            trainable_parameters=total,
            embedding_parameters=buckets["embedding"],
            attention_parameters=buckets["attention"],
            ffn_parameters=buckets["ffn"],
            normalization_parameters=buckets["normalization"],
            output_head_parameters=buckets["output"],
            total_parameters=total,
        )
    embedding = config.vocab_size * config.hidden_size + config.max_sequence_length * config.hidden_size
    attention = config.num_layers * 4 * config.hidden_size * config.hidden_size
    ffn = config.num_layers * 2 * config.hidden_size * config.intermediate_size
    normalization = config.num_layers * 2 * config.hidden_size + config.hidden_size
    output = config.vocab_size * config.hidden_size
    total = embedding + attention + ffn + normalization + output
    return ParameterBreakdown(
        method="EXACT_ARCHITECTURE_FORMULA_NOT_INSTANTIATED",
        trainable_parameters=total,
        embedding_parameters=embedding,
        attention_parameters=attention,
        ffn_parameters=ffn,
        normalization_parameters=normalization,
        output_head_parameters=output,
        total_parameters=total,
    )


def derive_scaled_config(target_parameters: int, *, vocab_size: int, context_length: int, version: str) -> UltronModelConfig:
    """Choose a decoder shape mechanically from a target, not a hand label.

    Candidates use a 64-wide head and an FFN near 8/3 of hidden size.  The
    closest exact architecture-formula count wins, with depth as a secondary
    preference.  The result is a planning configuration and must still pass an
    instantiated count before it is called measured.
    """
    if target_parameters < 1_000_000:
        raise ValueError("Scaling targets must be at least one million parameters.")
    candidates: list[UltronModelConfig] = []
    for layers in range(6, 37, 2):
        for hidden in range(384, 4097, 64):
            heads = hidden // 64
            intermediate = int(round((hidden * 8 / 3) / 64) * 64)
            candidate = UltronModelConfig(
                version=version,
                vocab_size=vocab_size,
                hidden_size=hidden,
                num_layers=layers,
                num_heads=heads,
                intermediate_size=intermediate,
                max_sequence_length=context_length,
            )
            count = parameter_breakdown(candidate, instantiate=False).total_parameters
            candidates.append(candidate)
            if count > target_parameters * 1.40:
                break
    return min(
        candidates,
        key=lambda item: (abs(parameter_breakdown(item, instantiate=False).total_parameters - target_parameters), -item.num_layers),
    )


def scaling_ladder(*, vocab_size: int, context_length: int) -> dict[str, UltronModelConfig]:
    targets = {"50m": 50_000_000, "100m": 100_000_000, "250m": 250_000_000, "500m": 500_000_000, "1b": 1_000_000_000}
    return {name: derive_scaled_config(target, vocab_size=vocab_size, context_length=context_length, version=f"0.7.0-phase-k-{name}") for name, target in targets.items()}


def phase_k_preflight(
    config: UltronModelConfig,
    *, batch_size: int,
    sequence_length: int,
    target_tokens: int | None = None,
    measured_tokens_per_second: float | None = None,
    checkpoint_count: int = 3,
    instantiate_count: bool = False,
    disk_path: Path | None = None,
) -> PhaseKPreflight:
    """Fail closed when memory or disk budgets cannot support a serious run."""
    if checkpoint_count < 1:
        raise ValueError("At least one checkpoint is required.")
    breakdown = parameter_breakdown(config, instantiate=instantiate_count)
    resource = resource_preflight(config, batch_size=batch_size, sequence_length=sequence_length)
    disk = shutil.disk_usage(disk_path or Path.cwd()).free
    checkpoint_total = resource.checkpoint_bytes_estimate * checkpoint_count
    flops = 6.0 * breakdown.total_parameters * target_tokens if target_tokens else None
    wall = target_tokens / measured_tokens_per_second if target_tokens and measured_tokens_per_second else None
    disk_ok = disk >= checkpoint_total
    available_memory = resource.available_memory_bytes
    estimated_swap = max(0, resource.total_bytes_estimate - available_memory) if available_memory is not None else None
    if not disk_ok:
        recommendation = ExecutionRecommendation.LIKELY_OOM
    elif resource.risk is ResourceRisk.SAFE:
        recommendation = ExecutionRecommendation.RUN
    elif resource.risk is ResourceRisk.CAUTION:
        recommendation = ExecutionRecommendation.RUN_WITH_SWAP
    elif resource.risk is ResourceRisk.HIGH_RISK:
        recommendation = ExecutionRecommendation.HIGH_COST_RUN
    else:
        recommendation = ExecutionRecommendation.LIKELY_OOM
    reasons = []
    if resource.risk is ResourceRisk.HIGH_RISK:
        reasons.append("resident estimate exceeds half of physical memory; use a bounded real micro-pilot")
    elif resource.risk is ResourceRisk.UNSUPPORTED:
        reasons.append("resident estimate exceeds 75% of physical memory; execution may still be attempted with a bounded micro-pilot")
    if not disk_ok:
        reasons.append("free disk is below planned checkpoint budget")
    return PhaseKPreflight(
        model_version=config.version,
        parameter_breakdown=breakdown,
        resource=resource,
        available_disk_bytes=disk,
        minimum_checkpoint_free_bytes=resource.checkpoint_bytes_estimate,
        estimated_checkpoint_count=checkpoint_count,
        estimated_total_checkpoint_bytes=checkpoint_total,
        approximate_training_flops=flops,
        estimated_wall_seconds=wall,
        estimated_swap_requirement_bytes=estimated_swap,
        confidence="LOW: MLX allocator, macOS compression, and swap behavior require real measurement.",
        recommendation=recommendation,
        status=PhaseKStatus.PREFLIGHTED,
        rationale="; ".join(reasons) if reasons else "Planning estimate only; a bounded real micro-pilot is required to establish feasibility.",
    )


def classify_swap(delta_bytes: int | None) -> SwapClass:
    """Classify observed swap as a performance characteristic, never failure."""
    if not delta_bytes or delta_bytes <= 0:
        return SwapClass.NO_SWAP
    if delta_bytes <= 256 * 1024 * 1024:
        return SwapClass.LOW_SWAP
    if delta_bytes <= 1024 * 1024 * 1024:
        return SwapClass.MODERATE_SWAP
    if delta_bytes <= 4 * 1024 * 1024 * 1024:
        return SwapClass.HIGH_SWAP
    return SwapClass.SEVERE_SWAP


def classify_swap_footprint(used_bytes: int | None) -> SwapClass:
    """Classify total observed swap footprint when a pilot starts nonzero."""
    return classify_swap(used_bytes)


def corpus_balance_report(release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer, *, maximum_major_domain_percent: float = 0.45) -> PhaseKCorpusReport:
    """Measure realized token and document composition, not intended mixture."""
    release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenized = TokenizedRelease.model_validate_json((tokenized_dir / "manifest.json").read_text(encoding="utf-8"))
    if tokenized.tokenizer_sha256 != tokenizer.sha256:
        raise ValueError("Tokenized corpus tokenizer hash does not match the requested tokenizer.")
    documents = [ProcessedDocument.model_validate_json(line) for line in (release_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    domains: dict[str, dict[str, int | float]] = {}
    categories: dict[str, dict[str, int | float]] = {}
    domain_docs: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    category_docs: Counter[str] = Counter()
    category_tokens: Counter[str] = Counter()
    licensed = 0
    for document in documents:
        domain = urlparse(document.provenance.source_url or "").netloc or "LOCAL"
        tokens = len(tokenizer.encode(document.content)) + 1
        domain_docs[domain] += 1
        domain_tokens[domain] += tokens
        category_docs[document.classification.value] += 1
        category_tokens[document.classification.value] += tokens
        licensed += int(document.provenance.license not in {"", "UNSPECIFIED"} and document.provenance.license_evidence is not None)
    total_tokens = sum(domain_tokens.values())
    for name in sorted(domain_docs):
        domains[name] = {"documents": domain_docs[name], "tokens": domain_tokens[name], "token_percent": round(domain_tokens[name] / max(1, total_tokens), 6)}
    for name in sorted(category_docs):
        categories[name] = {"documents": category_docs[name], "tokens": category_tokens[name], "token_percent": round(category_tokens[name] / max(1, total_tokens), 6)}
    percentages = [float(value["token_percent"]) for value in domains.values()]
    largest = max(percentages, default=0.0)
    major = [value for value in percentages if value >= 0.01]
    warnings = []
    if largest > maximum_major_domain_percent:
        warnings.append("DOMAIN_DOMINANCE_EXCEEDS_THRESHOLD")
    if release.leakage_checks.get("status") != "PASS":
        warnings.append("LEAKAGE_CHECK_FAILED")
    if licensed < len(documents):
        warnings.append("INCOMPLETE_LICENSE_EVIDENCE")
    return PhaseKCorpusReport(
        release_id=release.release_id,
        documents=len(documents),
        tokenizer_tokens=total_tokens,
        splits={"train": release.train_document_count, "validation": release.validation_document_count, "test": release.test_document_count},
        domains=domains,
        categories=categories,
        largest_domain_percent=largest,
        smallest_major_domain_percent=min(major) if major else None,
        source_families=len({item.provenance.source_group or item.provenance.source_identifier for item in documents}),
        provenance_coverage=release.provenance_coverage,
        license_coverage=round(licensed / max(1, len(documents)), 6),
        duplicate_statistics=release.duplicate_statistics,
        filtered_statistics=release.filtered_statistics,
        leakage_checks=release.leakage_checks,
        general_foundation_eligible=not warnings,
        warnings=tuple(warnings),
        source_sha256=hashlib.sha256((release_dir / "documents.jsonl").read_bytes()).hexdigest(),
    )


def phase_j_baseline_record() -> dict[str, Any]:
    """Immutable comparison record; the historical label and true count coexist."""
    return {
        "phase": "J",
        "historical_parameter_label": "~15.6M (legacy configuration estimate)",
        "instantiated_trainable_parameters": 14_750_592,
        "training_tokens": 10_000_384,
        "sealed_loss": 5.37593749538064,
        "sealed_perplexity": 216.1424098675792,
        "repeated_bigram_rate": 0.494681,
        "eos_rate": 0.0,
        "model_only_capability_score": 0.0,
        "assistant_promotion": "NOT_PROMOTED",
    }


def scaling_analysis(records: tuple[PhaseKExperimentRecord, ...]) -> dict[str, Any]:
    """Return only observed comparisons; never fit a law from sparse data."""
    completed = [record for record in records if record.status in {PhaseKStatus.COMPLETED, PhaseKStatus.RESEARCH_BEST} and record.validation_loss is not None]
    parameter_groups: dict[tuple[str, int, str], list[PhaseKExperimentRecord]] = {}
    for record in completed:
        parameter_groups.setdefault((record.corpus_release, record.token_budget, record.tokenizer_sha256), []).append(record)
    fair_parameter = [sorted(group, key=lambda item: item.parameter_count) for group in parameter_groups.values() if len(group) >= 2]
    return {
        "completed_runs": len(completed),
        "fair_parameter_comparisons": [
            [{"experiment_id": item.experiment_id, "parameters": item.parameter_count, "validation_loss": item.validation_loss} for item in group]
            for group in fair_parameter
        ],
        "conclusion": "INSUFFICIENT_EVIDENCE_FOR_SCALING_LAW" if len(fair_parameter) < 2 else "OBSERVED_LOCAL_TREND_ONLY",
    }
