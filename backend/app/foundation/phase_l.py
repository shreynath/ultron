"""Phase L evidence gates: corpus readiness, sustained runs, and economics.

These helpers deliberately report missing evidence rather than manufacturing a
training conclusion from a short allocator benchmark.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import CheckpointStore, sha256_file
from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DatasetRelease,
    ProcessedDocument,
    Provenance,
    SourceType,
    TokenizedRelease,
    TokenPacker,
)
from app.foundation.phase_k import corpus_balance_report
from app.foundation.runs import (
    CorpusTier,
    DevelopmentStatus,
    EvidenceClass,
    FormalGateStatus,
    TrainingRun,
)
from app.foundation.scaling import ScalingResult
from app.foundation.tokenizer import UltronTokenizer


class PhaseLReadiness(str, Enum):
    READY = "READY"
    BELOW_TARGET = "BELOW_TARGET"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    BALANCE_FAILURE = "BALANCE_FAILURE"


class SustainedRunClassification(str, Enum):
    NOT_RUN = "NOT_RUN"
    FAILED_EXECUTION = "FAILED_EXECUTION"
    NUMERICALLY_UNSTABLE = "NUMERICALLY_UNSTABLE"
    RESOURCE_UNSTABLE = "RESOURCE_UNSTABLE"
    CHECKPOINT_FAILURE = "CHECKPOINT_FAILURE"
    SUSTAINED_EXECUTION = "SUSTAINED_EXECUTION"
    SUSTAINED_EXECUTION_WITH_SWAP = "SUSTAINED_EXECUTION_WITH_SWAP"


class OneBFunctionalStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    TRAINING = "TRAINING"
    LEARNING_EVIDENCE = "LEARNING_EVIDENCE"
    FUNCTIONAL_CANDIDATE = "FUNCTIONAL_CANDIDATE"
    FUNCTIONAL_GATE_PASSED = "FUNCTIONAL_GATE_PASSED"
    FUNCTIONAL_GATE_FAILED = "FUNCTIONAL_GATE_FAILED"


class ScaleDecision(str, Enum):
    DO_NOT_SCALE = "DO_NOT_SCALE"
    READY_FOR_1B = "READY_FOR_1B"
    READY_FOR_2B = "READY_FOR_2B"
    READY_FOR_5B = "READY_FOR_5B"
    READY_FOR_10B = "READY_FOR_10B"
    READY_FOR_20B = "READY_FOR_20B"


class FunctionalFailure(str, Enum):
    DATA_SCALE_INSUFFICIENT = "DATA_SCALE_INSUFFICIENT"
    TRAINING_DURATION_INSUFFICIENT = "TRAINING_DURATION_INSUFFICIENT"
    TOKENIZER_FAILURE = "TOKENIZER_FAILURE"
    EOS_FAILURE = "EOS_FAILURE"
    PACKING_FAILURE = "PACKING_FAILURE"
    OPTIMIZER_FAILURE = "OPTIMIZER_FAILURE"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    GENERALIZATION_FAILURE = "GENERALIZATION_FAILURE"
    CAPABILITY_INSUFFICIENT = "CAPABILITY_INSUFFICIENT"
    CHECKPOINT_REPRODUCIBILITY_FAILURE = "CHECKPOINT_REPRODUCIBILITY_FAILURE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GenerationHealth(BaseModel):
    """Transparent MODEL_ONLY generation vector; no language-model judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    eos_rate: float | str
    mean_length: float | str
    repeated_token_rate: float | str
    repeated_bigram_rate: float | str
    distinct_1: float | str
    distinct_2: float | str
    entropy: float | str
    coherent_completion_rate: float | str = "INSUFFICIENT_EVIDENCE"
    instruction_following_rate: float | str = "INSUFFICIENT_EVIDENCE"
    malformed_output_rate: float | str = "INSUFFICIENT_EVIDENCE"


class PhaseLTrainingRecipeRecord(BaseModel):
    """Append-only reproducibility record for a meaningful Phase L run."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True)
    experiment_id: str
    phase: str = "L.1"
    evidence_class: EvidenceClass
    architecture_config: dict[str, Any] = Field(alias="model_config", serialization_alias="model_config")
    actual_parameter_count: int
    tokenizer_id: str
    tokenizer_vocab_size: int
    corpus_release_id: str
    corpus_token_count: int
    train_token_count: int
    validation_token_count: int
    test_token_count: int
    optimizer: str
    learning_rate: float
    scheduler: str
    warmup: int
    batch_size: int
    sequence_length: int
    gradient_accumulation: int
    weight_decay: float
    initialization: str
    precision: str
    seed: int
    training_steps: int
    training_tokens: int
    checkpoint_interval: int
    hardware: dict[str, Any]
    memory_peak: int | str = "INSUFFICIENT_EVIDENCE"
    swap_baseline: int | str = "INSUFFICIENT_EVIDENCE"
    swap_peak: int | str = "INSUFFICIENT_EVIDENCE"
    swap_final: int | str = "INSUFFICIENT_EVIDENCE"
    throughput: float | str = "INSUFFICIENT_EVIDENCE"
    wall_time: float | str = "INSUFFICIENT_EVIDENCE"
    train_loss: float | str = "INSUFFICIENT_EVIDENCE"
    validation_loss: float | str = "INSUFFICIENT_EVIDENCE"
    test_loss: float | str = "INSUFFICIENT_EVIDENCE"
    generation_metrics: dict[str, Any] = Field(default_factory=dict)
    capability_metrics: dict[str, Any] = Field(default_factory=dict)
    gate_status: OneBFunctionalStatus = OneBFunctionalStatus.NOT_STARTED
    failure_reasons: tuple[FunctionalFailure, ...] = ()
    parent_experiment: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PhaseLTrainingRecipeRegistry:
    """Immutable JSONL registry; completed evidence is never overwritten."""

    def __init__(self, root: Path) -> None:
        self.path = root / "experiments" / "phase-l-training-recipes.jsonl"

    def records(self) -> tuple[PhaseLTrainingRecipeRecord, ...]:
        if not self.path.exists():
            return ()
        return tuple(PhaseLTrainingRecipeRecord.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip())

    def record(self, record: PhaseLTrainingRecipeRecord) -> PhaseLTrainingRecipeRecord:
        if record.experiment_id in {item.experiment_id for item in self.records()}:
            raise ValueError(f"Phase L recipe already exists: {record.experiment_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(record.model_dump_json() + "\n")
        return record


class OneBFunctionalFoundationGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gate_id: str = "ONE_B_FUNCTIONAL_FOUNDATION_GATE"
    threshold_policy_version: str = "phase-l.1-observed-baseline-v1"
    status: OneBFunctionalStatus
    scaling_decision: ScaleDecision
    training_stability: str
    learning: str
    generalization: str
    generation: str
    instruction_following: str
    technical_programming: str
    reasoning: str
    reproducibility: str
    failure_reasons: tuple[FunctionalFailure, ...]
    rationale: str


def aggregate_generation_health(metrics: list[dict[str, Any]]) -> GenerationHealth:
    """Aggregate measured raw generations; absent measurements remain explicit."""
    if not metrics:
        return GenerationHealth(**{name: "INSUFFICIENT_EVIDENCE" for name in GenerationHealth.model_fields})
    mapping = {
        "eos_rate": "eos_rate", "mean_length": "average_length", "repeated_token_rate": "repeated_unigram_rate",
        "repeated_bigram_rate": "repeated_bigram_rate", "distinct_1": "distinct_1", "distinct_2": "distinct_2", "entropy": "entropy_bits",
    }
    values: dict[str, float | str] = {}
    for output, source in mapping.items():
        measured = [float(item[source]) for item in metrics if isinstance(item.get(source), (int, float))]
        values[output] = sum(measured) / len(measured) if measured else "INSUFFICIENT_EVIDENCE"
    return GenerationHealth(**values)


def one_b_functional_gate(records: tuple[PhaseLTrainingRecipeRecord, ...]) -> OneBFunctionalFoundationGate:
    """Fail closed: execution or a small-model result can never open 2B work."""
    candidates = [record for record in records if record.actual_parameter_count >= 900_000_000]
    if not candidates:
        return OneBFunctionalFoundationGate(status=OneBFunctionalStatus.NOT_STARTED, scaling_decision=ScaleDecision.DO_NOT_SCALE, training_stability="INSUFFICIENT_EVIDENCE", learning="INSUFFICIENT_EVIDENCE", generalization="INSUFFICIENT_EVIDENCE", generation="INSUFFICIENT_EVIDENCE", instruction_following="INSUFFICIENT_EVIDENCE", technical_programming="INSUFFICIENT_EVIDENCE", reasoning="INSUFFICIENT_EVIDENCE", reproducibility="INSUFFICIENT_EVIDENCE", failure_reasons=(FunctionalFailure.INSUFFICIENT_EVIDENCE,), rationale="No functional 1B training recipe record exists. Hardware-only execution evidence is not eligible.")
    latest = candidates[-1]
    required = ("generation", "capability_metrics")
    missing = [item for item in required if not getattr(latest, item if item != "generation" else "generation_metrics")]
    if missing:
        return OneBFunctionalFoundationGate(status=OneBFunctionalStatus.TRAINING, scaling_decision=ScaleDecision.DO_NOT_SCALE, training_stability="RECORDED", learning="INSUFFICIENT_EVIDENCE", generalization="INSUFFICIENT_EVIDENCE", generation="INSUFFICIENT_EVIDENCE", instruction_following="INSUFFICIENT_EVIDENCE", technical_programming="INSUFFICIENT_EVIDENCE", reasoning="INSUFFICIENT_EVIDENCE", reproducibility="RECORDED", failure_reasons=(FunctionalFailure.INSUFFICIENT_EVIDENCE,), rationale="A 1B run exists but lacks the required native-generation and model-only capability evidence.")
    return OneBFunctionalFoundationGate(status=OneBFunctionalStatus.FUNCTIONAL_CANDIDATE, scaling_decision=ScaleDecision.DO_NOT_SCALE, training_stability="RECORDED", learning="RECORDED", generalization="RECORDED", generation="RECORDED", instruction_following="RECORDED", technical_programming="RECORDED", reasoning="RECORDED", reproducibility="RECORDED", failure_reasons=(FunctionalFailure.INSUFFICIENT_EVIDENCE,), rationale="1B evidence is recorded, but threshold promotion is deliberately not automatic; review the versioned observed-baseline policy.")


class PhaseLCorpusGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    target_tokens: int
    valid_tokens: int
    rejected_tokens: int
    readiness: PhaseLReadiness
    integrity: str
    balance: dict[str, Any]
    release_manifest_sha256: str
    tokenized_manifest_sha256: str
    rationale: str
    corpus_tier: CorpusTier | None = None
    evidence_class: EvidenceClass | None = None
    formal_gate_status: FormalGateStatus = FormalGateStatus.SIZE_FAILURE
    development_status: DevelopmentStatus = DevelopmentStatus.BELOW_TARGET
    failure_reasons: tuple[str, ...] = ()


class CompositionStatus(str, Enum):
    UNDERREPRESENTED = "UNDERREPRESENTED"
    HEALTHY = "HEALTHY"
    OVERREPRESENTED = "OVERREPRESENTED"
    HARD_FAILURE = "HARD_FAILURE"


class CorpusCompositionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tokens: int
    percentage: float
    status: CompositionStatus


class CorpusDeficitPlan(BaseModel):
    """Token-realized acquisition plan, not an instruction to scrape the web."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    formal_target_tokens: int
    current_valid_tokens: int
    formal_deficit: int
    development_target_tokens: int
    development_deficit: int
    domain_deficits: dict[str, CorpusCompositionEntry]
    source_family_concentration: dict[str, CorpusCompositionEntry]
    source_domain_concentration: dict[str, CorpusCompositionEntry]
    underrepresented_domains: tuple[str, ...]
    overrepresented_domains: tuple[str, ...]
    overrepresented_source_families: tuple[str, ...]
    overrepresented_source_domains: tuple[str, ...]
    intended_cohorts: tuple[dict[str, Any], ...]


class PhaseLSustainedSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    steps: int
    initial_tokens_per_second: float | None
    steady_state_tokens_per_second: float | None
    minimum_tokens_per_second: float | None
    maximum_tokens_per_second: float | None
    median_tokens_per_second: float | None
    swap_trajectory: tuple[int | None, ...]
    swap_behavior: str
    memory_stable: bool
    training_status: str
    classification: SustainedRunClassification
    window_groups: dict[str, dict[str, float | int | None]]
    checkpoint_status: str
    evidence: str


def _formal_gate_status(readiness: PhaseLReadiness) -> FormalGateStatus:
    return {
        PhaseLReadiness.READY: FormalGateStatus.READY,
        PhaseLReadiness.BELOW_TARGET: FormalGateStatus.SIZE_FAILURE,
        PhaseLReadiness.INTEGRITY_FAILURE: FormalGateStatus.INTEGRITY_FAILURE,
        PhaseLReadiness.BALANCE_FAILURE: FormalGateStatus.BALANCE_FAILURE,
    }[readiness]


def phase_l_corpus_gate(release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer, *, target_tokens: int = 10_000_000, development_target_tokens: int = 1_000_000) -> PhaseLCorpusGate:
    release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenized = TokenizedRelease.model_validate_json((tokenized_dir / "manifest.json").read_text(encoding="utf-8"))
    balance = corpus_balance_report(release_dir, tokenized_dir, tokenizer).model_dump(mode="json")
    rejected_tokens = 0
    rejections = release_dir / "rejections.jsonl"
    if rejections.exists():
        # This is deliberately a conservative character proxy; rejected source
        # material never contributes to the valid-token threshold.
        rejected_tokens = sum(
            len(tokenizer.encode(ProcessedDocument.model_validate_json(line).content)) + 1
            for line in rejections.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    integrity = "PASS" if tokenized.tokenizer_sha256 == tokenizer.sha256 and balance["leakage_checks"].get("status") == "PASS" else "FAIL"
    if integrity != "PASS":
        readiness = PhaseLReadiness.INTEGRITY_FAILURE
    elif not balance["general_foundation_eligible"]:
        readiness = PhaseLReadiness.BALANCE_FAILURE
    elif tokenized.token_count < target_tokens:
        readiness = PhaseLReadiness.BELOW_TARGET
    else:
        readiness = PhaseLReadiness.READY
    formal_status = _formal_gate_status(readiness)
    if readiness is PhaseLReadiness.READY:
        corpus_tier, evidence_class = CorpusTier.FORMAL, EvidenceClass.FORMAL_PHASE_L
        development_status = DevelopmentStatus.READY
    elif integrity != "PASS":
        corpus_tier, evidence_class = None, None
        development_status = DevelopmentStatus.INTEGRITY_FAILURE
    elif not balance["general_foundation_eligible"]:
        corpus_tier, evidence_class = None, None
        development_status = DevelopmentStatus.BALANCE_FAILURE
    elif tokenized.token_count >= development_target_tokens:
        corpus_tier, evidence_class = CorpusTier.DEVELOPMENT, EvidenceClass.DEVELOPMENT_SCALE
        development_status = DevelopmentStatus.READY
    else:
        corpus_tier, evidence_class = CorpusTier.INFRASTRUCTURE, EvidenceClass.INFRASTRUCTURE_ONLY
        development_status = DevelopmentStatus.BELOW_TARGET
    reasons: list[str] = []
    if integrity != "PASS":
        reasons.append("INTEGRITY_FAILURE")
    if balance.get("leakage_checks", {}).get("status") != "PASS":
        reasons.append("SOURCE_ISOLATION_FAILURE")
    if balance.get("license_coverage", 0) < 1:
        reasons.append("LICENSE_FAILURE")
    if not balance["general_foundation_eligible"] and integrity == "PASS":
        reasons.append("BALANCE_FAILURE")
    if tokenized.token_count < target_tokens:
        reasons.append("SIZE_FAILURE")
    rationale = "Eligible immutable 10M+ release." if readiness is PhaseLReadiness.READY else "Formal Phase L scaling is not authorized; this release is classified independently for infrastructure or development work only."
    return PhaseLCorpusGate(release_id=release.release_id, target_tokens=target_tokens, valid_tokens=tokenized.token_count, rejected_tokens=rejected_tokens, readiness=readiness, integrity=integrity, balance=balance, release_manifest_sha256=release.manifest_sha256 or sha256_file(release_dir / "manifest.json"), tokenized_manifest_sha256=tokenized.manifest_sha256 or sha256_file(tokenized_dir / "manifest.json"), rationale=rationale, corpus_tier=corpus_tier, evidence_class=evidence_class, formal_gate_status=formal_status, development_status=development_status, failure_reasons=tuple(reasons))


def corpus_deficit_plan(
    release_dir: Path,
    tokenized_dir: Path,
    tokenizer: UltronTokenizer,
    *,
    formal_target_tokens: int = 10_000_000,
    development_target_tokens: int = 1_000_000,
    minimum_important_domain_percent: float = 0.01,
    maximum_concentration_percent: float = 0.45,
    hard_failure_percent: float = 0.60,
) -> CorpusDeficitPlan:
    """Report deficits from realized tokenizer tokens and explicit provenance."""
    release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenized = TokenizedRelease.model_validate_json((tokenized_dir / "manifest.json").read_text(encoding="utf-8"))
    if tokenized.tokenizer_sha256 != tokenizer.sha256:
        raise ValueError("Tokenized corpus tokenizer hash does not match the requested tokenizer.")
    documents = [ProcessedDocument.model_validate_json(line) for line in (release_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    domain_tokens: Counter[str] = Counter()
    family_tokens: Counter[str] = Counter()
    source_domain_tokens: Counter[str] = Counter()
    for document in documents:
        tokens = len(tokenizer.encode(document.content)) + 1
        domain_tokens[document.classification.value] += tokens
        family_tokens[document.provenance.source_group or document.provenance.source_identifier] += tokens
        source_domain_tokens[urlparse(document.provenance.source_url or "").netloc or "LOCAL"] += tokens
    realized_total = max(1, sum(domain_tokens.values()))

    def status_for(percent: float, *, important: bool) -> CompositionStatus:
        if percent > hard_failure_percent:
            return CompositionStatus.HARD_FAILURE
        if percent > maximum_concentration_percent:
            return CompositionStatus.OVERREPRESENTED
        if important and percent < minimum_important_domain_percent:
            return CompositionStatus.UNDERREPRESENTED
        return CompositionStatus.HEALTHY

    important_domains = ("CODE", "MATHEMATICS", "SCIENCE", "ENGINEERING", "TECHNICAL", "ACADEMIC", "INSTRUCTION")
    domains: dict[str, CorpusCompositionEntry] = {}
    for domain in sorted(set(important_domains) | set(domain_tokens)):
        count = domain_tokens[domain]
        percent = count / realized_total
        domains[domain] = CorpusCompositionEntry(tokens=count, percentage=round(percent, 6), status=status_for(percent, important=domain in important_domains))
    families = {
        name: CorpusCompositionEntry(tokens=count, percentage=round(count / realized_total, 6), status=status_for(count / realized_total, important=False))
        for name, count in sorted(family_tokens.items())
    }
    source_domains = {
        name: CorpusCompositionEntry(tokens=count, percentage=round(count / realized_total, 6), status=status_for(count / realized_total, important=False))
        for name, count in sorted(source_domain_tokens.items())
    }
    underrepresented = tuple(name for name, item in domains.items() if item.status is CompositionStatus.UNDERREPRESENTED)
    overrepresented_domains = tuple(name for name, item in domains.items() if item.status in {CompositionStatus.OVERREPRESENTED, CompositionStatus.HARD_FAILURE})
    overrepresented_families = tuple(name for name, item in families.items() if item.status in {CompositionStatus.OVERREPRESENTED, CompositionStatus.HARD_FAILURE})
    overrepresented_source_domains = tuple(name for name, item in source_domains.items() if item.status in {CompositionStatus.OVERREPRESENTED, CompositionStatus.HARD_FAILURE})
    cohorts: list[dict[str, Any]] = []
    if underrepresented:
        cohorts.append({"cohort": " + ".join(name.lower() for name in underrepresented), "target_tokens": "750k–1M", "reason": "realized tokenizer-token underrepresentation", "source_family_constraints": "at least 2 independent families"})
    if overrepresented_domains or overrepresented_families or overrepresented_source_domains:
        cohorts.append({"cohort": "non-dominant domains", "target_tokens": "750k–1M", "reason": "correct concentration before adding more material from dominant sources", "source_family_constraints": "exclude dominant family until below maximum concentration"})
    if not cohorts and tokenized.token_count < formal_target_tokens:
        cohorts.append({"cohort": "broad multi-domain expansion", "target_tokens": "750k–1M", "reason": "advance the formal deficit while preserving healthy realized composition", "source_family_constraints": "at least 3 independent families; retain all concentration ceilings"})
    return CorpusDeficitPlan(release_id=release.release_id, formal_target_tokens=formal_target_tokens, current_valid_tokens=tokenized.token_count, formal_deficit=max(0, formal_target_tokens - tokenized.token_count), development_target_tokens=development_target_tokens, development_deficit=max(0, development_target_tokens - tokenized.token_count), domain_deficits=domains, source_family_concentration=families, source_domain_concentration=source_domains, underrepresented_domains=underrepresented, overrepresented_domains=overrepresented_domains, overrepresented_source_families=overrepresented_families, overrepresented_source_domains=overrepresented_source_domains, intended_cohorts=tuple(cohorts))


def phase_l_experiment_report(run: TrainingRun) -> dict[str, Any]:
    """Produce a claim-bounded report from a typed registry record."""
    if run.evidence_class is EvidenceClass.FORMAL_PHASE_L:
        if run.corpus_tier is not CorpusTier.FORMAL or run.formal_gate_status is not FormalGateStatus.READY:
            raise ValueError("A formal Phase L report requires a FORMAL corpus with a READY gate.")
        claim = "FORMAL_PHASE_L"
    elif run.evidence_class is EvidenceClass.DEVELOPMENT_SCALE:
        claim = "DEVELOPMENT_SCALE: diagnostic evidence only; not a formal 10M scaling conclusion."
    else:
        claim = "INFRASTRUCTURE_ONLY: execution/pipeline evidence only."
    return {"run_id": run.run_id, "corpus_id": run.corpus_id, "corpus_tier": run.corpus_tier.value, "experiment_tier": run.experiment_tier.value, "evidence_class": run.evidence_class.value, "formal_gate_status": run.formal_gate_status.value, "development_status": run.development_status.value, "token_count": run.token_count, "model_parameters": run.model_parameters, "training_tokens": run.training_tokens, "validation_tokens": run.validation_tokens, "test_tokens": run.test_tokens, "claim_boundary": claim}


def _window_groups(windows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    groups = {"steps_1_10": (1, 10), "steps_11_50": (11, 50), "steps_51_100": (51, 100)}
    result: dict[str, dict[str, float | int | None]] = {}
    for label, (start, end) in groups.items():
        selected = [item for item in windows if int(item.get("start_step") or 0) >= start and int(item.get("end_step") or 0) <= end]
        rates = [float(item["tokens_per_second"]) for item in selected if item.get("tokens_per_second") is not None]
        losses = [float(item["mean_loss"]) for item in selected if item.get("mean_loss") is not None]
        result[label] = {"start_step": start, "end_step": end, "mean_tokens_per_second": sum(rates) / len(rates) if rates else None, "minimum_tokens_per_second": min(rates) if rates else None, "maximum_tokens_per_second": max(rates) if rates else None, "mean_loss": sum(losses) / len(losses) if losses else None}
    return result


def summarize_sustained_run(result: ScalingResult) -> PhaseLSustainedSummary:
    windows = list(result.training.get("windows") or [])
    rates = [float(item["tokens_per_second"]) for item in windows if item.get("tokens_per_second") is not None]
    swaps = tuple(item.get("swap_after_bytes") for item in windows)
    deltas = [later - earlier for earlier, later in pairwise(swaps) if isinstance(earlier, int) and isinstance(later, int)]
    behavior = "NOT_MEASURED" if not windows else ("GROWING" if len(deltas) >= 2 and all(delta > 0 for delta in deltas) else "STABILIZING_OR_OSCILLATING")
    ordered = sorted(rates)
    losses = [result.training.get(name) for name in ("loss", "loss_min", "loss_max")]
    numeric_valid = all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses)
    steps = int(result.training["steps"])
    if result.failure is not None:
        classification = SustainedRunClassification.FAILED_EXECUTION
    elif not numeric_valid:
        classification = SustainedRunClassification.NUMERICALLY_UNSTABLE
    elif steps < 100:
        classification = SustainedRunClassification.NOT_RUN
    elif behavior == "GROWING":
        classification = SustainedRunClassification.RESOURCE_UNSTABLE
    elif (result.memory.get("swap_peak_bytes") or 0) > (result.memory.get("swap_used_before_bytes") or 0):
        classification = SustainedRunClassification.SUSTAINED_EXECUTION_WITH_SWAP
    else:
        classification = SustainedRunClassification.SUSTAINED_EXECUTION
    return PhaseLSustainedSummary(experiment_id=result.experiment_id, steps=steps, initial_tokens_per_second=rates[0] if rates else None, steady_state_tokens_per_second=rates[-1] if rates else None, minimum_tokens_per_second=min(rates) if rates else None, maximum_tokens_per_second=max(rates) if rates else None, median_tokens_per_second=ordered[len(ordered)//2] if ordered else None, swap_trajectory=swaps, swap_behavior=behavior, memory_stable=behavior == "STABILIZING_OR_OSCILLATING", training_status="SUSTAINED_ESTABLISHED" if steps >= 100 and result.failure is None else "NOT_YET_ESTABLISHED", classification=classification, window_groups=_window_groups(windows), checkpoint_status="NOT_ATTEMPTED: scaling harness has no checkpoint operation; no checkpoint conclusion may be drawn.", evidence="Measured real optimizer steps on random token IDs; this measures execution stability only, never model learning or capability.")


def registry_corpus_candidate(root: Path, *, release_id: str, tokenizer: UltronTokenizer, context_length: int = 512, privacy_action: str = "REJECT") -> tuple[DatasetRelease, TokenizedRelease, PhaseLCorpusGate]:
    """Publish an immutable candidate using only previously acquired sources.

    The source registry supplies cache hashes, licences, and evidence.  Each
    cache hash is used once, so this routine cannot inflate scale by replaying
    duplicate registry entries.  The gate decides whether the candidate is
    eligible; publication never implies acceptance.
    """
    from app.brain.acquisition import AcquisitionStore

    records = AcquisitionStore(root).sources()
    seen: set[str] = set()
    documents: list[CorpusDocument] = []
    for entry in records:
        if entry.content_hash in seen:
            continue
        cache_path = Path(entry.provenance["cache_path"])
        # Historical acquisition jobs recorded cache paths relative to the
        # backend working directory, while newer jobs run from the repository
        # root.  Cache identity is the content hash, so resolve either legacy
        # spelling through this release's immutable cache before excluding it.
        if not cache_path.exists():
            cache_path = root / "acquisition" / "cache" / cache_path.name
        if not cache_path.exists():
            continue
        seen.add(entry.content_hash)
        documents.append(CorpusDocument(content=cache_path.read_text(encoding="utf-8"), provenance=Provenance(source_type=SourceType.PUBLIC_DATASET, source_identifier=entry.source_id, source_url=entry.canonical_url, license=entry.license, license_evidence=entry.license_evidence, source_group=entry.source_concentration_group, acquisition_version=entry.acquisition_version, parser_version=entry.parser_version, declared_category=entry.category)))
    pipeline = CorpusPipeline(root, CorpusConfig(split_strategy="source_group_balanced", validation_percent=10, test_percent=10, privacy_action=privacy_action, source_dominance_threshold=0.50))
    release = pipeline.release(release_id, documents)
    release_dir = root / "corpora" / "releases" / release.release_id
    tokenized = TokenPacker(root).pack(release_dir, tokenizer, context_length=context_length)
    tokenized_dir = root / "corpora" / "tokenized" / release.release_id / tokenizer.sha256[:16] / f"context-{context_length}"
    return release, tokenized, phase_l_corpus_gate(release_dir, tokenized_dir, tokenizer)


def checkpoint_portability(checkpoint: Path) -> dict[str, Any]:
    """Verify a complete checkpoint's architecture and weight artifact hashes.

    Loading is performed by the ordinary checkpoint/generation backends; this
    portable invariant remains useful before a second hardware backend exists.
    """
    manifest = CheckpointStore.inspect(checkpoint)
    if not hasattr(manifest, "architecture_config_sha256"):
        raise TypeError("Portability requires a complete resumable checkpoint.")
    weights = checkpoint / "model.safetensors"
    return {"status": "PORTABLE_LOCAL_FORMAT_VERIFIED", "checkpoint_id": manifest.checkpoint_id, "architecture_config_sha256": manifest.architecture_config_sha256, "weights_sha256": sha256_file(weights), "backend": "MLX_LOCAL", "cross_backend_generation": "NOT_MEASURED: no second backend is implemented."}
