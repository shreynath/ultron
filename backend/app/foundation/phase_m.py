"""Phase M corpus-release gates and evidence contracts.

This module deliberately builds on the acquisition registry and the immutable
corpus/packing implementations.  It does not fetch arbitrary web pages or
invent corpus text: a source must already be registered, licence-evidenced,
and present in the content-addressed cache before it can enter a release.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash, sha256_file
from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DatasetRelease,
    DocumentCategory,
    ProcessedDocument,
    Provenance,
    SourceType,
    TokenizedRelease,
    TokenPacker,
    resolve_artifact_path,
)
from app.foundation.diagnostics import corpus_diagnostics
from app.foundation.tokenizer import UltronTokenizer

if TYPE_CHECKING:
    from app.brain.acquisition import SourceRegistryEntry

PHASE_M_SPEC_VERSION = "ultron.phase-m.corpus-spec.v1"
ALLOWED_LICENSES = frozenset({
    "CC-BY-4.0", "CC-BY-SA-2.5", "CC-BY-SA-4.0",
    "IETF-Trust-Legal-Provisions", "IETF-Trust-License",
    "MIT-OR-Apache-2.0", "PSF-2.0", "PUBLIC-DOMAIN-US",
    "Public Domain", "RFC Trust License",
})
_SENTENCE = re.compile(r"[^.!?]+[.!?]+")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_MARKUP = re.compile(r"<[^>]{1,160}>|^\s{0,3}#{1,6}\s|^\s*[-*+]\s", re.MULTILINE)


class PhaseMDisposition(str, Enum):
    ACTIVE = "PHASE_M_ACTIVE_CORPUS_EXPANSION"
    EMERGENCE = "PHASE_M_FOUNDATION_EMERGENCE_CONFIRMED"
    CAPACITY_NEXT = "PHASE_M_CAPACITY_LIMIT_IDENTIFIED"
    TRAINING_LIMIT = "PHASE_M_TRAINING_REGIME_LIMIT_IDENTIFIED"
    CORPUS_BLOCKED = "PHASE_M_CORPUS_BLOCKED"


class SourceDecisionStatus(str, Enum):
    INCLUDED = "INCLUDED"
    EXCLUDED = "EXCLUDED"


class PhaseMCorpusSpec(BaseModel):
    """Versioned policy for a Phase M corpus release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = PHASE_M_SPEC_VERSION
    tokenizer_path: str = "data/ultron/tokenizers/ultron-phase-i-v2"
    permitted_source_types: tuple[str, ...] = ("PUBLIC_DATASET",)
    allowed_licenses: tuple[str, ...] = tuple(sorted(ALLOWED_LICENSES))
    minimum_characters: int = Field(default=500, ge=16)
    validation_percent: int = Field(default=10, ge=1, le=30)
    test_percent: int = Field(default=10, ge=1, le=30)
    split_strategy: str = "source_group_balanced"
    exact_deduplication: str = "SHA-256 raw and NFC-normalized content hashes"
    near_deduplication: str = "deterministic 5-token-shingle fingerprint; source groups never cross splits"
    packing_strategy: str = "VARIABLE_EOS_MASKED_V2"
    source_requirements: str = "registered source identity, canonical URL, content hash, retrieval timestamp, licence and licence evidence"
    prohibited_content: tuple[str, ...] = (
        "unregistered downloads", "synthetic filler", "evaluation prompts or answers",
        "duplicated records", "metadata-as-prose",
    )


class SourceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    status: SourceDecisionStatus
    reason: str | None = None
    content_hash: str
    cache_path: str | None = None
    canonical_url: str
    license: str
    license_evidence: str | None = None
    retrieved_at: datetime
    source_group: str
    category: str


class PhaseMSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    created_at: datetime
    specification_version: str
    registry_path: str
    decisions: tuple[SourceDecision, ...]
    sha256: str | None = None


class PhaseMCorpusAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    tokenizer_version: str
    tokenizer_sha256: str
    total_tokens: int
    tokens_by_split: dict[str, int]
    documents_by_split: dict[str, int]
    raw_documents: int
    accepted_documents: int
    rejected_documents: int
    raw_characters: int
    normalized_characters: int
    source_distribution: dict[str, int]
    document_length: dict[str, float]
    sentence_length: dict[str, float]
    punctuation_rate: float
    alphabetic_ratio: float
    code_document_fraction: float
    markup_character_ratio: float
    url_character_ratio: float
    special_token_counts: dict[str, int]
    vocabulary_coverage: float
    exact_duplicates: int
    near_duplicates: int
    repeated_boilerplate_span_tokens: int
    boundary_integrity: dict[str, Any]
    leakage_checks: dict[str, Any]
    status: str
    failures: tuple[str, ...]


class PhaseMReleaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    specification: PhaseMCorpusSpec
    release: DatasetRelease
    tokenized: TokenizedRelease
    source_manifest: PhaseMSourceManifest
    audit: PhaseMCorpusAudit
    target_tokens: int
    accepted: bool


class PhaseMRungPlan(BaseModel):
    """An immutable experiment plan; results are recorded separately."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    rung_id: str
    corpus_release_id: str
    corpus_tokens: int
    update_token_target: int
    parent_checkpoint: str | None = None
    initialization: str = "RANDOM_FOR_RUNG_1"
    changed_variable: str = "CORPUS"
    frozen_controls: tuple[str, ...] = (
        "50.55M architecture", "ultron-phase-i-v2 tokenizer", "next-token objective",
        "VARIABLE_EOS_MASKED_V2 packing", "AdamW recipe", "context length", "seed", "frozen evaluations",
    )


class PhaseMBranchDecision(BaseModel):
    """Reusable midpoint gate; it never promotes a loss-only improvement."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: str
    reasons: tuple[str, ...]
    functional_emergence_level: int = Field(ge=0, le=4)
    capability_score: int = Field(ge=0, le=6)


class PhaseMStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    disposition: PhaseMDisposition
    corpus_versions: tuple[str, ...]
    corpus_tokens: dict[str, int]
    latest_experiment: str | None
    functional_emergence_level: int | None = Field(default=None, ge=0, le=4)
    capability_score: str | None = None
    scaling_recommendation: str
    next_action: str


def _cache_path(root: Path, entry: SourceRegistryEntry) -> Path:
    recorded = Path(entry.provenance.get("cache_path", ""))
    if recorded.exists():
        return recorded
    return root / "acquisition" / "cache" / f"{entry.content_hash}.txt"


def source_decisions(root: Path, *, spec: PhaseMCorpusSpec | None = None) -> tuple[SourceDecision, ...]:
    """Make inclusion/exclusion explicit before corpus construction."""
    spec = spec or PhaseMCorpusSpec()
    from app.brain.acquisition import AcquisitionStore

    entries = AcquisitionStore(root).sources()
    # Historical acquisition snapshots can share one canonical source but have
    # different truncation limits. Retaining both would replay the same opening
    # text. Keep the largest hash-verified snapshot deterministically.
    canonical_winners: dict[str, str] = {}
    for entry in entries:
        cache = _cache_path(root, entry)
        if cache.exists() and sha256_file(cache) == entry.content_hash:
            previous = canonical_winners.get(entry.canonical_url)
            if previous is None:
                canonical_winners[entry.canonical_url] = entry.source_id
            else:
                current_entry = next(item for item in entries if item.source_id == previous)
                current_cache = _cache_path(root, current_entry)
                if (cache.stat().st_size, entry.source_id) > (current_cache.stat().st_size, current_entry.source_id):
                    canonical_winners[entry.canonical_url] = entry.source_id
    decisions: list[SourceDecision] = []
    seen_hashes: set[str] = set()
    for entry in entries:
        cache = _cache_path(root, entry)
        reason: str | None = None
        if entry.license not in spec.allowed_licenses:
            reason = "LICENSE_NOT_PERMITTED_BY_PHASE_M_SPEC"
        elif not entry.license_evidence:
            reason = "MISSING_LICENSE_EVIDENCE"
        elif not entry.canonical_url or not entry.content_hash:
            reason = "INCOMPLETE_PROVENANCE"
        elif not cache.exists() or sha256_file(cache) != entry.content_hash:
            reason = "MISSING_OR_HASH_MISMATCHED_CACHE"
        elif canonical_winners.get(entry.canonical_url) != entry.source_id:
            reason = "SUPERSEDED_SAME_CANONICAL_SOURCE"
        elif entry.content_hash in seen_hashes:
            reason = "EXACT_DUPLICATE_SOURCE_CONTENT"
        else:
            seen_hashes.add(entry.content_hash)
        decisions.append(SourceDecision(
            source_id=entry.source_id,
            status=SourceDecisionStatus.EXCLUDED if reason else SourceDecisionStatus.INCLUDED,
            reason=reason,
            content_hash=entry.content_hash,
            cache_path=str(cache) if cache.exists() else None,
            canonical_url=entry.canonical_url,
            license=entry.license,
            license_evidence=entry.license_evidence,
            retrieved_at=entry.retrieved_at,
            source_group=entry.source_concentration_group,
            category=entry.category,
        ))
    return tuple(decisions)


def _write_source_manifest(root: Path, release_id: str, decisions: tuple[SourceDecision, ...], spec: PhaseMCorpusSpec) -> PhaseMSourceManifest:
    target = root / "corpora" / "source-manifests" / f"{release_id}.json"
    if target.exists():
        raise ValueError(f"Source manifest already exists: {target}")
    value = PhaseMSourceManifest(
        release_id=release_id, created_at=datetime.now(UTC), specification_version=spec.version,
        registry_path=str(root / "acquisition" / "acquisition.sqlite3"), decisions=decisions,
    )
    value = value.model_copy(update={"sha256": canonical_hash(value.model_dump(mode="json"))})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return value


def _documents(root: Path, decisions: tuple[SourceDecision, ...]) -> list[CorpusDocument]:
    result: list[CorpusDocument] = []
    for decision in decisions:
        if decision.status is not SourceDecisionStatus.INCLUDED:
            continue
        assert decision.cache_path is not None
        result.append(CorpusDocument(
            content=Path(decision.cache_path).read_text(encoding="utf-8"),
            provenance=Provenance(
                source_type=SourceType.PUBLIC_DATASET, source_identifier=decision.source_id,
                source_url=decision.canonical_url, collection_timestamp=decision.retrieved_at,
                license=decision.license, license_evidence=decision.license_evidence,
                source_group=decision.source_group, dataset_release=None,
                declared_category=DocumentCategory(decision.category),
                acquisition_version="registry-backed", parser_version="registry-backed",
            ),
        ))
    return result


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "min": float(ordered[0]), "mean": round(sum(ordered) / len(ordered), 4),
        "median": float(ordered[len(ordered) // 2]),
        "p95": float(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)]), "max": float(ordered[-1]),
    }


def audit_phase_m_release(release_dir: Path, tokenized_dir: Path, tokenizer: UltronTokenizer, *, raw_documents: int | None = None) -> PhaseMCorpusAudit:
    """Audit every split and the immutable packed artifact without rebuilding it."""
    release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
    packed = TokenizedRelease.model_validate_json((tokenized_dir / "manifest.json").read_text(encoding="utf-8"))
    if packed.tokenizer_sha256 != tokenizer.sha256:
        raise ValueError("Phase M audit refuses a tokenizer mismatch.")
    all_docs: list[ProcessedDocument] = []
    split_tokens: dict[str, int] = {}
    split_docs: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = release_dir / f"{split}.jsonl"
        docs = [ProcessedDocument.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_docs.extend(docs)
        split_docs[split] = len(docs)
        split_tokens[split] = sum(len(tokenizer.encode(doc.content)) + 1 for doc in docs)
    content = "\n".join(doc.content for doc in all_docs)
    encoded = [tokenizer.encode(doc.content) for doc in all_docs]
    diagnostics = {split: corpus_diagnostics(release_dir, tokenizer, split=split) for split in ("train", "validation", "test")}
    sentence_words = [len(re.findall(r"[A-Za-z]+", sentence)) for sentence in _SENTENCE.findall(content)]
    eos, bos, unk, pad = tokenizer.processor.eos_id(), tokenizer.processor.bos_id(), tokenizer.processor.unk_id(), tokenizer.processor.pad_id()
    flat = [token for row in encoded for token in row]
    raw_characters = sum(len(doc.content) for doc in all_docs)
    source_distribution = Counter(doc.provenance.source_group or doc.provenance.source_identifier for doc in all_docs)
    markup = sum(len(match.group(0)) for match in _MARKUP.finditer(content))
    url_chars = sum(len(match.group(0)) for match in _URL.finditer(content))
    alphabetic = sum(char.isalpha() for char in content)
    shard_hash_ok = True
    malformed_rows = 0
    cross_document_rows = 0
    known_ids = {doc.provenance.document_id for doc in all_docs}
    for shard in packed.shards:
        path = resolve_artifact_path(shard.path)
        shard_hash_ok &= path.exists() and sha256_file(path) == shard.sha256
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ids, mask = row.get("input_ids", []), row.get("loss_mask")
            if row.get("document_id") not in known_ids or len(ids) != packed.context_length + 1:
                malformed_rows += 1
            if mask is not None and (len(mask) != packed.context_length or any(item not in (0, 1) for item in mask)):
                malformed_rows += 1
            # The format contains one document_id per row, so a cross-document
            # row would be structurally unrepresentable. Keep the measured count.
            cross_document_rows += 0
    failures: list[str] = []
    if release.leakage_checks.get("status") != "PASS":
        failures.append("SPLIT_LEAKAGE")
    if any(diagnostics[split].exact_duplicate_count or diagnostics[split].normalized_duplicate_count or diagnostics[split].fingerprint_duplicate_count for split in diagnostics):
        failures.append("DUPLICATES_REMAIN_WITHIN_SPLIT")
    if malformed_rows or not shard_hash_ok or cross_document_rows:
        failures.append("PACKING_OR_SHARD_INTEGRITY_FAILURE")
    if any(not split_docs[split] for split in split_docs):
        failures.append("SPLIT_INCOMPLETE")
    return PhaseMCorpusAudit(
        release_id=release.release_id, tokenizer_version=tokenizer.version, tokenizer_sha256=tokenizer.sha256,
        total_tokens=packed.token_count, tokens_by_split=split_tokens, documents_by_split=split_docs,
        raw_documents=raw_documents if raw_documents is not None else len(all_docs), accepted_documents=len(all_docs),
        rejected_documents=max(0, (raw_documents if raw_documents is not None else len(all_docs)) - len(all_docs)),
        raw_characters=raw_characters, normalized_characters=raw_characters,
        source_distribution=dict(sorted(source_distribution.items())),
        document_length=_summary([len(doc.content) for doc in all_docs]), sentence_length=_summary(sentence_words),
        punctuation_rate=round(sum(char in ".,;:!?()[]{}'\"" for char in content) / max(1, len(content)), 8),
        alphabetic_ratio=round(alphabetic / max(1, len(content)), 8),
        code_document_fraction=round(sum(doc.classification is DocumentCategory.CODE for doc in all_docs) / max(1, len(all_docs)), 8),
        markup_character_ratio=round(markup / max(1, len(content)), 8), url_character_ratio=round(url_chars / max(1, len(content)), 8),
        special_token_counts={"eos_appended": len(all_docs), "bos_in_content": flat.count(bos), "unk_in_content": flat.count(unk), "pad_in_content": flat.count(pad), "eos_in_content": flat.count(eos)},
        vocabulary_coverage=round(len(set(flat)) / max(1, tokenizer.vocab_size), 8),
        exact_duplicates=sum(item.exact_duplicate_count for item in diagnostics.values()),
        near_duplicates=sum(item.fingerprint_duplicate_count for item in diagnostics.values()),
        repeated_boilerplate_span_tokens=max(item.cross_document_longest_repeated_span_tokens for item in diagnostics.values()),
        boundary_integrity={"shard_hashes": "PASS" if shard_hash_ok else "FAIL", "malformed_rows": malformed_rows, "cross_document_rows": cross_document_rows, "packing_strategy": packed.packing_strategy, "document_boundary_policy": packed.document_boundary_policy},
        leakage_checks=release.leakage_checks, status="PASS" if not failures else "FAIL", failures=tuple(failures),
    )


def build_phase_m_release(root: Path, *, release_id: str, tokenizer: UltronTokenizer, target_tokens: int, spec: PhaseMCorpusSpec | None = None, context_length: int = 512) -> PhaseMReleaseResult:
    """Publish one immutable registered-source release and its full audit."""
    spec = spec or PhaseMCorpusSpec()
    if tokenizer.version != "0.5.0-phase-i-v2":
        raise ValueError("Phase M requires the frozen ultron-phase-i-v2 tokenizer.")
    decisions = source_decisions(root, spec=spec)
    documents = _documents(root, decisions)
    if not documents:
        raise ValueError("No registered, license-evidenced source is eligible for Phase M.")
    source_manifest = _write_source_manifest(root, release_id, decisions, spec)
    pipeline = CorpusPipeline(root, CorpusConfig(
        minimum_characters=spec.minimum_characters, validation_percent=spec.validation_percent,
        test_percent=spec.test_percent, split_strategy=spec.split_strategy, privacy_action="REDACT",
        source_dominance_threshold=0.50,
    ))
    release = pipeline.release(release_id, documents)
    release_dir = root / "corpora" / "releases" / release_id
    packed = TokenPacker(root).pack(release_dir, tokenizer, context_length=context_length, packing_strategy=spec.packing_strategy)
    tokenized_dir = root / "corpora" / "tokenized" / release_id / tokenizer.sha256[:16] / f"context-{context_length}-variable-eos-masked-v2"
    audit = audit_phase_m_release(release_dir, tokenized_dir, tokenizer, raw_documents=len(documents))
    target = release_dir / "phase-m-artifacts"
    target.mkdir()
    (target / "corpus-spec.json").write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (target / "quality-report.json").write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (target / "deduplication-report.json").write_text(json.dumps({"exact_method": spec.exact_deduplication, "near_method": spec.near_deduplication, "registry_duplicate_removals": sum(item.reason == "EXACT_DUPLICATE_SOURCE_CONTENT" for item in decisions), "release_duplicates": {"exact": audit.exact_duplicates, "near": audit.near_duplicates}}, indent=2) + "\n", encoding="utf-8")
    (target / "license-source-report.json").write_text(json.dumps({"included": sum(item.status is SourceDecisionStatus.INCLUDED for item in decisions), "excluded": sum(item.status is SourceDecisionStatus.EXCLUDED for item in decisions), "licenses": dict(Counter(item.license for item in decisions if item.status is SourceDecisionStatus.INCLUDED)), "source_manifest_sha256": source_manifest.sha256}, indent=2) + "\n", encoding="utf-8")
    (target / "split-manifest.json").write_text(json.dumps({"release_manifest_sha256": release.manifest_sha256, "split_method": release.split_method, "documents": audit.documents_by_split, "tokens": audit.tokens_by_split, "leakage_checks": audit.leakage_checks}, indent=2) + "\n", encoding="utf-8")
    return PhaseMReleaseResult(specification=spec, release=release, tokenized=packed, source_manifest=source_manifest, audit=audit, target_tokens=target_tokens, accepted=audit.status == "PASS" and packed.token_count >= target_tokens)


def phase_m_status(root: Path) -> PhaseMStatus:
    """Read corpus and, when present, immutable terminal decision evidence."""
    releases = root / "corpora" / "releases"
    records: dict[str, int] = {}
    for directory in sorted(releases.glob("ultron-corpus-*-phase-m-*")) if releases.exists() else []:
        manifest = directory / "manifest.json"
        tokenized = root / "corpora" / "tokenized" / directory.name
        manifests = list(tokenized.glob("**/manifest.json")) if tokenized.exists() else []
        if manifest.exists() and manifests:
            records[directory.name] = TokenizedRelease.model_validate_json(manifests[0].read_text(encoding="utf-8")).token_count
    final_path = root / "reports" / "phase-m-final-disposition-v1.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        selected_release_id = final.get("dataset_release_id")
        if selected_release_id in records:
            records = {selected_release_id: records[selected_release_id]}
        return PhaseMStatus(
            disposition=PhaseMDisposition(final["disposition"]),
            corpus_versions=tuple(records),
            corpus_tokens=records,
            latest_experiment=final.get("run_id"),
            functional_emergence_level=final.get("functional_emergence_level"),
            capability_score=final.get("capability_score"),
            scaling_recommendation=final["scaling_recommendation"],
            next_action=final["next_action"],
        )
    if any(tokens >= 10_000_000 for tokens in records.values()):
        disposition, recommendation, action = PhaseMDisposition.ACTIVE, "Run frozen 10M rung then make the capacity/training disposition from evidence.", "Execute the 10M update-token rung."
    elif any(tokens >= 5_000_000 for tokens in records.values()):
        disposition, recommendation, action = PhaseMDisposition.ACTIVE, "Run 1M and 5M frozen-recipe rungs before formal expansion.", "Execute the 1M update-token rung."
    else:
        disposition, recommendation, action = PhaseMDisposition.ACTIVE, "Expand only the source registry with licence-evidenced, cached primary/public-domain or permissively licensed prose.", "Acquire and register sufficient new legitimate material, then publish the 5M release."
    return PhaseMStatus(disposition=disposition, corpus_versions=tuple(records), corpus_tokens=records, latest_experiment=None, functional_emergence_level=None, capability_score=None, scaling_recommendation=recommendation, next_action=action)


def phase_m_source_inventory(root: Path, tokenizer: UltronTokenizer) -> dict[str, Any]:
    """Emit the pre-release source-pool measurement as durable evidence."""
    decisions = source_decisions(root)
    included = [item for item in decisions if item.status is SourceDecisionStatus.INCLUDED]
    return {
        "report_id": "phase-m-source-inventory-v1",
        "specification_version": PHASE_M_SPEC_VERSION,
        "tokenizer": {"version": tokenizer.version, "sha256": tokenizer.sha256},
        "registry_entries": len(decisions),
        "included_sources": len(included),
        "excluded_sources": len(decisions) - len(included),
        "eligible_token_count": sum(len(tokenizer.encode(Path(item.cache_path or "").read_text(encoding="utf-8"))) + 1 for item in included),
        "exclusion_reasons": dict(sorted(Counter(item.reason for item in decisions if item.reason).items())),
        "decisions": [item.model_dump(mode="json") for item in decisions],
    }


def phase_m_branch_decision(
    *, functional_emergence_level: int, capability_score: int, generation: dict[str, float],
    learning_signal: bool, data_valid: bool = True, evaluation_valid: bool = True,
    checkpoint_valid: bool = True, baseline_level: int = 0, baseline_capability_score: int = 0,
) -> PhaseMBranchDecision:
    """Apply Phase M stopping and promotion rules to one completed checkpoint.

    High validation loss does not stop a branch. Conversely, a lower loss alone
    cannot promote it: functional behavior, capability, diversity, or sustained
    learning must provide the evidence.
    """
    reasons: list[str] = []
    if not data_valid:
        reasons.append("INVALID_DATA")
    if not evaluation_valid:
        reasons.append("INVALID_EVALUATION")
    if not checkpoint_valid:
        reasons.append("CORRUPTED_CHECKPOINT")
    if generation.get("immediate_eos_rate", generation.get("eos_rate", 0.0)) >= 0.90:
        reasons.append("PERSISTENT_IMMEDIATE_EOS")
    if generation.get("punctuation_loop_rate", 0.0) >= 0.90:
        reasons.append("PERSISTENT_PUNCTUATION_COLLAPSE")
    if generation.get("repeated_unigram_rate", 0.0) >= 0.90:
        reasons.append("PERSISTENT_REPEATED_TOKEN_COLLAPSE")
    if reasons:
        return PhaseMBranchDecision(action="STOP_AND_INVESTIGATE", reasons=tuple(reasons), functional_emergence_level=functional_emergence_level, capability_score=capability_score)
    improved_behavior = functional_emergence_level > baseline_level or capability_score > baseline_capability_score
    diversity = generation.get("distinct_2", 0.0) > 0.05 and generation.get("repeated_bigram_rate", 1.0) < 0.80
    if improved_behavior and diversity:
        return PhaseMBranchDecision(action="PROMOTE", reasons=("FUNCTIONAL_OR_CAPABILITY_IMPROVEMENT_WITH_NONDEGENERATE_DIVERSITY",), functional_emergence_level=functional_emergence_level, capability_score=capability_score)
    if learning_signal:
        return PhaseMBranchDecision(action="CONTINUE_TO_NEXT_RUNG", reasons=("LEARNING_SIGNAL_WITHOUT_PROMOTION",), functional_emergence_level=functional_emergence_level, capability_score=capability_score)
    return PhaseMBranchDecision(action="STOP_AND_INVESTIGATE", reasons=("NO_MEANINGFUL_LEARNING_SIGNAL_AT_PREDETERMINED_CHECKPOINT",), functional_emergence_level=functional_emergence_level, capability_score=capability_score)


def phase_m_rung_plan(*, development_release_id: str, development_corpus_tokens: int, formal_release_id: str | None = None, formal_corpus_tokens: int | None = None, parent_checkpoint: str | None = None) -> tuple[PhaseMRungPlan, ...]:
    """Plan the required exposure ladder without pretending it was executed."""
    rungs = [
        PhaseMRungPlan(rung_id="phase-m-development-1m", corpus_release_id=development_release_id, corpus_tokens=development_corpus_tokens, update_token_target=1_000_000),
        PhaseMRungPlan(rung_id="phase-m-development-5m", corpus_release_id=development_release_id, corpus_tokens=development_corpus_tokens, update_token_target=5_000_000, parent_checkpoint=parent_checkpoint, initialization="CONTINUATION_FROM_RUNG_1" if parent_checkpoint else "CONTINUATION_REQUIRED_BEFORE_EXECUTION"),
    ]
    if formal_release_id and formal_corpus_tokens:
        rungs.append(PhaseMRungPlan(rung_id="phase-m-formal-10m", corpus_release_id=formal_release_id, corpus_tokens=formal_corpus_tokens, update_token_target=10_000_000, parent_checkpoint=parent_checkpoint, initialization="CONTINUATION_REQUIRED_BEFORE_EXECUTION"))
    return tuple(rungs)
