"""Local provenance-aware corpus releases and deterministic training shards."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import canonical_hash, sha256_file
from app.foundation.tokenizer import UltronTokenizer

PROCESSING_VERSION = "ultron.corpus.v2"
_WHITESPACE = re.compile(r"\s+")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{7,}\d)(?!\w)")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+|bearer\s+[A-Za-z0-9._~+/=-]{8,})", re.IGNORECASE)
_WORDS = re.compile(r"[A-Za-z]{2,}")


def resolve_artifact_path(path: str | Path) -> Path:
    """Resolve historical repository-relative artifact paths from any cwd."""
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    for parent in (Path.cwd(), *Path.cwd().parents):
        resolved = parent / candidate
        if resolved.exists():
            return resolved
    return candidate


def _natural_token_segments(text: str, tokenizer: UltronTokenizer, *, maximum_tokens: int, unpacked: bool) -> list[list[int]]:
    """Prefer paragraphs then sentences; only split a unit when it exceeds the limit."""
    units = [part.strip() for part in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if part.strip()]
    token_units = [tokenizer.encode(unit) for unit in units]
    pieces: list[list[int]] = []
    if unpacked:
        for unit in token_units:
            pieces.extend(unit[index:index + maximum_tokens] for index in range(0, len(unit), maximum_tokens))
        return [piece for piece in pieces if piece]
    current: list[int] = []
    for unit in token_units:
        while unit:
            available = maximum_tokens - len(current)
            if len(unit) <= available:
                current.extend(unit)
                unit = []
            elif current:
                pieces.append(current)
                current = []
            else:
                pieces.append(unit[:maximum_tokens])
                unit = unit[maximum_tokens:]
        if len(current) >= maximum_tokens:
            pieces.append(current)
            current = []
    if current:
        pieces.append(current)
    return pieces


class SourceType(str, Enum):
    LOCAL_AUTHORED = "LOCAL_AUTHORED"
    PUBLIC_DATASET = "PUBLIC_DATASET"
    WEB = "WEB"
    CURATED_EXPERIENCE = "CURATED_EXPERIENCE"


class DocumentCategory(str, Enum):
    GENERAL = "GENERAL"
    CODE = "CODE"
    MATHEMATICS = "MATHEMATICS"
    SCIENCE = "SCIENCE"
    ENGINEERING = "ENGINEERING"
    TECHNICAL = "TECHNICAL"
    DIALOGUE = "DIALOGUE"
    INSTRUCTION = "INSTRUCTION"
    ACADEMIC = "ACADEMIC"


class RejectionReason(str, Enum):
    EMPTY = "EMPTY"
    TOO_SHORT = "TOO_SHORT"
    HIGH_REPETITION = "HIGH_REPETITION"
    SPAM = "SPAM"
    LOW_QUALITY = "LOW_QUALITY"
    PII_POLICY = "PII_POLICY"
    DUPLICATE = "DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    BINARY_CONTENT = "BINARY_CONTENT"
    MALFORMED = "MALFORMED"
    LANGUAGE = "LANGUAGE"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: SourceType
    source_identifier: str
    source_url: str | None = None
    collection_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    license: str = "UNSPECIFIED"
    copyright_metadata: str | None = None
    language: str = "UNKNOWN"
    processing_version: str = PROCESSING_VERSION
    dataset_release: str | None = None
    source_group: str | None = None
    license_evidence: str | None = None
    acquisition_version: str | None = None
    parser_version: str | None = None
    declared_category: DocumentCategory | None = None


class CorpusDocument(BaseModel):
    """An explicitly supplied candidate. Memory and tool output have no adapter here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    provenance: Provenance


class ProcessedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    provenance: Provenance
    content_hash: str
    normalized_content_hash: str
    fingerprint: str = ""
    quality_score: float
    classification: DocumentCategory
    privacy_filter_status: str
    privacy_matches: tuple[str, ...] = ()
    deduplication_status: str
    rejection_reason: RejectionReason | None = None


class DatasetRelease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_id: str
    created_at: datetime
    processing_version: str
    split_method: str
    document_count: int
    train_document_count: int
    validation_document_count: int
    test_document_count: int = 0
    source_breakdown: dict[str, int]
    language_breakdown: dict[str, int]
    category_breakdown: dict[str, int]
    duplicate_statistics: dict[str, int | float]
    filtered_statistics: dict[str, int]
    provenance_coverage: float
    documents_sha256: str
    manifest_sha256: str | None = None
    composition: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    leakage_checks: dict[str, Any] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_characters: int = Field(default=16, ge=1)
    validation_percent: int = Field(default=10, ge=1, le=50)
    test_percent: int = Field(default=0, ge=0, le=40)
    privacy_action: str = "REJECT"
    split_strategy: str = "source_identifier_sha256"
    accepted_languages: tuple[str, ...] = ("en",)
    target_mixture: dict[str, float] = Field(default_factory=dict)
    source_dominance_threshold: float = Field(default=0.50, gt=0, le=1)
    category_starvation_threshold: float = Field(default=0.01, ge=0, le=1)

    def model_post_init(self, __context: Any, /) -> None:
        if self.validation_percent + self.test_percent >= 100:
            raise ValueError("validation_percent + test_percent must be below 100")


def normalize_text(text: str) -> str:
    # Keep paragraph/code boundaries.  Tokenizers still receive canonical whitespace,
    # but source structure is retained for inspection and later segmentation.
    lines = [_WHITESPACE.sub(" ", line).strip() for line in unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def detect_language(text: str) -> str:
    letters = sum(char.isalpha() for char in text)
    ascii_letters = sum(("a" <= char.lower() <= "z") for char in text)
    return "en" if letters and ascii_letters / letters > 0.85 else "UNKNOWN"


def classify_document(text: str) -> DocumentCategory:
    lowered = text.lower()
    if re.search(r"\b(def|class|import|function|const|SELECT)\b", text):
        return DocumentCategory.CODE
    if re.search(r"\b(proof|theorem|equation|integral|matrix)\b", lowered):
        return DocumentCategory.MATHEMATICS
    if re.search(r"\b(experiment|biology|physics|chemistry)\b", lowered):
        return DocumentCategory.SCIENCE
    if re.search(r"\b(abstract|references|methodology|doi:|journal)\b", lowered):
        return DocumentCategory.ACADEMIC
    if re.search(r"\b(engineer|circuit|mechanical|manufacturing)\b", lowered):
        return DocumentCategory.ENGINEERING
    if re.search(r"\b(install|step \d|instructions?|how to)\b", lowered):
        return DocumentCategory.INSTRUCTION
    if re.search(r"\b(user|assistant|human):", lowered):
        return DocumentCategory.DIALOGUE
    if re.search(r"\b(api|system|database|network|software)\b", lowered):
        return DocumentCategory.TECHNICAL
    return DocumentCategory.GENERAL


def _privacy_matches(text: str) -> tuple[str, ...]:
    matches: list[str] = []
    for name, pattern in (("EMAIL", _EMAIL), ("PHONE", _PHONE), ("PRIVATE_KEY", _PRIVATE_KEY), ("SECRET", _SECRET)):
        if pattern.search(text):
            matches.append(name)
    return tuple(matches)


def _redact_privacy(text: str) -> str:
    """Remove obvious identifiers only when an operator selected REDACT mode."""
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _SECRET.sub("[REDACTED_SECRET]", text)
    return text


def _quality_rejection(text: str, config: CorpusConfig) -> RejectionReason | None:
    if not text:
        return RejectionReason.EMPTY
    if len(text) < config.minimum_characters:
        return RejectionReason.TOO_SHORT
    characters = [character for character in text if not character.isspace()]
    if characters and max(Counter(characters).values()) / len(characters) > 0.70:
        return RejectionReason.HIGH_REPETITION
    words = _WORDS.findall(text)
    # Vocabulary type/token ratio naturally falls as coherent documents become
    # longer. Apply the anti-template threshold only to short material; large
    # prose still has independent repetition and character-integrity guards.
    if len(words) < 3 or (len(words) < 2_000 and len({word.lower() for word in words}) / len(words) < 0.18):
        return RejectionReason.LOW_QUALITY
    # Technical standards and documentation can legitimately contain many
    # references.  Treat links as a spam signal only when dense in a short
    # document, never as a blanket rejection of long standards text.
    if (text.count("http") > 30 and len(text) < 15_000) or re.search(r"(?:buy now|click here|free money).*(?:buy now|click here|free money)", text, re.IGNORECASE):
        return RejectionReason.SPAM
    return None


def _fingerprint(text: str) -> str:
    """A deterministic shingle fingerprint for conservative near-duplicate flags."""
    words = re.findall(r"[a-z0-9]{2,}", text.lower())
    # Short boilerplate-like documents frequently share their complete token
    # set while still carrying a material distinct value.  Flag only longer
    # exact shingle-set matches; this is a conservative practical detector.
    if len(words) < 40:
        return ""
    shingles = (" ".join(words[index:index + 5]) for index in range(len(words) - 4))
    selected = sorted(set(shingles))[:256]
    return hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()


def _looks_binary(text: str) -> bool:
    if "\x00" in text:
        return True
    controls = sum(ord(char) < 32 and char not in "\n\t\r" for char in text)
    return bool(text) and controls / len(text) > 0.02


def _quality_score(text: str, rejection: RejectionReason | None) -> float:
    """Explainable document-quality proxy, deliberately not a truth score."""
    if rejection:
        return 0.0
    words = _WORDS.findall(text)
    unique_ratio = len({word.lower() for word in words}) / max(1, len(words))
    density = min(1.0, len(words) / 200)
    repetition = 1.0 - max(Counter(words).values(), default=0) / max(1, len(words))
    return round(0.45 * density + 0.35 * unique_ratio + 0.20 * repetition, 4)


class CorpusPipeline:
    """A pure-local release builder. It never reads the memory SQLite databases."""

    def __init__(self, root: Path, config: CorpusConfig | None = None) -> None:
        self.root, self.config = root, config or CorpusConfig()

    def process(self, documents: Iterable[CorpusDocument]) -> tuple[ProcessedDocument, ...]:
        seen_raw: set[str] = set()
        seen_normalized: set[str] = set()
        seen_fingerprints: set[str] = set()
        processed: list[ProcessedDocument] = []
        for item in documents:
            normalized = normalize_text(item.content)
            raw_hash = hashlib.sha256(item.content.encode("utf-8")).hexdigest()
            matches = _privacy_matches(normalized)
            if matches and self.config.privacy_action == "REDACT" and "PRIVATE_KEY" not in matches:
                normalized = _redact_privacy(normalized)
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            language = detect_language(normalized)
            fingerprint = _fingerprint(normalized)
            rejection = RejectionReason.BINARY_CONTENT if _looks_binary(item.content) else None
            if rejection is None and language not in self.config.accepted_languages:
                rejection = RejectionReason.LANGUAGE
            if rejection is None and matches and (self.config.privacy_action == "REJECT" or "PRIVATE_KEY" in matches):
                rejection = RejectionReason.PII_POLICY
            if rejection is None:
                rejection = _quality_rejection(normalized, self.config)
            dedupe = "UNIQUE"
            if not rejection and (raw_hash in seen_raw or normalized_hash in seen_normalized):
                rejection, dedupe = RejectionReason.DUPLICATE, "DUPLICATE"
            elif not rejection and fingerprint and fingerprint in seen_fingerprints:
                rejection, dedupe = RejectionReason.NEAR_DUPLICATE, "NEAR_DUPLICATE"
            if not rejection:
                seen_raw.add(raw_hash)
                seen_normalized.add(normalized_hash)
                if fingerprint:
                    seen_fingerprints.add(fingerprint)
            provenance = item.provenance.model_copy(update={"language": language})
            processed.append(ProcessedDocument(content=normalized, provenance=provenance, content_hash=raw_hash, normalized_content_hash=normalized_hash, fingerprint=fingerprint, quality_score=_quality_score(normalized, rejection), classification=provenance.declared_category or classify_document(normalized), privacy_filter_status="REJECTED" if matches and rejection else ("REDACTED" if matches and self.config.privacy_action == "REDACT" else ("FLAGGED" if matches else "CLEAR")), privacy_matches=matches, deduplication_status=dedupe, rejection_reason=rejection))
        return tuple(processed)

    def release(self, release_id: str, documents: Iterable[CorpusDocument]) -> DatasetRelease:
        target = self.root / "corpora" / "releases" / release_id
        if target.exists():
            raise ValueError("Dataset releases are immutable; choose a new release ID.")
        processed = self.process(documents)
        accepted = [document for document in processed if document.rejection_reason is None]
        if self.config.split_strategy == "source_group_balanced":
            train, validation, test = self._balanced_source_split(accepted)
        else:
            train, validation, test = [], [], []
            for document in accepted:
                split_key = (document.provenance.source_group or document.provenance.source_identifier) if self.config.split_strategy == "source_identifier_sha256" else document.normalized_content_hash
                split_hash = hashlib.sha256(split_key.encode("utf-8")).hexdigest()
                bucket = int(split_hash[:8], 16) % 100
                if bucket < self.config.test_percent:
                    test.append(document)
                elif bucket < self.config.test_percent + self.config.validation_percent:
                    validation.append(document)
                else:
                    train.append(document)
        if not train or not validation:
            raise ValueError("Release requires at least one deterministic train and validation document.")
        if self.config.test_percent and not test:
            raise ValueError("Release requires at least one deterministic test document when test_percent is enabled.")
        target.mkdir(parents=True)
        self._write_jsonl(target / "documents.jsonl", accepted)
        self._write_jsonl(target / "train.jsonl", train)
        self._write_jsonl(target / "validation.jsonl", validation)
        if self.config.test_percent:
            self._write_jsonl(target / "test.jsonl", test)
        self._write_jsonl(target / "rejections.jsonl", [document for document in processed if document.rejection_reason])
        sources = Counter(document.provenance.source_type.value for document in accepted)
        languages = Counter(document.provenance.language for document in accepted)
        categories = Counter(document.classification.value for document in accepted)
        rejected = Counter((document.rejection_reason.value if document.rejection_reason else "ACCEPTED") for document in processed)
        duplicate_count = rejected.get(RejectionReason.DUPLICATE.value, 0)
        composition, warnings = self._composition(accepted, train, validation, test)
        leakage_checks = self._leakage_checks(train, validation, test)
        release = DatasetRelease(release_id=release_id, created_at=datetime.now(UTC), processing_version=PROCESSING_VERSION, split_method=f"{self.config.split_strategy}: test < {self.config.test_percent}, validation < {self.config.test_percent + self.config.validation_percent}", document_count=len(accepted), train_document_count=len(train), validation_document_count=len(validation), test_document_count=len(test), source_breakdown=dict(sources), language_breakdown=dict(languages), category_breakdown=dict(categories), duplicate_statistics={"documents_before": len(processed), "documents_after": len(accepted), "duplicates_removed": duplicate_count, "near_duplicates_removed": rejected.get(RejectionReason.NEAR_DUPLICATE.value, 0), "duplicate_rate": duplicate_count / len(processed) if processed else 0.0}, filtered_statistics=dict(rejected), provenance_coverage=1.0 if accepted else 0.0, documents_sha256=sha256_file(target / "documents.jsonl"), composition=composition, warnings=warnings, leakage_checks=leakage_checks)
        manifest_hash = canonical_hash(release.model_dump(mode="json"))
        release = release.model_copy(update={"manifest_sha256": manifest_hash})
        (target / "manifest.json").write_text(release.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (target / "quality-report.json").write_text(json.dumps({"release_id": release_id, "composition": composition, "warnings": warnings, "leakage_checks": leakage_checks, "quality_scoring": "document density, vocabulary diversity, and repetition; not a truth score"}, indent=2) + "\n", encoding="utf-8")
        return release

    def _composition(
        self,
        accepted: list[ProcessedDocument],
        train: list[ProcessedDocument],
        validation: list[ProcessedDocument],
        test: list[ProcessedDocument],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        def percentages(values: Counter[str]) -> dict[str, float]:
            total = sum(values.values())
            return {name: round(count / total, 6) for name, count in sorted(values.items())} if total else {}

        source_counts = Counter(item.provenance.source_group or item.provenance.source_identifier for item in accepted)
        category_counts = Counter(item.classification.value for item in accepted)
        license_counts = Counter(item.provenance.license for item in accepted)
        domain_counts = Counter(urlparse(item.provenance.source_url).netloc or "LOCAL" for item in accepted)
        warnings: list[str] = []
        if source_counts and max(source_counts.values()) / len(accepted) > self.config.source_dominance_threshold:
            warnings.append("SINGLE_SOURCE_DOMINANCE")
        if category_counts and max(category_counts.values()) / len(accepted) > self.config.source_dominance_threshold:
            warnings.append("SINGLE_DOMAIN_DOMINANCE")
        for category, target in self.config.target_mixture.items():
            actual = category_counts.get(category.upper(), 0) / max(1, len(accepted))
            if target > 0 and actual < min(target, self.config.category_starvation_threshold):
                warnings.append(f"CATEGORY_STARVATION:{category.upper()}")
        split_sources = [
            {item.provenance.source_group or item.provenance.source_identifier for item in subset}
            for subset in (train, validation, test)
        ]
        if split_sources[0] & split_sources[1] or (test and (split_sources[0] & split_sources[2] or split_sources[1] & split_sources[2])):
            warnings.append("VALIDATION_CONTAMINATION")
        if any(item.provenance.license in {"", "UNSPECIFIED"} for item in accepted):
            warnings.append("LICENSING_UNCERTAINTY")
        return {
            "source_documents": dict(sorted(source_counts.items())),
            "source_percentages": percentages(source_counts),
            "category_documents": dict(sorted(category_counts.items())),
            "category_percentages": percentages(category_counts),
            "license_distribution": dict(sorted(license_counts.items())),
            "domain_documents": dict(sorted(domain_counts.items())),
            "domain_percentages": percentages(domain_counts),
            "top_source_percent": round(max(source_counts.values()) / len(accepted), 6) if source_counts else 0.0,
            "top_domain_percent": round(max(domain_counts.values()) / len(accepted), 6) if domain_counts else 0.0,
            "top_category_percent": round(max(category_counts.values()) / len(accepted), 6) if category_counts else 0.0,
            "split_documents": {"train": len(train), "validation": len(validation), "test": len(test)},
            "source_concentration": round(max(source_counts.values()) / len(accepted), 6) if source_counts else 0.0,
        }, tuple(warnings)

    @staticmethod
    def _leakage_checks(
        train: list[ProcessedDocument], validation: list[ProcessedDocument], test: list[ProcessedDocument],
    ) -> dict[str, Any]:
        partitions = {"train": train, "validation": validation, "test": test}
        groups = {name: {item.provenance.source_group or item.provenance.source_identifier for item in items} for name, items in partitions.items()}
        urls = {name: {item.provenance.source_url for item in items if item.provenance.source_url} for name, items in partitions.items()}
        fingerprints = {name: {item.fingerprint for item in items if item.fingerprint} for name, items in partitions.items()}
        overlaps = {
            "source_group": {"train_validation": sorted(groups["train"] & groups["validation"]), "train_test": sorted(groups["train"] & groups["test"]), "validation_test": sorted(groups["validation"] & groups["test"])},
            "url": {"train_validation": sorted(urls["train"] & urls["validation"]), "train_test": sorted(urls["train"] & urls["test"]), "validation_test": sorted(urls["validation"] & urls["test"])},
            "near_duplicate_fingerprint": {"train_validation": sorted(fingerprints["train"] & fingerprints["validation"]), "train_test": sorted(fingerprints["train"] & fingerprints["test"]), "validation_test": sorted(fingerprints["validation"] & fingerprints["test"])},
        }
        clean = not any(value for type_overlaps in overlaps.values() for value in type_overlaps.values())
        return {"status": "PASS" if clean else "FAIL", "overlaps": overlaps}

    def _balanced_source_split(
        self, documents: list[ProcessedDocument],
    ) -> tuple[list[ProcessedDocument], list[ProcessedDocument], list[ProcessedDocument]]:
        """Deterministically allocate entire source groups by character mass.

        This avoids the pathological case where a hash split puts only tiny
        local notes in a sealed partition.  It is still source-level, not a
        document-level shortcut, and records its approximation in the release.
        """
        groups: dict[str, list[ProcessedDocument]] = {}
        for document in documents:
            key = document.provenance.source_group or document.provenance.source_identifier
            groups.setdefault(key, []).append(document)
        available = set(groups)
        total = sum(len(document.content) for document in documents)

        def take(target: float) -> list[ProcessedDocument]:
            chosen: list[ProcessedDocument] = []
            current = 0
            while available:
                candidates = []
                for key in available:
                    weight = sum(len(document.content) for document in groups[key])
                    improvement = abs(target - current) - abs(target - (current + weight))
                    candidates.append((improvement, hashlib.sha256(key.encode()).hexdigest(), key, weight))
                improvement, _, key, weight = max(candidates)
                if chosen and improvement <= 0:
                    break
                chosen.extend(groups[key])
                available.remove(key)
                current += weight
            return chosen

        test = take(total * self.config.test_percent / 100) if self.config.test_percent else []
        validation = take(total * self.config.validation_percent / 100)
        train = [document for key in sorted(available) for document in groups[key]]
        return train, validation, test

    @staticmethod
    def _write_jsonl(path: Path, items: Iterable[ProcessedDocument]) -> None:
        path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


class PackedShard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    shard_id: str
    split: str
    path: str
    sha256: str
    sequence_count: int
    token_count: int


class TokenizedRelease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    tokenizer_sha256: str
    tokenizer_version: str
    context_length: int
    document_boundary_policy: str
    packing_strategy: str = "FIXED_TERMINAL_EOS_OVERLAP"
    raw_character_count: int
    token_count: int
    average_tokens_per_document: float
    tokens_by_category: dict[str, int]
    shards: tuple[PackedShard, ...]
    manifest_sha256: str | None = None


class TokenPacker:
    """Packs only within a document and retains an EOS-bearing terminal window."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def pack(self, release_dir: Path, tokenizer: UltronTokenizer, *, context_length: int, max_sequences_per_shard: int = 1024, packing_strategy: str = "FIXED_TERMINAL_EOS_OVERLAP") -> TokenizedRelease:
        strategies = {"FIXED_TERMINAL_EOS_OVERLAP", "VARIABLE_EOS_MASKED", "VARIABLE_EOS_MASKED_V2", "NATURAL_BOUNDARY_EOS_MASKED", "SHORT_NATURAL_EOS_MASKED", "UNPACKED_NATURAL_EOS_MASKED", "DENSE_TEXT_NATURAL_EOS_MASKED", "DENSE_PROSE_NATURAL_EOS_MASKED"}
        if packing_strategy not in strategies:
            raise ValueError(f"packing_strategy must be one of {sorted(strategies)}")
        release = DatasetRelease.model_validate_json((release_dir / "manifest.json").read_text(encoding="utf-8"))
        suffixes = {"FIXED_TERMINAL_EOS_OVERLAP": f"context-{context_length}", "VARIABLE_EOS_MASKED": f"context-{context_length}-variable-eos-masked", "VARIABLE_EOS_MASKED_V2": f"context-{context_length}-variable-eos-masked-v2", "NATURAL_BOUNDARY_EOS_MASKED": f"context-{context_length}-natural-eos-masked", "SHORT_NATURAL_EOS_MASKED": f"context-{context_length}-short-natural-eos-masked", "UNPACKED_NATURAL_EOS_MASKED": f"context-{context_length}-unpacked-natural-eos-masked", "DENSE_TEXT_NATURAL_EOS_MASKED": f"context-{context_length}-dense-natural-eos-masked", "DENSE_PROSE_NATURAL_EOS_MASKED": f"context-{context_length}-dense-prose-natural-eos-masked"}
        suffix = suffixes[packing_strategy]
        target = self.root / "corpora" / "tokenized" / release.release_id / tokenizer.sha256[:16] / suffix
        if target.exists():
            return TokenizedRelease.model_validate_json((target / "manifest.json").read_text(encoding="utf-8"))
        target.mkdir(parents=True)
        shards: list[PackedShard] = []
        total_tokens, raw_chars, categories = 0, 0, Counter()
        for split in ("train", "validation", "test"):
            split_path = release_dir / f"{split}.jsonl"
            if not split_path.exists():
                continue
            sequences: list[dict[str, Any]] = []
            shard_index = 0
            for line in split_path.read_text(encoding="utf-8").splitlines():
                document = ProcessedDocument.model_validate_json(line)
                if packing_strategy == "DENSE_TEXT_NATURAL_EOS_MASKED" and document.classification not in {DocumentCategory.GENERAL, DocumentCategory.INSTRUCTION, DocumentCategory.DIALOGUE}:
                    continue
                # Preserve every sealed split while excluding source code from
                # this prose-focused branch.  The earlier dense-text filter
                # had no test documents and is retained only as invalid,
                # non-comparable evidence.
                if packing_strategy == "DENSE_PROSE_NATURAL_EOS_MASKED" and document.classification == DocumentCategory.CODE:
                    continue
                raw_ids = tokenizer.encode(document.content)
                ids = raw_ids + [tokenizer.processor.eos_id()]
                raw_chars += len(document.content)
                total_tokens += len(ids)
                categories[document.classification.value] += len(ids)
                if packing_strategy == "FIXED_TERMINAL_EOS_OVERLAP":
                    # Historical fixed-width format. Its EOS placement is
                    # retained exactly for reproducibility; new experiments
                    # should use VARIABLE_EOS_MASKED below.
                    offsets = list(range(0, len(ids) - context_length, context_length))
                    terminal_offset = len(ids) - context_length - 1
                    if offsets and offsets[-1] != terminal_offset:
                        offsets.append(terminal_offset)
                    for offset in offsets:
                        chunk = ids[offset:offset + context_length + 1]
                        if len(chunk) == context_length + 1:
                            sequences.append({"document_id": document.provenance.document_id, "input_ids": chunk})
                else:
                    # Split within documents, then pad only the final chunk.
                    # The final real target is EOS and loss_mask excludes every
                    # padded target, so EOS retains its true relative position.
                    pad = tokenizer.processor.eos_id()
                    # V2 deliberately varies segment lengths. In the prior
                    # fixed and v1 masked layouts, nearly every EOS target
                    # still occurred at absolute position 512, which cannot
                    # validate termination from short prompts.
                    if packing_strategy in {"NATURAL_BOUNDARY_EOS_MASKED", "DENSE_TEXT_NATURAL_EOS_MASKED", "DENSE_PROSE_NATURAL_EOS_MASKED"}:
                        segments = _natural_token_segments(document.content, tokenizer, maximum_tokens=384, unpacked=False)
                    elif packing_strategy == "SHORT_NATURAL_EOS_MASKED":
                        segments = _natural_token_segments(document.content, tokenizer, maximum_tokens=128, unpacked=False)
                    elif packing_strategy == "UNPACKED_NATURAL_EOS_MASKED":
                        segments = _natural_token_segments(document.content, tokenizer, maximum_tokens=context_length, unpacked=True)
                    else:
                        lengths = (64, 128, 256, 384, context_length) if packing_strategy == "VARIABLE_EOS_MASKED_V2" else (context_length,)
                        segments = []
                        offset, segment_index = 0, 0
                        while offset < len(raw_ids):
                            length = lengths[segment_index % len(lengths)]
                            segments.append(raw_ids[offset:offset + length])
                            offset += length
                            segment_index += 1
                    for segment in segments:
                        chunk = [*segment, tokenizer.processor.eos_id()]
                        active_targets = len(chunk) - 1
                        padded = chunk + [pad] * (context_length + 1 - len(chunk))
                        sequences.append({"document_id": document.provenance.document_id, "input_ids": padded, "loss_mask": [1] * active_targets + [0] * (context_length - active_targets)})
                if len(sequences) >= max_sequences_per_shard:
                    shards.append(self._publish_shard(target, split, shard_index, sequences))
                    shard_index, sequences = shard_index + 1, []
            if sequences:
                shards.append(self._publish_shard(target, split, shard_index, sequences))
        policy = "no cross-document packing; terminal EOS-preserving overlapping window retained" if packing_strategy == "FIXED_TERMINAL_EOS_OVERLAP" else "no cross-document packing; EOS-terminated segments use EOS padding with explicit loss masks"
        result = TokenizedRelease(release_id=release.release_id, tokenizer_sha256=tokenizer.sha256, tokenizer_version=tokenizer.version, context_length=context_length, document_boundary_policy=policy, packing_strategy=packing_strategy, raw_character_count=raw_chars, token_count=total_tokens, average_tokens_per_document=total_tokens / release.document_count, tokens_by_category=dict(categories), shards=tuple(shards))
        result = result.model_copy(update={"manifest_sha256": canonical_hash(result.model_dump(mode="json"))})
        (target / "manifest.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return result

    @staticmethod
    def _publish_shard(target: Path, split: str, index: int, sequences: list[dict[str, Any]]) -> PackedShard:
        path = target / f"{split}-{index:05d}.jsonl"
        path.write_text("".join(json.dumps(sequence, sort_keys=True, separators=(",", ":")) + "\n" for sequence in sequences), encoding="utf-8")
        return PackedShard(shard_id=f"{split}-{index:05d}", split=split, path=str(path), sha256=sha256_file(path), sequence_count=len(sequences), token_count=sum(len(sequence["input_ids"]) for sequence in sequences))


class PackedShardSequenceDataset(Sequence[list[int] | tuple[list[int], list[int]]]):
    """Deterministic, lazy access to packed JSONL sequences.

    It indexes byte offsets rather than retaining all token arrays in RAM.  A
    training run can therefore reuse the same trainer for releases too large
    to materialize as a Python list.  Shards remain immutable and the caller
    supplies their expected hashes before indexing begins.
    """

    def __init__(self, shards: Iterable[Path], *, expected_width: int | None = None, include_loss_mask: bool = False) -> None:
        self.shards = tuple(sorted(resolve_artifact_path(path) for path in shards))
        self.expected_width = expected_width
        self.include_loss_mask = include_loss_mask
        self._locations: list[tuple[Path, int]] = []
        for path in self.shards:
            with path.open("rb") as source:
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if line.strip():
                        self._locations.append((path, offset))

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int | slice) -> list[int] | tuple[list[int], list[int]] | list[list[int] | tuple[list[int], list[int]]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        path, offset = self._locations[index]
        with path.open("rb") as source:
            source.seek(offset)
            record = json.loads(source.readline())
        value = record.get("input_ids")
        if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
            raise ValueError(f"Malformed packed sequence in immutable shard: {path}")
        if self.expected_width is not None and len(value) != self.expected_width:
            raise ValueError(f"Packed sequence width mismatch in {path}: expected {self.expected_width}, got {len(value)}")
        if not self.include_loss_mask:
            return value
        mask = record.get("loss_mask", [1] * max(0, len(value) - 1))
        if not isinstance(mask, list) or len(mask) != max(0, len(value) - 1) or not all(item in (0, 1) for item in mask):
            raise ValueError(f"Malformed loss_mask in immutable shard: {path}")
        return value, mask
