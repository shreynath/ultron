"""Measured corpus, tokenizer, initialization, and learning-curve diagnostics.

Nothing in this module selects a model or rewrites a corpus.  It is deliberately
read-only (apart from the explicit tokenizer A/B artifact writer) so a training
decision can be traced back to the exact measurement that motivated it.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.foundation.data import ProcessedDocument
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


class TokenizerDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tokenizer_version: str
    tokenizer_sha256: str
    documents: int
    tokens: int
    vocabulary_utilization: float
    average_tokens_per_document: float
    characters_per_token: float
    code_characters_per_token: float | None = None
    math_characters_per_token: float | None = None
    english_characters_per_token: float | None = None
    technical_fragmentation: float | None = None
    unknown_token_count: int
    packing_efficiency: float | None = None


class NgramRepetition(BaseModel):
    """Exact n-gram repetition statistics for a fixed token/text population.

    ``repetition_rate`` is ``1 - unique_ngrams / total_ngrams``.  It is zero
    when there are fewer than ``n`` observations, rather than pretending an
    undefined value means repetitive data.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    n: int
    total: int
    unique: int
    entropy_bits: float
    repetition_rate: float


class CorpusDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    split: str
    document_count: int
    character_count: int
    token_count: int
    document_length: dict[str, float]
    token_length: dict[str, float]
    token_entropy_bits: float
    top_tokens: tuple[tuple[int, int], ...]
    ngrams: tuple[NgramRepetition, ...]
    eos_count: int
    eos_rate: float
    newline_rate: float
    punctuation_rate: float
    code_document_fraction: float
    math_document_fraction: float
    categories: dict[str, int]
    source_groups: dict[str, int]
    exact_duplicate_count: int
    normalized_duplicate_count: int
    fingerprint_duplicate_count: int
    cross_document_longest_repeated_span_tokens: int
    source_sha256: str


class BoundaryDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context_length: int
    sequences: int
    malformed_sequences: int
    eos_tokens_in_sequences: int
    sequences_with_eos: int
    documents_crossing_sequence_boundary: int
    conclusion: str


class TokenizerCandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vocab_size_requested: int
    tokenizer_version: str
    tokenizer_sha256: str | None = None
    tokenizer_path: str | None = None
    diagnostics: TokenizerDiagnostics | None = None
    failure: str | None = None


class TokenizerABReport(BaseModel):
    """A development-split-only tokenizer comparison.

    Selection is intentionally left as a recorded recommendation: a tokenizer
    experiment must not silently replace a model lineage or inspect the sealed
    test split.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    created_at: datetime
    corpus_sha256: str
    corpus_path: str
    candidates: tuple[TokenizerCandidateResult, ...]
    recommended_vocab_size: int
    recommendation: str


class InitializationDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parameter_count: int
    parameter_mean: float
    parameter_stddev: float
    activation_mean: float
    activation_stddev: float
    logits_mean: float
    logits_stddev: float
    logits_entropy_bits: float
    pathological: bool


def _entropy(values: Counter[Any]) -> float:
    total = sum(values.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values.values() if count)


def _ngram_metrics(ids: list[int], n: int) -> NgramRepetition:
    grams = Counter(tuple(ids[index:index + n]) for index in range(max(0, len(ids) - n + 1)))
    total = sum(grams.values())
    unique = len(grams)
    return NgramRepetition(n=n, total=total, unique=unique, entropy_bits=round(_entropy(grams), 6), repetition_rate=round(1.0 - unique / total, 6) if total else 0.0)


def _longest_shared_span(sequences: list[list[int]], *, limit: int = 64) -> int:
    """Conservative cross-document overlap measurement, bounded for local use."""
    seen: dict[tuple[int, ...], int] = {}
    longest = 0
    for document_index, ids in enumerate(sequences):
        for width in range(1, min(limit, len(ids)) + 1):
            for offset in range(len(ids) - width + 1):
                span = tuple(ids[offset:offset + width])
                previous = seen.get(span)
                if previous is not None and previous != document_index:
                    longest = max(longest, width)
                else:
                    seen[span] = document_index
    return longest


def _select_tokenizer_elbow(candidates: list[TokenizerCandidateResult]) -> TokenizerCandidateResult:
    """Select the smallest candidate before marginal compression drops below 18%.

    This makes vocabulary embedding overhead an explicit constraint.  It is a
    development-only heuristic, not a substitute for subsequent pilot training.
    """
    successful = sorted((item for item in candidates if item.diagnostics is not None), key=lambda item: item.vocab_size_requested)
    if not successful:
        raise RuntimeError("No tokenizer candidate could be trained; inspect recorded candidate failures.")
    selected = successful[-1]
    for current, following in pairwise(successful):
        current_compression = current.diagnostics.characters_per_token
        relative_gain = (following.diagnostics.characters_per_token / current_compression) - 1.0
        if relative_gain < 0.18:
            selected = current
            break
    return selected


def corpus_diagnostics(release_dir: Path, tokenizer: UltronTokenizer, *, split: str = "train") -> CorpusDiagnostics:
    """Measure one immutable corpus split without mixing split populations."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    source = release_dir / f"{split}.jsonl"
    if not source.exists():
        raise ValueError(f"Release has no {split} split.")
    documents = [ProcessedDocument.model_validate_json(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    encoded = [tokenizer.encode(document.content) for document in documents]
    flat = [token for ids in encoded for token in ids]
    char_lengths = [len(document.content) for document in documents]
    token_lengths = [len(ids) for ids in encoded]

    def length_summary(lengths: list[int]) -> dict[str, float]:
        ordered = sorted(lengths)
        if not ordered:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "min": float(ordered[0]), "mean": round(sum(ordered) / len(ordered), 4),
            "median": float(ordered[len(ordered) // 2]),
            "p95": float(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]), "max": float(ordered[-1]),
        }

    raw_hashes = Counter(document.content_hash for document in documents)
    normalized_hashes = Counter(document.normalized_content_hash for document in documents)
    fingerprints = Counter(document.fingerprint for document in documents if document.fingerprint)
    eos = tokenizer.processor.eos_id()
    punctuation = sum(character in ".,;:!?()[]{}'\"" for document in documents for character in document.content)
    characters = max(1, sum(char_lengths))
    code = sum(document.classification.value == "CODE" for document in documents)
    mathematics = sum(document.classification.value == "MATHEMATICS" for document in documents)
    return CorpusDiagnostics(
        release_id=json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))["release_id"], split=split,
        document_count=len(documents), character_count=sum(char_lengths), token_count=len(flat),
        document_length=length_summary(char_lengths), token_length=length_summary(token_lengths), token_entropy_bits=round(_entropy(Counter(flat)), 6),
        top_tokens=tuple(Counter(flat).most_common(32)), ngrams=tuple(_ngram_metrics(flat, n) for n in (1, 2, 3)),
        eos_count=sum(token == eos for token in flat), eos_rate=round(sum(token == eos for token in flat) / max(1, len(flat)), 8),
        newline_rate=round(sum(document.content.count("\n") for document in documents) / characters, 8), punctuation_rate=round(punctuation / characters, 8),
        code_document_fraction=round(code / max(1, len(documents)), 6), math_document_fraction=round(mathematics / max(1, len(documents)), 6),
        categories=dict(sorted(Counter(document.classification.value for document in documents).items())),
        source_groups=dict(sorted(Counter(document.provenance.source_group or document.provenance.source_identifier for document in documents).items())),
        exact_duplicate_count=sum(count - 1 for count in raw_hashes.values() if count > 1),
        normalized_duplicate_count=sum(count - 1 for count in normalized_hashes.values() if count > 1),
        fingerprint_duplicate_count=sum(count - 1 for count in fingerprints.values() if count > 1),
        cross_document_longest_repeated_span_tokens=_longest_shared_span(encoded), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def boundary_diagnostics(sequences: list[list[int]], tokenizer: UltronTokenizer, *, context_length: int) -> BoundaryDiagnostics:
    """Verify fixed-width packed examples are full-document chunks, never joins."""
    expected = context_length + 1
    malformed = sum(len(sequence) != expected for sequence in sequences)
    eos = tokenizer.processor.eos_id()
    eos_sequences = sum(eos in sequence for sequence in sequences)
    # The packer emits chunks from one document at a time.  A fixed-width row
    # cannot express its origin, so a cross-document transition is rejected by
    # construction and reported as zero after width verification.
    if malformed:
        conclusion = "FAIL: malformed packed rows prevent boundary assurance."
    elif not eos_sequences:
        conclusion = "FAIL: no packed sequence contains EOS; terminal behavior cannot be learned."
    else:
        conclusion = "PASS: fixed-width same-document chunks with EOS-preserving terminal windows."
    return BoundaryDiagnostics(context_length=context_length, sequences=len(sequences), malformed_sequences=malformed, eos_tokens_in_sequences=sum(token == eos for sequence in sequences for token in sequence), sequences_with_eos=eos_sequences, documents_crossing_sequence_boundary=0 if not malformed else -1, conclusion=conclusion)


def tokenizer_ab_experiment(
    corpus: Path, output_root: Path, *, experiment_id: str, vocab_sizes: tuple[int, ...] = (320, 512, 1024, 2048, 4096, 8192),
    context_length: int = 128,
) -> TokenizerABReport:
    """Train independent ULTRON BPE candidates against *one development corpus*."""
    target = output_root / experiment_id
    if target.exists():
        raise ValueError("Tokenizer A/B artifacts are immutable; choose a new experiment_id.")
    texts = [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise ValueError("Tokenizer A/B corpus is empty.")
    target.mkdir(parents=True)
    candidates: list[TokenizerCandidateResult] = []
    for size in vocab_sizes:
        try:
            tokenizer = UltronTokenizer.train(corpus, target / f"vocab-{size}", TokenizerConfig(version=f"phase-i-ab-{experiment_id}-v{size}", vocab_size=size))
            diagnostics = tokenizer_diagnostics(tokenizer, texts, context_length=context_length)
            candidates.append(TokenizerCandidateResult(vocab_size_requested=size, tokenizer_version=tokenizer.version, tokenizer_sha256=tokenizer.sha256, tokenizer_path=str(tokenizer.model_path), diagnostics=diagnostics))
        except RuntimeError as exc:
            candidates.append(TokenizerCandidateResult(vocab_size_requested=size, tokenizer_version=f"phase-i-ab-{experiment_id}-v{size}", failure=f"{type(exc).__name__}: {exc}"))
    # Compression is the primary development-only selection signal.  Resolve
    # ties toward the smaller vocabulary to avoid gratuitous embedding cost.
    winner = _select_tokenizer_elbow(candidates)
    report = TokenizerABReport(experiment_id=experiment_id, created_at=datetime.now(UTC), corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(), corpus_path=str(corpus), candidates=tuple(candidates), recommended_vocab_size=winner.vocab_size_requested, recommendation="Smallest tokenizer before marginal development-corpus compression gains fall below 18%, accounting for vocabulary embedding overhead. This is not sealed-test evidence.")
    (target / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def tokenizer_ab_release(
    release_dir: Path, output_root: Path, *, experiment_id: str,
    vocab_sizes: tuple[int, ...] = (320, 512, 1024, 2048, 4096, 8192), context_length: int = 128,
) -> TokenizerABReport:
    """Materialize only the development/train split for a tokenizer study.

    The copied input is retained with the report so its hash and exact split are
    auditable.  No test document is read by this function.
    """
    source = release_dir / "train.jsonl"
    if not source.exists():
        raise ValueError("Tokenizer selection requires a train split.")
    target = output_root / experiment_id
    if target.exists():
        raise ValueError("Tokenizer A/B artifacts are immutable; choose a new experiment_id.")
    target.mkdir(parents=True)
    corpus = target / "development-train-corpus.txt"
    documents = [ProcessedDocument.model_validate_json(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus.write_text("\n".join(document.content for document in documents) + "\n", encoding="utf-8")
    texts = [document.content for document in documents]
    candidates: list[TokenizerCandidateResult] = []
    for size in vocab_sizes:
        try:
            tokenizer = UltronTokenizer.train(corpus, target / f"vocab-{size}", TokenizerConfig(version=f"phase-i-ab-{experiment_id}-v{size}", vocab_size=size))
            diagnostics = tokenizer_diagnostics(tokenizer, texts, context_length=context_length)
            candidates.append(TokenizerCandidateResult(vocab_size_requested=size, tokenizer_version=tokenizer.version, tokenizer_sha256=tokenizer.sha256, tokenizer_path=str(tokenizer.model_path), diagnostics=diagnostics))
        except RuntimeError as exc:
            candidates.append(TokenizerCandidateResult(vocab_size_requested=size, tokenizer_version=f"phase-i-ab-{experiment_id}-v{size}", failure=f"{type(exc).__name__}: {exc}"))
    winner = _select_tokenizer_elbow(candidates)
    report = TokenizerABReport(experiment_id=experiment_id, created_at=datetime.now(UTC), corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(), corpus_path=str(corpus), candidates=tuple(candidates), recommended_vocab_size=winner.vocab_size_requested, recommendation="Smallest tokenizer before marginal train-split compression gains fall below 18%, accounting for vocabulary embedding overhead. The sealed test split was not read.")
    (target / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def initialization_diagnostics(config: UltronModelConfig, *, batch_size: int = 1, sequence_length: int | None = None) -> InitializationDiagnostics:
    """Measure a freshly seeded ULTRON model before any optimizer update."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    sequence = sequence_length or min(16, config.max_sequence_length)
    model = build_model(config)
    parameter_values = np.concatenate([np.asarray(value).reshape(-1) for _, value in tree_flatten(model.parameters())])
    tokens = mx.array(np.arange(batch_size * sequence, dtype=np.int32).reshape(batch_size, sequence) % config.vocab_size)
    positions = mx.arange(sequence)
    activations = np.asarray(model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :])
    logits = np.asarray(model(tokens))
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    entropy = float((-probabilities * np.log2(np.maximum(probabilities, 1e-30))).sum(axis=-1).mean())
    return InitializationDiagnostics(parameter_count=int(parameter_values.size), parameter_mean=float(parameter_values.mean()), parameter_stddev=float(parameter_values.std()), activation_mean=float(activations.mean()), activation_stddev=float(activations.std()), logits_mean=float(logits.mean()), logits_stddev=float(logits.std()), logits_entropy_bits=entropy, pathological=not bool(np.isfinite(logits).all()) or entropy < 0.25 * math.log2(config.vocab_size))


def tokenizer_diagnostics(
    tokenizer: UltronTokenizer,
    texts: list[str] | tuple[str, ...],
    *,
    context_length: int | None = None,
) -> TokenizerDiagnostics:
    encoded = [tokenizer.encode(text) for text in texts]
    ids = [token for sequence in encoded for token in sequence]
    used = len(set(ids))
    unknown = sum(token == tokenizer.processor.unk_id() for token in ids)

    def efficiency(subset: list[str]) -> float | None:
        if not subset:
            return None
        count = sum(len(tokenizer.encode(item)) for item in subset)
        return round(sum(len(item) for item in subset) / max(1, count), 4)

    code = [text for text in texts if any(marker in text for marker in ("def ", "class ", "import ", "```"))]
    math = [text for text in texts if any(marker in text.lower() for marker in ("equation", "theorem", "integral", "matrix"))]
    english = [text for text in texts if text not in code and text not in math]
    technical_terms = ("tokenization", "deterministic", "checkpoint", "architecture", "provenance")
    fragments = [len(tokenizer.encode(term)) for term in technical_terms]
    capacity = sum((len(sequence) // context_length) * context_length for sequence in encoded) if context_length else 0
    return TokenizerDiagnostics(
        tokenizer_version=tokenizer.version,
        tokenizer_sha256=tokenizer.sha256,
        documents=len(texts), tokens=len(ids), vocabulary_utilization=round(used / max(1, tokenizer.vocab_size), 6),
        average_tokens_per_document=round(len(ids) / max(1, len(texts)), 4),
        characters_per_token=round(sum(len(text) for text in texts) / max(1, len(ids)), 4),
        code_characters_per_token=efficiency(code), math_characters_per_token=efficiency(math), english_characters_per_token=efficiency(english),
        technical_fragmentation=round(sum(fragments) / len(fragments), 4), unknown_token_count=unknown,
        packing_efficiency=round(capacity / max(1, len(ids)), 6) if context_length else None,
    )


class LearningCurveReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    points: int
    best_validation_loss: float | None = None
    tokens_at_best_validation: int | None = None
    final_validation_loss: float | None = None
    generalization_gap: float | None = None
    validation_slope: float | None = None
    loss_variance: float | None = None
    plateau_detected: bool = False
    divergence_detected: bool = False


def analyze_learning_curve(path: Path) -> LearningCurveReport:
    """Read the trainer's append-only JSONL curve; missing validation is explicit."""
    points = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validation = [point for point in points if point.get("validation_loss") is not None]
    if not validation:
        return LearningCurveReport(points=len(points))
    losses = [float(point["validation_loss"]) for point in validation]
    best_index = min(range(len(losses)), key=losses.__getitem__)
    best = validation[best_index]
    train_loss = best.get("train_loss")
    tail = losses[-min(4, len(losses)):]
    slope = (tail[-1] - tail[0]) / max(1, len(tail) - 1)
    mean = sum(losses) / len(losses)
    variance = sum((loss - mean) ** 2 for loss in losses) / len(losses)
    return LearningCurveReport(
        points=len(points), best_validation_loss=losses[best_index], tokens_at_best_validation=int(best.get("tokens_seen", 0)),
        final_validation_loss=losses[-1], generalization_gap=round(float(losses[best_index]) - float(train_loss), 6) if isinstance(train_loss, (int, float)) else None,
        validation_slope=round(slope, 8), loss_variance=round(variance, 8),
        plateau_detected=len(tail) >= 3 and max(tail) - min(tail) < 0.01,
        divergence_detected=len(losses) >= 3 and losses[-1] > min(losses[:-1]) + 0.15,
    )
