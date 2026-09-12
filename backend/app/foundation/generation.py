"""Bounded local decoding for ULTRON's own from-scratch checkpoints.

The current decoder recomputes the active prefix for each token because the
underlying MLX attention block has no validated incremental-KV interface.
Correctness and lineage checks take precedence over cache optimization.
"""
from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from threading import Event

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.foundation.checkpoints import (
    CheckpointManifest,
    CheckpointStore,
    canonical_hash,
)
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.tokenizer import UltronTokenizer


class GenerationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TerminationReason(str, Enum):
    EOS = "EOS"
    MAX_NEW_TOKENS = "MAX_NEW_TOKENS"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_new_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0, le=5)
    top_k: int | None = Field(default=None, ge=1, le=320)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int = 17
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    stop_token_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_decoding(self) -> GenerationConfig:
        if self.temperature == 0 and (self.top_k is not None or self.top_p is not None):
            raise ValueError("top_k/top_p require temperature sampling; greedy decoding has temperature=0.")
        return self


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: str
    checkpoint_id: str
    tokenizer_version: str
    tokenizer_sha256: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    text: str
    generated_token_ids: tuple[int, ...]
    status: GenerationStatus
    termination_reason: TerminationReason
    generation_seconds: float
    tokens_per_second: float
    peak_mlx_bytes: int | None = None
    mean_token_entropy_bits: float | None = None
    mean_top_token_probability: float | None = None
    mean_previous_token_probability: float | None = None
    cache_implementation: str = "NONE_FULL_PREFIX_RECOMPUTE"
    error: str | None = None


class UltronGenerator:
    """A local-only decoder.  It cannot load third-party model identities."""

    def __init__(self, checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig) -> None:
        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("Generation requires a complete resumable ULTRON checkpoint.")
        if manifest.lineage.get("origin") != "ULTRON_FROM_SCRATCH" or manifest.lineage.get("pretrained_weights") != "NONE":
            raise ValueError("Generation rejects non-ULTRON or pretrained checkpoint lineage.")
        if manifest.tokenizer_sha256 != tokenizer.sha256 or manifest.tokenizer_version != tokenizer.version:
            raise ValueError("Checkpoint tokenizer does not match the supplied ULTRON tokenizer.")
        if manifest.architecture_config_sha256 != canonical_hash(model_config.model_dump(mode="json")):
            raise ValueError("Checkpoint architecture does not match the supplied model configuration.")
        self.checkpoint, self.tokenizer, self.model_config, self.manifest = checkpoint, tokenizer, model_config, manifest
        self.model = build_model(model_config)
        self.model.load_weights(str(checkpoint / "model.safetensors"))

    def generate(self, prompt: str, config: GenerationConfig | None = None, *, cancel: Event | None = None) -> GenerationResult:
        import mlx.core as mx

        config = config or GenerationConfig()
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.processor.bos_id()]
        if len(prompt_ids) > self.model_config.max_sequence_length:
            raise ValueError("Prompt exceeds model context; explicit caller-side truncation is required.")
        generated: list[int] = []
        entropies: list[float] = []
        top_probabilities: list[float] = []
        previous_probabilities: list[float] = []
        started = time.perf_counter()
        rng = np.random.default_rng(config.seed)
        reason, status, error = TerminationReason.MAX_NEW_TOKENS, GenerationStatus.COMPLETED, None
        mx.reset_peak_memory()
        try:
            for _ in range(config.max_new_tokens):
                if cancel is not None and cancel.is_set():
                    reason, status = TerminationReason.CANCELLED, GenerationStatus.CANCELLED
                    break
                if time.perf_counter() - started >= config.timeout_seconds:
                    reason, status = TerminationReason.ERROR, GenerationStatus.FAILED
                    error = "Generation timeout exceeded."
                    break
                active = [*prompt_ids, *generated]
                if len(active) >= self.model_config.max_sequence_length:
                    reason = TerminationReason.CONTEXT_LIMIT
                    break
                logits = np.asarray(self.model(mx.array([active]))[:, -1, :]).reshape(-1)
                probabilities = self._probabilities(logits)
                entropies.append(float(-(probabilities * np.log2(np.maximum(probabilities, 1e-30))).sum()))
                top_probabilities.append(float(probabilities.max()))
                if generated:
                    previous_probabilities.append(float(probabilities[generated[-1]]))
                next_token = self._select(logits, config, rng)
                generated.append(next_token)
                if next_token == self.tokenizer.processor.eos_id() or next_token in config.stop_token_ids:
                    reason = TerminationReason.EOS
                    break
        except (RuntimeError, ValueError, OSError) as exc:
            reason, status, error = TerminationReason.ERROR, GenerationStatus.FAILED, f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        return GenerationResult(model_version=self.model_config.version, checkpoint_id=self.manifest.checkpoint_id, tokenizer_version=self.tokenizer.version, tokenizer_sha256=self.tokenizer.sha256, prompt_tokens=len(prompt_ids), generated_tokens=len(generated), total_tokens=len(prompt_ids) + len(generated), text=self.tokenizer.decode(generated), generated_token_ids=tuple(generated), status=status, termination_reason=reason, generation_seconds=elapsed, tokens_per_second=len(generated) / elapsed if elapsed else 0.0, peak_mlx_bytes=int(mx.get_peak_memory()), mean_token_entropy_bits=float(np.mean(entropies)) if entropies else None, mean_top_token_probability=float(np.mean(top_probabilities)) if top_probabilities else None, mean_previous_token_probability=float(np.mean(previous_probabilities)) if previous_probabilities else None, error=error)

    @staticmethod
    def _probabilities(logits: np.ndarray) -> np.ndarray:
        probabilities = np.exp(logits - np.max(logits))
        return probabilities / probabilities.sum()

    @staticmethod
    def _select(logits: np.ndarray, config: GenerationConfig, rng: np.random.Generator) -> int:
        if config.temperature == 0:
            return int(np.argmax(logits))
        scaled = logits / config.temperature
        if config.top_k is not None:
            permitted = np.argpartition(scaled, -config.top_k)[-config.top_k:]
            mask = np.full_like(scaled, -np.inf)
            mask[permitted] = scaled[permitted]
            scaled = mask
        probabilities = UltronGenerator._probabilities(scaled)
        if config.top_p is not None:
            ordered = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[ordered])
            keep = ordered[cumulative <= config.top_p]
            if not len(keep):
                keep = ordered[:1]
            filtered = np.zeros_like(probabilities)
            filtered[keep] = probabilities[keep]
            probabilities = filtered / filtered.sum()
        return int(rng.choice(len(probabilities), p=probabilities))


def degeneration_metrics(result: GenerationResult) -> dict[str, float | int | bool | None]:
    """Formal raw-generation metrics; no decoding repair is applied.

    ``repetition_ngram_rate`` is ``1 - unique_ngrams / total_ngrams`` and
    ``distinct_n`` is its complement.  Entropy is the mean next-token softmax
    entropy gathered before every decoded token.
    """
    ids = result.generated_token_ids
    def ngram(n: int) -> tuple[float, float]:
        grams = [tuple(ids[index:index + n]) for index in range(max(0, len(ids) - n + 1))]
        if not grams:
            return 0.0, 0.0
        distinct = len(set(grams)) / len(grams)
        return round(1.0 - distinct, 6), round(distinct, 6)

    repeat2, distinct2 = ngram(2)
    repeat3, distinct3 = ngram(3)
    repeat4, _ = ngram(4)
    distinct1 = len(set(ids)) / len(ids) if ids else 0.0
    return {
        "unique_token_ratio": round(distinct1, 6),
        "repeated_unigram_rate": round(1.0 - distinct1, 6),
        "repeated_bigram_rate": repeat2,
        "repetition_2gram": repeat2,
        "repetition_3gram": repeat3,
        "repetition_4gram": repeat4,
        "distinct_1": round(distinct1, 6),
        "distinct_2": distinct2,
        "distinct_3": distinct3,
        "average_length": result.generated_tokens,
        "entropy_bits": result.mean_token_entropy_bits,
        "eos_rate": float(result.termination_reason is TerminationReason.EOS),
        "immediate_eos": bool(result.termination_reason is TerminationReason.EOS and result.generated_tokens == 1),
        "entropy_available": result.mean_token_entropy_bits is not None,
    }
