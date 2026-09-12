"""Versioned, local evaluation and explicit candidate-promotion decisions."""
from __future__ import annotations

import hashlib
import math
import shutil
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import CheckpointManifest, CheckpointStore
from app.foundation.data import DatasetRelease, ProcessedDocument
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.tokenizer import UltronTokenizer


class EvaluationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class PromotionDecision(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    version: str
    validation_text_path: str
    validation_text_sha256: str
    max_tokens: int = Field(default=512, ge=8)


class EvaluationDatasetRelease(BaseModel):
    """A sealed copy of a corpus test split used only by evaluators."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    dataset_release_id: str
    dataset_hash: str
    tokenizer_version: str
    tokenizer_sha256: str
    source_manifest_sha256: str
    split_manifest: str
    created_at: datetime
    preprocessing_version: str


def create_evaluation_release(
    target: Path, *, release_id: str, source_release_dir: Path, tokenizer: UltronTokenizer, split: str = "test",
) -> EvaluationDatasetRelease:
    """Copy an immutable named split; callers reserve ``test`` for sealed use."""
    if target.exists():
        raise ValueError("Evaluation dataset releases are immutable.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Evaluation split must be train, validation, or test.")
    source_manifest = DatasetRelease.model_validate_json((source_release_dir / "manifest.json").read_text(encoding="utf-8"))
    source_test = source_release_dir / f"{split}.jsonl"
    if not source_test.exists():
        raise ValueError(f"Corpus release has no {split} split for evaluation release creation.")
    target.mkdir(parents=True)
    copied = target / "test.jsonl"
    shutil.copyfile(source_test, copied)
    dataset_hash = hashlib.sha256(copied.read_bytes()).hexdigest()
    record = EvaluationDatasetRelease(release_id=release_id, dataset_release_id=source_manifest.release_id, dataset_hash=dataset_hash, tokenizer_version=tokenizer.version, tokenizer_sha256=tokenizer.sha256, source_manifest_sha256=source_manifest.manifest_sha256 or "UNKNOWN", split_manifest=f"{split}: {source_manifest.split_method}", created_at=datetime.now(UTC), preprocessing_version=source_manifest.processing_version)
    (target / "manifest.json").write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    suite_version: str
    checkpoint_id: str
    model_id: str
    dataset_release_id: str
    created_at: datetime
    validation_tokens: int
    validation_loss: float
    perplexity: float
    status: EvaluationStatus
    contamination_check: str


class EvaluationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_checkpoint: str
    candidate_checkpoint: str
    baseline_loss: float
    candidate_loss: float
    loss_delta: float
    status: EvaluationStatus
    rationale: str


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_loss_regression: float = Field(default=0.0, ge=0)
    require_improvement: bool = True


def suite_from_text(suite_id: str, version: str, validation_text_path: Path, *, max_tokens: int = 512) -> EvaluationSuite:
    return EvaluationSuite(suite_id=suite_id, version=version, validation_text_path=str(validation_text_path), validation_text_sha256=hashlib.sha256(validation_text_path.read_bytes()).hexdigest(), max_tokens=max_tokens)


class FoundationEvaluator:
    """Scores a frozen local checkpoint only; it has no training or promotion side effects."""

    def evaluate(self, checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig, suite: EvaluationSuite) -> EvaluationReport:
        import mlx.core as mx
        from mlx import nn

        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("Evaluation requires a complete checkpoint manifest.")
        text_path = Path(suite.validation_text_path)
        if hashlib.sha256(text_path.read_bytes()).hexdigest() != suite.validation_text_sha256:
            raise ValueError("Evaluation fixture hash changed; create a new suite version.")
        ids = tokenizer.encode(text_path.read_text(encoding="utf-8"))[: min(suite.max_tokens, model_config.max_sequence_length) + 1]
        if len(ids) < 2:
            raise ValueError("Evaluation fixture needs at least two tokens.")
        model = build_model(model_config)
        model.load_weights(str(checkpoint / "model.safetensors"))
        batch = mx.array([ids[:-1]])
        loss = float(mx.mean(nn.losses.cross_entropy(model(batch), mx.array([ids[1:]]))))
        return EvaluationReport(suite_id=suite.suite_id, suite_version=suite.version, checkpoint_id=manifest.checkpoint_id, model_id=manifest.model_id, dataset_release_id=manifest.dataset_release_id, created_at=datetime.now(UTC), validation_tokens=len(ids) - 1, validation_loss=loss, perplexity=math.exp(min(loss, 20.0)), status=EvaluationStatus.PASSED, contamination_check="UNKNOWN: text-hash separation is recorded, but semantic contamination is not established.")

    def evaluate_sealed_release(
        self, checkpoint: Path, tokenizer: UltronTokenizer, model_config: UltronModelConfig,
        evaluation_dir: Path, *, suite_id: str, suite_version: str, max_sequences: int = 128,
    ) -> EvaluationReport:
        """Measure deterministic windows from an immutable copied test split."""
        import mlx.core as mx
        from mlx import nn

        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("Evaluation requires a complete checkpoint manifest.")
        release = EvaluationDatasetRelease.model_validate_json((evaluation_dir / "manifest.json").read_text(encoding="utf-8"))
        test_path = evaluation_dir / "test.jsonl"
        if hashlib.sha256(test_path.read_bytes()).hexdigest() != release.dataset_hash:
            raise ValueError("Sealed evaluation split hash changed.")
        sequences: list[list[int]] = []
        for line in test_path.read_text(encoding="utf-8").splitlines():
            document = ProcessedDocument.model_validate_json(line)
            ids = tokenizer.encode(document.content) + [tokenizer.processor.eos_id()]
            for offset in range(0, len(ids) - model_config.max_sequence_length, model_config.max_sequence_length):
                chunk = ids[offset:offset + model_config.max_sequence_length + 1]
                if len(chunk) == model_config.max_sequence_length + 1:
                    sequences.append(chunk)
                if len(sequences) >= max_sequences:
                    break
            if len(sequences) >= max_sequences:
                break
        if not sequences:
            raise ValueError("Sealed evaluation split has no complete context window.")
        model = build_model(model_config)
        model.load_weights(str(checkpoint / "model.safetensors"))
        losses = [float(mx.mean(nn.losses.cross_entropy(model(mx.array([item[:-1]])), mx.array([item[1:]])))) for item in sequences]
        loss = sum(losses) / len(losses)
        source_split = release.split_manifest.split(":", 1)[0]
        contamination = "SOURCE_SPLIT_SEALED: immutable copied test split; semantic contamination remains unproven." if source_split == "test" else f"DEVELOPMENT_{source_split.upper()}_SPLIT: immutable copied probe; not a sealed test."
        return EvaluationReport(suite_id=suite_id, suite_version=suite_version, checkpoint_id=manifest.checkpoint_id, model_id=manifest.model_id, dataset_release_id=release.dataset_release_id, created_at=datetime.now(UTC), validation_tokens=len(sequences) * model_config.max_sequence_length, validation_loss=loss, perplexity=math.exp(min(loss, 20.0)), status=EvaluationStatus.PASSED, contamination_check=contamination)

    @staticmethod
    def compare(baseline: EvaluationReport, candidate: EvaluationReport) -> EvaluationComparison:
        if (baseline.suite_id, baseline.suite_version) != (candidate.suite_id, candidate.suite_version):
            raise ValueError("Only reports from the same suite version are comparable.")
        delta = candidate.validation_loss - baseline.validation_loss
        return EvaluationComparison(baseline_checkpoint=baseline.checkpoint_id, candidate_checkpoint=candidate.checkpoint_id, baseline_loss=baseline.validation_loss, candidate_loss=candidate.validation_loss, loss_delta=delta, status=EvaluationStatus.PASSED, rationale="Measured held-out token loss comparison; lower is better.")

    @staticmethod
    def promotion_decision(comparison: EvaluationComparison, candidate_checkpoint: Path, policy: PromotionPolicy) -> tuple[PromotionDecision, str]:
        manifest = CheckpointStore.inspect(candidate_checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            return PromotionDecision.REJECT, "Candidate checkpoint is incomplete."
        lineage = manifest.lineage
        if lineage.get("initialization") != "RANDOM" or lineage.get("pretrained_weights") != "NONE":
            return PromotionDecision.REJECT, "Candidate lineage is not independently from scratch."
        if comparison.loss_delta > policy.max_loss_regression:
            return PromotionDecision.REJECT, "Candidate regressed beyond the configured loss limit."
        if policy.require_improvement and comparison.loss_delta >= 0:
            return PromotionDecision.REJECT, "Candidate did not improve the measured suite."
        return PromotionDecision.INSUFFICIENT_EVIDENCE, "Metric policy passes, but promotion still requires capability, safety, and regression suites."

    @staticmethod
    def write_report(path: Path, report: EvaluationReport | EvaluationComparison) -> None:
        if path.exists():
            raise ValueError("Evaluation reports are immutable.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
