from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.checkpoints import CheckpointError, CheckpointStore
from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    Provenance,
    RejectionReason,
    SourceType,
    TokenPacker,
)
from app.foundation.model import UltronModelConfig
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


@pytest.fixture()
def tokenizer(tmp_path):
    corpus = tmp_path / "tokenizer-source.txt"
    corpus.write_text("ULTRON builds local systems with repeatable training data.\n" * 80, encoding="utf-8")
    return UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(vocab_size=320))


def test_complete_checkpoint_resumes_to_same_parameters_and_state(tmp_path, tokenizer):
    import mlx.core as mx

    text = "ULTRON builds local systems with repeatable training data and transparent checkpoints. " * 50
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=8, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32)
    continuous = FoundationTrainer(tmp_path / "continuous")
    config_a = TrainingConfig(run_id="continuous", steps=4, batch_size=1, warmup_steps=2, checkpoint_every=4, validation_every=4)
    result_a = continuous.train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=config_a)

    resumed = FoundationTrainer(tmp_path / "resumed")
    first = TrainingConfig(run_id="resumed", steps=2, batch_size=1, warmup_steps=2, checkpoint_every=2, validation_every=2)
    checkpoint = resumed.train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=first).checkpoint_path
    final = TrainingConfig(run_id="resumed", steps=4, batch_size=1, warmup_steps=2, checkpoint_every=4, validation_every=4)
    result_b = resumed.train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=final, resume_checkpoint=Path(checkpoint))

    weights_a = mx.load(str(Path(result_a.checkpoint_path) / "model.safetensors"))
    weights_b = mx.load(str(Path(result_b.checkpoint_path) / "model.safetensors"))
    assert weights_a.keys() == weights_b.keys()
    assert max(float(mx.max(mx.abs(weights_a[key] - weights_b[key]))) for key in weights_a) < 1e-6
    state_a = CheckpointStore.inspect(Path(result_a.checkpoint_path))
    state_b = CheckpointStore.inspect(Path(result_b.checkpoint_path))
    assert state_a.global_step == state_b.global_step == 4
    assert state_a.tokens_consumed == state_b.tokens_consumed
    optimizer_a = mx.load(str(Path(result_a.checkpoint_path) / "optimizer.safetensors"))
    optimizer_b = mx.load(str(Path(result_b.checkpoint_path) / "optimizer.safetensors"))
    assert max(float(mx.max(mx.abs(optimizer_a[key] - optimizer_b[key]))) for key in optimizer_a) < 1e-6


def test_checkpoint_rejects_corruption(tmp_path, tokenizer):
    text = "A deterministic local sentence supplies enough tokens for a tiny test run. " * 30
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=8, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32)
    result = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=TrainingConfig(run_id="corrupt", steps=1, batch_size=1, checkpoint_every=1, validation_every=1))
    checkpoint = Path(result.checkpoint_path)
    (checkpoint / "scheduler.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointError):
        CheckpointStore.inspect(checkpoint)


def test_corpus_release_filters_and_packs_deterministically(tmp_path, tokenizer):
    documents = [
        CorpusDocument(content=f"Locally authored training document number {number} describes a deterministic local system.", provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier=f"doc-{number}", license="CC0-1.0"))
        for number in range(40)
    ]
    documents += [
        CorpusDocument(content=documents[0].content + "  ", provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier="duplicate")),
        CorpusDocument(content="Please contact alice@example.com with this secret.", provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier="pii")),
    ]
    pipeline = CorpusPipeline(tmp_path, CorpusConfig(validation_percent=25))
    release = pipeline.release("ultron-corpus-test-0.0.1", documents)
    assert release.document_count == 40
    rejections = [json.loads(line) for line in (tmp_path / "corpora" / "releases" / release.release_id / "rejections.jsonl").read_text().splitlines()]
    assert {item["rejection_reason"] for item in rejections} >= {RejectionReason.DUPLICATE.value, RejectionReason.PII_POLICY.value}
    packed = TokenPacker(tmp_path).pack(tmp_path / "corpora" / "releases" / release.release_id, tokenizer, context_length=8, max_sequences_per_shard=2)
    assert packed.shards
    assert packed.document_boundary_policy.startswith("no cross-document")


def test_model_constraints_reject_invalid_attention_shape():
    with pytest.raises(ValueError, match="divisible"):
        UltronModelConfig(vocab_size=320, hidden_size=30, num_heads=4)
