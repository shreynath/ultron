from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DocumentCategory,
    Provenance,
    TokenPacker,
)
from app.foundation.diagnostics import (
    boundary_diagnostics,
    corpus_diagnostics,
    initialization_diagnostics,
    tokenizer_ab_experiment,
)
from app.foundation.experiments import (
    ExperimentRecord,
    ExperimentRegistry,
    parameter_matched_variants,
)
from app.foundation.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStatus,
    TerminationReason,
    degeneration_metrics,
)
from app.foundation.model import UltronModelConfig
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import (
    FoundationTrainer,
    LinearWarmupConstant,
    TrainingConfig,
)


@pytest.fixture()
def phase_i_release(tmp_path):
    def word(number: int) -> str:
        return "term" + chr(97 + number // 26) + chr(97 + number % 26)

    documents = [
        CorpusDocument(content="\n".join(f"aaa{word(index * 50 + segment)} topic{word(index)} segment{word(segment)} detail{word(segment + 52)} contains independent local technical material about matrices and functions." for segment in range(50)), provenance=Provenance(source_type="LOCAL_AUTHORED", source_identifier=f"source-{index}", source_group=f"group-{index}", license="CC0", declared_category=DocumentCategory.MATHEMATICS if index % 2 else DocumentCategory.CODE))
        for index in range(8)
    ]
    return CorpusPipeline(tmp_path, CorpusConfig(split_strategy="source_group_balanced")).release("phase-i-test", documents)


def test_phase_i_corpus_and_boundary_diagnostics(phase_i_release, tmp_path):
    release = tmp_path / "corpora" / "releases" / phase_i_release.release_id
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(json.loads(line)["content"] for line in (release / "train.jsonl").read_text().splitlines()), encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(version="phase-i-test", vocab_size=320))
    report = corpus_diagnostics(release, tokenizer)
    assert report.document_count > 0
    assert report.source_sha256
    packed = TokenPacker(tmp_path).pack(release, tokenizer, context_length=16)
    train_shard = next(shard for shard in packed.shards if shard.split == "train")
    sequences = [json.loads(line)["input_ids"] for line in Path(train_shard.path).read_text(encoding="utf-8").splitlines()]
    assert boundary_diagnostics(sequences, tokenizer, context_length=16).malformed_sequences == 0


def test_phase_i_tokenizer_ab_is_immutable_and_development_only(tmp_path):
    corpus = tmp_path / "development.txt"
    corpus.write_text(("ULTRON tokenizer diagnostics include code: def add(x, y): return x + y.\n" * 40), encoding="utf-8")
    report = tokenizer_ab_experiment(corpus, tmp_path / "tokenizers", experiment_id="ab-v1", vocab_sizes=(320, 512))
    assert len(report.candidates) == 2
    assert report.recommended_vocab_size in {320, 512}
    with pytest.raises(ValueError, match="immutable"):
        tokenizer_ab_experiment(corpus, tmp_path / "tokenizers", experiment_id="ab-v1", vocab_sizes=(320,))


def test_phase_i_initialization_architecture_and_mixture_registry(tmp_path):
    base = UltronModelConfig(vocab_size=320, hidden_size=32, num_heads=4, num_layers=2, intermediate_size=64, max_sequence_length=16)
    assert not initialization_diagnostics(base).pathological
    variants = parameter_matched_variants(base)
    assert set(variants) == {"baseline", "deeper", "wider", "ffn-expanded"}
    registry = ExperimentRegistry(tmp_path)
    record = ExperimentRecord(experiment_id="phase-i-registry", created_at=datetime.now(UTC), hypothesis="A measured tokenizer improves compression.", variables={"vocab": 512}, controls={"corpus": "development"}, model_version="v1", dataset_version="d1", tokenizer_version="t1", training_config={}, seed=1, hardware={})
    registry.record(record)
    assert registry.records()[0].hypothesis == record.hypothesis
    with pytest.raises(ValueError, match="already"):
        registry.record(record)


def test_phase_i_generation_metrics_and_schedule():
    generated = GenerationResult(model_version="v", checkpoint_id="c", tokenizer_version="t", tokenizer_sha256="h", prompt_tokens=1, generated_tokens=5, total_tokens=6, text="", generated_token_ids=(1, 2, 1, 2, 1), status=GenerationStatus.COMPLETED, termination_reason=TerminationReason.MAX_NEW_TOKENS, generation_seconds=1, tokens_per_second=5, mean_token_entropy_bits=2.0)
    metrics = degeneration_metrics(generated)
    assert metrics["repetition_2gram"] > 0
    assert metrics["entropy_bits"] == 2.0
    config = TrainingConfig(steps=10, warmup_steps=2, schedule="linear_warmup_cosine")
    scheduler = LinearWarmupConstant(config)
    values = [scheduler.advance() for _ in range(10)]
    assert values[1] == config.learning_rate
    assert values[-1] < values[1]
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig(temperature=0, top_k=2)


def test_phase_i_training_telemetry_and_early_stopping(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = ("local pretraining telemetry measures stable gradients and EOS behavior.\n" * 100)
    corpus.write_text(text, encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(vocab_size=320))
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    result = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=TrainingConfig(run_id="phase-i-telemetry", model_id="phase-i-telemetry", steps=3, batch_size=1, warmup_steps=0, checkpoint_every=3, validation_every=1, early_stopping_patience=1, early_stopping_min_delta=1.0))
    assert result.gradient_norm_p50 is not None
    assert result.steps_completed <= 3
