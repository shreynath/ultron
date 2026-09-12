from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DocumentCategory,
    PackedShardSequenceDataset,
    Provenance,
    TokenPacker,
)
from app.foundation.diagnostics import boundary_diagnostics
from app.foundation.model import UltronModelConfig
from app.foundation.phase_k import (
    DistributedTrainingConfig,
    PhaseKExperimentRecord,
    PhaseKRegistry,
    PhaseKStatus,
    SwapClass,
    classify_swap,
    corpus_balance_report,
    derive_scaled_config,
    parameter_breakdown,
    phase_j_baseline_record,
    phase_k_preflight,
)
from app.foundation.scaling import ResourceRisk, resource_preflight
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


def _documents() -> list[CorpusDocument]:
    def word(number: int) -> str:
        return "term" + chr(97 + number // 26) + chr(97 + number % 26)

    categories = (DocumentCategory.CODE, DocumentCategory.MATHEMATICS, DocumentCategory.SCIENCE)
    return [
        CorpusDocument(
            content="\n".join(
                f"{word(index * 50 + segment)} {word(segment + 300)} explains independent {categories[index % 3].value.lower()} material with local methods and reproducible details."
                for segment in range(50)
            ),
            provenance=Provenance(
                source_type="LOCAL_AUTHORED",
                source_identifier=f"phase-k-source-{index}",
                source_group=f"phase-k-family-{index}",
                license="CC0",
                license_evidence="fixture",
                declared_category=categories[index % 3],
            ),
        )
        for index in range(15)
    ]


def test_phase_k_instantiated_parameter_count_matches_exact_formula():
    config = UltronModelConfig(vocab_size=320, hidden_size=32, num_heads=4, num_layers=2, intermediate_size=96, max_sequence_length=16)
    measured = parameter_breakdown(config)
    assert measured.method == "INSTANTIATED_MLX_PARAMETER_TREE"
    assert measured.total_parameters == config.parameter_estimate()
    assert measured.total_parameters == sum((measured.embedding_parameters, measured.attention_parameters, measured.ffn_parameters, measured.normalization_parameters, measured.output_head_parameters))


def test_phase_k_lazy_packed_dataset_and_balance_report(tmp_path):
    release = CorpusPipeline(tmp_path, CorpusConfig(split_strategy="source_group_balanced", test_percent=20)).release("phase-k-release", _documents())
    source = tmp_path / "source.txt"
    source.write_text("\n".join(document.content for document in _documents()), encoding="utf-8")
    tokenizer = UltronTokenizer.train(source, tmp_path / "tokenizer", TokenizerConfig(version="phase-k-test", vocab_size=320))
    release_dir = tmp_path / "corpora" / "releases" / release.release_id
    tokenized = TokenPacker(tmp_path).pack(release_dir, tokenizer, context_length=8)
    train_shards = [Path(shard.path) for shard in tokenized.shards if shard.split == "train"]
    dataset = PackedShardSequenceDataset(train_shards, expected_width=9)
    assert len(dataset) > 0
    assert len(dataset[0]) == 9
    report = corpus_balance_report(release_dir, Path(tokenized.shards[0].path).parent, tokenizer)
    assert report.leakage_checks["status"] == "PASS"
    assert report.license_coverage == 1.0
    assert report.source_families == release.document_count
    rows = [dataset[index] for index in range(len(dataset))]
    boundary = boundary_diagnostics(rows, tokenizer, context_length=8)
    assert boundary.sequences_with_eos > 0
    assert boundary.conclusion.startswith("PASS")


def test_phase_k_preflight_registry_and_infeasible_memory_classification(tmp_path):
    config = derive_scaled_config(5_000_000, vocab_size=320, context_length=16, version="phase-k-test")
    preflight = phase_k_preflight(config, batch_size=1, sequence_length=16, target_tokens=1000, checkpoint_count=1)
    assert preflight.parameter_breakdown.method == "EXACT_ARCHITECTURE_FORMULA_NOT_INSTANTIATED"
    assert preflight.approximate_training_flops is not None
    assert preflight.status is PhaseKStatus.PREFLIGHTED
    assert resource_preflight(config, batch_size=1, sequence_length=16, memory_bytes=1).risk is ResourceRisk.UNSUPPORTED
    registry = PhaseKRegistry(tmp_path)
    record = PhaseKExperimentRecord(
        experiment_id="phase-k-fixture",
        experiment_class="infrastructure",
        hypothesis="A preflight records a local pilot before training.",
        status=PhaseKStatus.PREFLIGHTED,
        model_scale="fixture",
        model=config.model_dump(mode="json"),
        parameter_count=preflight.parameter_breakdown.total_parameters,
        token_budget=1000,
        corpus_release="fixture",
        corpus_hash="a" * 64,
        tokenizer_version="fixture",
        tokenizer_sha256="b" * 64,
        training_config={},
        seed=1,
        hardware={},
        conclusion="Preflight only.",
    )
    registry.record(record)
    assert registry.records() == (record,)
    with pytest.raises(ValueError, match="already"):
        registry.record(record)
    assert phase_j_baseline_record()["assistant_promotion"] == "NOT_PROMOTED"
    assert classify_swap(0) is SwapClass.NO_SWAP
    assert classify_swap(512 * 1024 * 1024) is SwapClass.MODERATE_SWAP
    assert DistributedTrainingConfig().distributed_mode == "UNIMPLEMENTED"
