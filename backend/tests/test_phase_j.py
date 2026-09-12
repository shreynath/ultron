from __future__ import annotations

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
from app.foundation.phase_j import (
    PhaseJRegistry,
    PhaseJRunRecord,
    phase_j_data_report,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


def _documents() -> list[CorpusDocument]:
    return [
        CorpusDocument(
            content=f"aaa unique{index} alpha{index} beta{index} explains a local programming instruction with mathematical notation and independent details.",
            provenance=Provenance(
                source_type="LOCAL_AUTHORED",
                source_identifier=f"phase-j-source-{index}",
                source_group=f"phase-j-family-{index}",
                license="CC0",
                license_evidence="test fixture",
                declared_category=DocumentCategory.INSTRUCTION if index % 2 else DocumentCategory.CODE,
            ),
        )
        for index in range(12)
    ]


def test_phase_j_data_report_enforces_tokenizer_and_source_isolation(tmp_path):
    release = CorpusPipeline(
        tmp_path,
        CorpusConfig(split_strategy="source_group_balanced", test_percent=20),
    ).release("phase-j-test", _documents())
    source = tmp_path / "tokenizer-source.txt"
    source.write_text("\n".join(document.content for document in _documents()), encoding="utf-8")
    tokenizer = UltronTokenizer.train(source, tmp_path / "tokenizer", TokenizerConfig(version="phase-j-test", vocab_size=320))
    release_dir = tmp_path / "corpora" / "releases" / release.release_id
    tokenized = TokenPacker(tmp_path).pack(release_dir, tokenizer, context_length=8)
    report = phase_j_data_report(release_dir, Path(tokenized.shards[0].path).parent, tokenizer)
    assert report.source_family_isolation["status"] == "PASS"
    assert report.license_coverage == 1.0
    assert report.instruction_percent > 0


def test_phase_j_registry_is_append_only(tmp_path):
    registry = PhaseJRegistry(tmp_path)
    record = PhaseJRunRecord(
        experiment_id="phase-j-test-run",
        hypothesis="A larger corpus changes greedy repetition.",
        model_version="v1",
        tokenizer_version="t1",
        dataset_version="d1",
        training_config={},
        seed=1,
        hardware={},
        token_budget=100,
        actual_tokens=100,
        duration_seconds=1.0,
        throughput_tokens_per_second=100.0,
        peak_memory_bytes=None,
        swap_delta_bytes=None,
        result="COMPLETED",
        conclusion="Test-only record.",
    )
    registry.record(record)
    assert registry.records() == (record,)
    with pytest.raises(ValueError, match="already"):
        registry.record(record)


def test_phase_j_label_shift_loss_and_causal_mask():
    import mlx.core as mx
    from mlx import nn

    logits = mx.array([[[2.0, 0.0], [0.0, 2.0]]])
    targets = mx.array([[0, 1]])
    measured = float(mx.mean(nn.losses.cross_entropy(logits, targets)))
    expected = -2.0 + float(mx.log(mx.exp(mx.array(2.0)) + mx.exp(mx.array(0.0))))
    assert measured == pytest.approx(expected, abs=1e-6)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(4)
    assert float(mask[0, 1]) < -1e8
    assert float(mask[3, 0]) == 0.0
