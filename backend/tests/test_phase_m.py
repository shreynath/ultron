from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.brain.acquisition import AcquisitionStore, SourceQuality, SourceRegistryEntry
from app.foundation.phase_m import (
    PhaseMDisposition,
    build_phase_m_release,
    phase_m_branch_decision,
    phase_m_rung_plan,
    phase_m_status,
    source_decisions,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


def _text(index: int) -> str:
    def letters(value: int) -> str:
        output = ""
        while True:
            output = chr(ord("a") + value % 26) + output
            value = value // 26 - 1
            if value < 0:
                return output

    words = [f"lexeme{letters(index * 200 + offset)}" for offset in range(100)]
    return " ".join(words) + ".\n\n" + " ".join(reversed(words)) + "."


def _register(root: Path, source_id: str, content: str, *, group: str, duplicate_of: str | None = None) -> None:
    digest = duplicate_of or hashlib.sha256(content.encode("utf-8")).hexdigest()
    cache = root / "acquisition" / "cache" / f"{digest}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        cache.write_text(content, encoding="utf-8")
    AcquisitionStore(root).register_source(SourceRegistryEntry(
        source_id=source_id, canonical_url=f"https://example.test/{source_id}", title=source_id,
        publisher="fixture", source_type="primary", category="GENERAL", license="PUBLIC-DOMAIN-US",
        license_evidence="https://example.test/license", retrieved_at=datetime.now(UTC), published_at=None,
        content_hash=digest, retrieval_status="RETRIEVED", parser_version="fixture", acquisition_version="fixture",
        provenance={"cache_path": str(cache)},
        quality=SourceQuality(authority=1, provenance_completeness=1, structural_quality=1, machine_readability=1, stability=1, licensing_clarity=1, duplication_risk=0, noise_risk=0, content_density=1, score=1),
        source_concentration_group=group,
    ))


def test_phase_m_release_is_registry_backed_split_complete_and_audited(tmp_path: Path) -> None:
    root = tmp_path / "ultron"
    corpus = tmp_path / "tokenizer.txt"
    corpus.write_text("\n".join(_text(index) for index in range(15)), encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "ultron-phase-i-v2", TokenizerConfig(version="0.5.0-phase-i-v2", vocab_size=320))
    for index in range(15):
        _register(root, f"source-{index}", _text(index), group=f"group-{index}")
    _register(root, "source-duplicate", _text(0), group="duplicate", duplicate_of=hashlib.sha256(_text(0).encode("utf-8")).hexdigest())

    decisions = source_decisions(root)
    assert sum(item.status.value == "INCLUDED" for item in decisions) == 15
    assert any(item.reason == "EXACT_DUPLICATE_SOURCE_CONTENT" for item in decisions)

    result = build_phase_m_release(root, release_id="ultron-corpus-test-phase-m-development-5m", tokenizer=tokenizer, target_tokens=1, context_length=512)
    assert result.accepted
    assert result.audit.status == "PASS"
    assert result.audit.documents_by_split["train"] > 0
    assert result.audit.documents_by_split["validation"] > 0
    assert result.audit.documents_by_split["test"] > 0
    assert result.audit.leakage_checks["status"] == "PASS"
    assert result.audit.special_token_counts["eos_appended"] == 15
    assert (root / "corpora" / "source-manifests" / "ultron-corpus-test-phase-m-development-5m.json").exists()


def test_phase_m_status_and_rungs_do_not_claim_unexecuted_training(tmp_path: Path) -> None:
    status = phase_m_status(tmp_path)
    assert status.disposition is PhaseMDisposition.ACTIVE
    assert status.latest_experiment is None
    rungs = phase_m_rung_plan(development_release_id="development", development_corpus_tokens=5_200_000)
    assert [rung.update_token_target for rung in rungs] == [1_000_000, 5_000_000]
    assert rungs[1].initialization == "CONTINUATION_REQUIRED_BEFORE_EXECUTION"
    assert phase_m_branch_decision(
        functional_emergence_level=0, capability_score=0,
        generation={"immediate_eos_rate": 1.0}, learning_signal=True,
    ).action == "STOP_AND_INVESTIGATE"
    assert phase_m_branch_decision(
        functional_emergence_level=1, capability_score=1,
        generation={"distinct_2": 0.4, "repeated_bigram_rate": 0.1}, learning_signal=True,
    ).action == "PROMOTE"


def test_phase_m_status_uses_recorded_terminal_disposition(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "phase-m-final-disposition-v1.json").write_text(json.dumps({
        "disposition": "PHASE_M_CAPACITY_LIMIT_IDENTIFIED",
        "run_id": "phase-m-run",
        "functional_emergence_level": 0,
        "capability_score": "0/6",
        "scaling_recommendation": "Stop.",
        "next_action": "Preserve evidence.",
    }), encoding="utf-8")
    status = phase_m_status(tmp_path)
    assert status.disposition is PhaseMDisposition.CAPACITY_NEXT
    assert status.latest_experiment == "phase-m-run"
    assert status.functional_emergence_level == 0
