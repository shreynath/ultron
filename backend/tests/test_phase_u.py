from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.checkpoints import CheckpointManifest, CheckpointStore
from app.foundation.data import PackedShardSequenceDataset
from app.foundation.model import UltronModelConfig
from app.foundation.phase_r import LearningAttribution, LearningEvidenceLevel
from app.foundation.phase_t import register_architecture
from app.foundation.phase_u import (
    PHASE_U_MINIMUM_EFFECTIVE_TOKENS,
    CorpusAccounting,
    LearningVerdict,
    NativeMixtureSequenceDataset,
    PhaseUEngine,
    SourceProvenance,
    account_corpus,
    assess_resources,
    create_experiment,
    create_mixture_configuration,
    learning_verdict,
    training_budget,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


def _accounting(**updates: object) -> CorpusAccounting:
    values: dict[str, object] = {
        "dataset_release_id": "phase-u-owned-v1",
        "dataset_sha256": "a" * 64,
        "raw_documents": 120,
        "normalized_documents": 115,
        "unique_documents": 100,
        "unique_records": 100,
        "raw_tokens": 25_000_000,
        "deduplicated_tokens": 22_000_000,
        "train_tokens": 21_000_000,
        "validation_tokens": 500_000,
        "test_tokens": 500_000,
        "teacher_generated_tokens": 0,
        "source_derived_tokens": 22_000_000,
        "synthetic_tokens": 0,
        "repeated_tokens": 100_000,
        "contaminated_tokens": 50_000,
        "quarantined_tokens": 50_000,
        "duplicate_rate": 0.04,
        "contamination_rate": 0.002,
    }
    values.update(updates)
    return account_corpus(**values)


def test_phase_u_budget_requires_real_effective_10000x_tokens() -> None:
    accounting = _accounting()
    assert accounting.effective_unique_training_tokens == 20_800_000
    assert accounting.meets_phase_u_minimum
    budget = training_budget(accounting=accounting, batch_size=2, context_length=64)
    assert budget.requested_tokens == PHASE_U_MINIMUM_EFFECTIVE_TOKENS
    assert budget.optimizer_steps == 160_000
    assert budget.multiplier_vs_phase_t == 10_000
    with pytest.raises(ValueError, match="budget exceeds"):
        training_budget(
            accounting=_accounting(train_tokens=20_000_000),
            batch_size=2,
            context_length=64,
        )


def test_effective_token_accounting_rejects_repeated_or_contaminated_inflation() -> (
    None
):
    invalid = _accounting().model_dump()
    invalid.update(
        {
            "effective_unique_training_tokens": 21_000_000,
            "training_token_multiplier_vs_phase_t": 21_000_000 / 2_048,
        }
    )
    with pytest.raises(ValueError, match="exclude repeated"):
        CorpusAccounting(**invalid)


def test_source_provenance_requires_license_evidence_and_is_immutable(
    tmp_path: Path,
) -> None:
    source = SourceProvenance(
        source_id="cpython-3.13",
        source_uri="https://github.com/python/cpython",
        license="PSF-2.0",
        retrieval_timestamp=datetime.now(UTC),
        content_hash="b" * 64,
        license_evidence="https://github.com/python/cpython/blob/main/LICENSE",
        domain="programming",
        document_type="permissively licensed source and documentation",
    )
    engine = PhaseUEngine(tmp_path)
    assert engine.record_source(source) == source
    with pytest.raises(ValueError, match="artifact already exists"):
        engine.record_source(source.model_copy(update={"domain": "mathematics"}))
    with pytest.raises(ValueError, match="explicit license"):
        invalid_source = source.model_dump()
        invalid_source.update({"license": "UNSPECIFIED", "source_id": "invalid"})
        SourceProvenance(**invalid_source)


def test_native_mixture_dataset_selects_real_packed_sequences_and_checks_exposure(
    tmp_path: Path,
) -> None:
    def shard(name: str, records: list[list[int]]) -> PackedShardSequenceDataset:
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"input_ids": record, "loss_mask": [1, 1, 1]}) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return PackedShardSequenceDataset(
            [path], expected_width=4, include_loss_mask=True
        )

    mixture = create_mixture_configuration(
        "phase-u-mixture",
        release_weights={"natural": 0.70, "code": 0.30},
        seed=8,
        tolerance=0.05,
    )
    dataset = NativeMixtureSequenceDataset(
        datasets={
            "natural": shard("natural", [[1, 2, 3, 4], [5, 6, 7, 8]]),
            "code": shard("code", [[9, 10, 11, 12]]),
        },
        configuration=mixture,
        samples=10_000,
        tokens_per_sample=3,
    )
    assert dataset[0] == dataset[0]
    assert len(dataset) == 10_000
    assert not dataset.exposure.materially_diverged
    assert (
        dataset.repeated_samples > 0
    )  # the tiny fixture intentionally exhausts releases
    assert dataset.sampler_state(23)["position"] == 23


def test_model_registry_experiment_resource_gate_and_honest_verdict(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "tokenizer.txt"
    corpus.write_text("licensed independently authored technical corpus " * 80)
    tokenizer = UltronTokenizer.train(
        corpus, tmp_path / "tokenizer", TokenizerConfig(version="u", vocab_size=320)
    )
    model = UltronModelConfig(
        model_name="ultron-phase-u-1m",
        version="v1",
        vocab_size=tokenizer.vocab_size,
        hidden_size=160,
        num_layers=4,
        num_heads=4,
        intermediate_size=640,
        max_sequence_length=64,
    )
    architecture = register_architecture(
        model_config=model, tokenizer=tokenizer, instantiate=False
    )
    baseline = FoundationTrainer(tmp_path).initialize_from_scratch(
        model_config=model,
        tokenizer=tokenizer,
        config=TrainingConfig(run_id="phase-u-zero", model_id=model.model_name),
    )
    manifest = CheckpointStore.inspect(baseline)
    assert isinstance(manifest, CheckpointManifest)
    assert manifest.global_step == 0
    budget = training_budget(accounting=_accounting(), batch_size=2, context_length=64)
    mixture = create_mixture_configuration(
        "one-release", release_weights={"phase-u-owned-v1": 1.0}, seed=17
    )
    experiment = create_experiment(
        experiment_id="phase-u-e2e",
        architecture=architecture,
        tokenizer_id=tokenizer.model_path.stem,
        dataset_release_id="phase-u-owned-v1",
        dataset_sha256="a" * 64,
        mixture=mixture,
        training_config_sha256=hashlib.sha256(b"phase-u").hexdigest(),
        initialization_seed=17,
        budget=budget,
    )
    assert experiment.parameter_count == 1_342_880
    assert experiment.checkpoint_steps[-1] == 160_000
    resources = assess_resources(
        tokens_per_second=3_000,
        tokens_target=budget.requested_tokens,
        maximum_wall_seconds=600,
        disk_usage_bytes=12,
        peak_memory_bytes=34,
    )
    assert resources.status == "PHASE_U_RESOURCE_BLOCKED"
    attribution = LearningAttribution(
        attribution_id="u",
        experiment_id="u",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        weights_changed=True,
        training_loss_delta=-1,
        held_out_loss_delta=-1,
        target_capability_delta=1,
        generalization_delta=1,
        regression_rate=0,
        evidence=LearningEvidenceLevel.GENERALIZATION_EVIDENCE,
        rationale="fixture",
    )
    assert (
        learning_verdict(attribution=attribution, resources=resources)
        is LearningVerdict.PHASE_U_RESOURCE_BLOCKED
    )
