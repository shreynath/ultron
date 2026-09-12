from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("mlx")

from app.foundation.data import PackedShardSequenceDataset
from app.foundation.phase_l import (
    FunctionalFailure,
    OneBFunctionalStatus,
    PhaseLTrainingRecipeRecord,
    PhaseLTrainingRecipeRegistry,
    SustainedRunClassification,
    aggregate_generation_health,
    one_b_functional_gate,
    phase_l_experiment_report,
    summarize_sustained_run,
)
from app.foundation.phase_l_diagnostics import (
    compare_branches,
    functional_level,
    objective_sanity,
)
from app.foundation.runs import (
    CorpusTier,
    DevelopmentStatus,
    EvidenceClass,
    ExperimentTier,
    FormalGateStatus,
    TrainingRun,
)
from app.foundation.scaling import Feasibility, ScalingResult
from app.foundation.training import TrainingConfig


def test_phase_l_sustained_summary_requires_one_hundred_real_steps():
    result = ScalingResult(experiment_id="phase-l-fixture", created_at=datetime.now(UTC), model={}, training={"steps": 100, "loss": 1.0, "loss_min": 0.5, "loss_max": 1.5, "windows": [{"tokens_per_second": 10.0, "swap_after_bytes": 10}, {"tokens_per_second": 8.0, "swap_after_bytes": 10}]}, memory={"swap_used_before_bytes": 0, "swap_peak_bytes": 10}, hardware={}, status=Feasibility.FEASIBLE)
    summary = summarize_sustained_run(result)
    assert summary.training_status == "SUSTAINED_ESTABLISHED"
    assert summary.steady_state_tokens_per_second == 8.0
    assert summary.swap_behavior == "STABILIZING_OR_OSCILLATING"
    assert summary.classification is SustainedRunClassification.SUSTAINED_EXECUTION_WITH_SWAP


def test_development_registry_record_cannot_claim_formal_evidence():
    run = TrainingRun(
        run_id="phase-l-development-fixture",
        model_id="fixture",
        model_version="fixture",
        dataset_release="development-release",
        tokenizer_version="fixture",
        training_config={},
        corpus_id="development-release",
        corpus_tier=CorpusTier.DEVELOPMENT,
        token_count=1_000_000,
        experiment_tier=ExperimentTier.DEVELOPMENT,
        evidence_class=EvidenceClass.DEVELOPMENT_SCALE,
        formal_gate_status=FormalGateStatus.SIZE_FAILURE,
        development_status=DevelopmentStatus.READY,
        model_parameters=50_000_000,
        training_tokens=102_400,
        validation_tokens=100_000,
        test_tokens=100_000,
    )
    report = phase_l_experiment_report(run)
    assert report["evidence_class"] == "DEVELOPMENT_SCALE"
    assert "not a formal" in report["claim_boundary"]
    with pytest.raises(ValueError, match="READY 10M"):
        TrainingConfig(
            corpus_tier=CorpusTier.FORMAL,
            experiment_tier=ExperimentTier.FORMAL,
            evidence_class=EvidenceClass.FORMAL_PHASE_L,
            formal_gate_status=FormalGateStatus.SIZE_FAILURE,
            corpus_token_count=10_000_000,
        )


def test_phase_l_recipe_registry_is_append_only_and_one_b_gate_fails_closed(tmp_path):
    registry = PhaseLTrainingRecipeRegistry(tmp_path)
    record = PhaseLTrainingRecipeRecord(
        experiment_id="phase-l-recipe-fixture",
        evidence_class=EvidenceClass.DEVELOPMENT_SCALE,
        architecture_config={}, actual_parameter_count=50_000_000,
        tokenizer_id="fixture", tokenizer_vocab_size=320, corpus_release_id="fixture", corpus_token_count=1_000_000,
        train_token_count=800_000, validation_token_count=100_000, test_token_count=100_000,
        optimizer="AdamW", learning_rate=0.001, scheduler="linear_warmup_constant", warmup=1,
        batch_size=1, sequence_length=16, gradient_accumulation=1, weight_decay=0.0,
        initialization="RANDOM", precision="float32", seed=1, training_steps=100, training_tokens=1600,
        checkpoint_interval=25, hardware={}, failure_reasons=(FunctionalFailure.TRAINING_DURATION_INSUFFICIENT,),
    )
    registry.record(record)
    with pytest.raises(ValueError, match="already exists"):
        registry.record(record)
    gate = one_b_functional_gate(registry.records())
    assert gate.status is OneBFunctionalStatus.NOT_STARTED
    assert gate.failure_reasons == (FunctionalFailure.INSUFFICIENT_EVIDENCE,)
    health = aggregate_generation_health([{"eos_rate": 1.0, "average_length": 4, "repeated_unigram_rate": 0.25, "repeated_bigram_rate": 0.0, "distinct_1": 0.75, "distinct_2": 1.0, "entropy_bits": 2.0}])
    assert health.eos_rate == 1.0
    assert health.coherent_completion_rate == "INSUFFICIENT_EVIDENCE"


def test_duration_rungs_require_distinct_immutable_recipe_ids(tmp_path):
    registry = PhaseLTrainingRecipeRegistry(tmp_path)
    common = {
        "evidence_class": EvidenceClass.DEVELOPMENT_SCALE,
        "architecture_config": {}, "actual_parameter_count": 50_000_000,
        "tokenizer_id": "fixture", "tokenizer_vocab_size": 320, "corpus_release_id": "fixture", "corpus_token_count": 1_000_000,
        "train_token_count": 800_000, "validation_token_count": 100_000, "test_token_count": 100_000,
        "optimizer": "AdamW", "learning_rate": 0.001, "scheduler": "linear_warmup_constant", "warmup": 1,
        "batch_size": 1, "sequence_length": 16, "gradient_accumulation": 1, "weight_decay": 0.0,
        "initialization": "RANDOM", "precision": "float32", "seed": 1, "checkpoint_interval": 25, "hardware": {},
    }
    registry.record(PhaseLTrainingRecipeRecord(experiment_id="run-training-000100", training_steps=100, training_tokens=1_600, **common))
    registry.record(PhaseLTrainingRecipeRecord(experiment_id="run-training-000200", training_steps=200, training_tokens=3_200, **common))
    assert [record.training_tokens for record in registry.records()] == [1_600, 3_200]


def test_phase_l_objective_sanity_verifies_shift_mask_and_eos_targets():
    report = objective_sanity()
    assert report["result"] == "PASS"
    assert report["checks"]["correct_lower_than_shifted"]
    assert report["checks"]["masked_target_zero_contribution"]
    assert report["checks"]["eos_target_supported"]


def test_masked_packed_rows_preserve_width_and_expose_only_active_targets(tmp_path):
    shard = tmp_path / "train-00000.jsonl"
    shard.write_text('{"document_id":"fixture","input_ids":[11,12,2,2,2],"loss_mask":[1,1,0,0]}\n', encoding="utf-8")
    dataset = PackedShardSequenceDataset([shard], expected_width=5, include_loss_mask=True)
    ids, mask = dataset[0]
    assert ids == [11, 12, 2, 2, 2]
    assert mask == [1, 1, 0, 0]


def test_branch_comparison_refuses_to_promote_loss_only_winner():
    branch = {
        "experiment_id": "loss-only", "functional_level": 0,
        "generation": {"eos_rate": 0.0, "repeated_unigram_rate": 0.984375, "distinct_1": 0.015625},
        "capability": {"language": 0.0}, "test": {"loss": 1.0},
    }
    report = compare_branches([branch])
    assert report["result"] == "NO_BRANCH_FUNCTIONAL_IMPROVEMENT"
    assert report["overall_recommended_branch"] is None


def test_branch_comparison_rejects_immediate_eos_collapse_as_functional_signal():
    generation = {"records": [{"aggregate": {"eos_rate": 1.0, "average_length": 1.0, "repeated_unigram_rate": 0.0, "distinct_1": 1.0}}]}
    assert functional_level(generation, {"scores": {"language": 0.0}}) == 0
