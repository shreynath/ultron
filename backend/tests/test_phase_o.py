from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.checkpoints import sha256_file
from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    Provenance,
    SourceType,
)
from app.foundation.development import CandidatePlan, PromotionPolicy
from app.foundation.evaluation import suite_from_text
from app.foundation.model import UltronModelConfig
from app.foundation.phase_o import (
    AutonomyLevel,
    DatasetEligibility,
    PhaseOEngine,
    TrainingJobStatus,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


def _release(root: Path) -> tuple[Path, UltronTokenizer]:
    documents = []
    for index in range(15):
        words = " ".join(f"token{index}_{item}" for item in range(30))
        documents.append(CorpusDocument(
            content=f"ULTRON local training document {index}. {words}. This document has independent provenance and sufficient lexical variety.",
            provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier=f"fixture-{index}", source_group=f"source-{index}", license="CC0-1.0", license_evidence="fixture"),
        ))
    release = CorpusPipeline(root, CorpusConfig(validation_percent=20, test_percent=20)).release("phase-o-fixture-release", documents)
    corpus = root / "tokenizer.txt"
    corpus.write_text("\n".join(document.content for document in documents), encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, root / "tokenizer", TokenizerConfig(version="phase-o-fixture-tokenizer", vocab_size=320))
    return root / "corpora" / "releases" / release.release_id, tokenizer


def test_phase_o_runs_real_owned_training_registers_checkpoint_and_rejects_without_capability_evidence(tmp_path: Path) -> None:
    release_dir, tokenizer = _release(tmp_path)
    engine = PhaseOEngine(tmp_path, autonomy=AutonomyLevel.EVALUATION_AND_REJECTION)
    dataset = engine.register_dataset(dataset_id="phase-o-fixture", release_dir=release_dir, tokenizer=tokenizer, curriculum_roles=("FOUNDATIONAL_LANGUAGE",))
    assert dataset.eligibility is DatasetEligibility.ELIGIBLE

    model = UltronModelConfig(model_name="ultron-phase-o-fixture", version="v1", vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    baseline_config = TrainingConfig(run_id="phase-o-fixture-run", model_id="ultron-phase-o-fixture", steps=1, batch_size=1, checkpoint_every=1, validation_every=1, dataset_release_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256)
    train = engine._split_text(release_dir, "train")
    validation = engine._split_text(release_dir, "validation")
    baseline = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=train, validation_text=validation, config=baseline_config)
    baseline_checkpoint = Path(baseline.checkpoint_path)
    engine.register_checkpoint(baseline_checkpoint, model)

    config = baseline_config.model_copy(update={"steps": 2})
    plan = CandidatePlan(
        candidate_id="phase-o-candidate", experiment_id="phase-o-exp-001", model_id=config.model_id, model_version=model.version,
        dataset_release_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256,
        tokenizer_sha256=tokenizer.sha256, training_config=config.model_dump(mode="json"), seed=config.seed,
        hardware={"environment": "pytest"}, curriculum_stage="FOUNDATIONAL_LANGUAGE", training_token_budget=64,
        reason="Continue the owned baseline for one bounded real optimizer update.", parent_experiment="phase-o-baseline",
    )
    engine.development.configure(PromotionPolicy(policy_id="phase-o-policy", max_training_tokens_per_experiment=64, max_training_tokens_total=128, allow_automatic_training_start=True))
    engine.development.plan_candidate(plan)
    job = engine.queue_job(plan=plan, dataset_id=dataset.dataset_id, tokenizer_path=tokenizer.model_path.parent, model_config=model, training_config=config, resume_checkpoint=baseline_checkpoint)
    completed = engine.run_job(job.job_id)
    assert completed.status is TrainingJobStatus.EVALUATING
    assert completed.checkpoint_path
    assert completed.current_tokens == 16
    candidate_checkpoint = Path(completed.checkpoint_path)
    assert sha256_file(candidate_checkpoint / "model.safetensors") != sha256_file(baseline_checkpoint / "model.safetensors")
    engine.register_checkpoint(candidate_checkpoint, model)

    probe = tmp_path / "model-only-probe.txt"
    probe.write_text(validation, encoding="utf-8")
    suite = suite_from_text("phase-o-model-only", "v1", probe, max_tokens=16)
    engine.run_once(suite=suite)
    evaluation = engine.store.evaluations()[0]
    assert evaluation.mode.value == "BASE_MODEL"
    assert evaluation.report.checkpoint_id
    decision = engine.development.validate_candidate(plan.candidate_id)
    assert decision.decision == "REJECTED"
    restarted = PhaseOEngine(tmp_path)
    assert restarted.monitor().latest_evaluation is not None


def test_phase_o_recovers_only_jobs_with_a_valid_checkpoint(tmp_path: Path) -> None:
    release_dir, tokenizer = _release(tmp_path)
    engine = PhaseOEngine(tmp_path)
    dataset = engine.register_dataset(dataset_id="phase-o-fixture", release_dir=release_dir, tokenizer=tokenizer, curriculum_roles=("GENERAL_KNOWLEDGE",))
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    config = TrainingConfig(run_id="phase-o-interrupted", model_id="interrupted", steps=1, batch_size=1, checkpoint_every=1, validation_every=1, dataset_release_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256)
    plan = CandidatePlan(candidate_id="interrupted-candidate", experiment_id="interrupted-exp", model_id="interrupted", model_version=model.version, dataset_release_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256, tokenizer_sha256=tokenizer.sha256, training_config=config.model_dump(mode="json"), seed=config.seed, hardware={}, curriculum_stage="GENERAL_KNOWLEDGE", training_token_budget=16, reason="Failure-recovery fixture.")
    engine.development.configure(PromotionPolicy(policy_id="recovery-policy", max_training_tokens_per_experiment=16, max_training_tokens_total=16))
    engine.development.plan_candidate(plan)
    job = engine.queue_job(plan=plan, dataset_id=dataset.dataset_id, tokenizer_path=tokenizer.model_path.parent, model_config=model, training_config=config)
    engine.store.replace_job(job.model_copy(update={"status": TrainingJobStatus.RUNNING}))
    recovered = engine.recover_interrupted_jobs()
    assert recovered[0].status is TrainingJobStatus.FAILED
