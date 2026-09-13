"""A bounded Phase P evidence cycle; it intentionally does not manufacture success."""
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
from app.foundation.model import UltronModelConfig
from app.foundation.phase_o import AutonomyLevel, PhaseOEngine, TrainingJobStatus
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    DevelopmentTrace,
    EvaluationPartition,
    ItemRole,
    PhasePEngine,
    create_capability_suite,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import TrainingConfig


def _release(root: Path) -> tuple[Path, UltronTokenizer]:
    documents = [
        CorpusDocument(
            content=f"ULTRON locally authored corpus document {index}. Independent language sequence {index} contains varied ordinary prose for proprietary model training.",
            provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier=f"phase-p-e2e-{index}", source_group=f"phase-p-e2e-source-{index}", license="CC0-1.0", license_evidence="owned fixture"),
        )
        for index in range(30)
    ]
    release = CorpusPipeline(root, CorpusConfig(validation_percent=20, test_percent=20)).release("phase-p-e2e-release", documents)
    corpus = root / "tokenizer.txt"
    corpus.write_text("\n".join(item.content for item in documents), encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, root / "tokenizer", TokenizerConfig(version="phase-p-e2e-tokenizer", vocab_size=320))
    return root / "corpora" / "releases" / release.release_id, tokenizer


def _suite(root: Path):
    domains = (
        CapabilityDomain.GENERAL_KNOWLEDGE,
        CapabilityDomain.FACTUALITY,
        CapabilityDomain.MATHEMATICS,
        CapabilityDomain.PROGRAMMING,
        CapabilityDomain.REASONING,
        CapabilityDomain.INSTRUCTION_FOLLOWING,
        CapabilityDomain.GENERALIZATION,
    )
    items = tuple(
        CapabilityItem(
            task_id=f"phase-p-{domain.value.lower()}", domain=domain, partition=EvaluationPartition.TEST,
            role=ItemRole.GENERALIZATION if domain is CapabilityDomain.GENERALIZATION else ItemRole.TARGET,
            prompt=f"probe{index}",
            expected_answer="unreachable fixture answer alpha beta gamma delta epsilon", difficulty=5,
            provenance={"source": "Phase P task-owned benchmark fixture"},
        )
        for index, domain in enumerate(domains)
    )
    return create_capability_suite(root / "phase-p-e2e-suite.json", suite_id="phase-p-e2e-capability", version="v1", partition=EvaluationPartition.TEST, items=items)


def _plan(*, candidate_id: str, experiment_id: str, config: TrainingConfig, model: UltronModelConfig, dataset_id: str, dataset_manifest_sha256: str, tokenizer: UltronTokenizer, tokens: int, parent: str | None = None) -> CandidatePlan:
    return CandidatePlan(
        candidate_id=candidate_id, experiment_id=experiment_id, model_id=model.model_name, model_version=model.version,
        dataset_release_id=dataset_id, dataset_manifest_sha256=dataset_manifest_sha256, tokenizer_sha256=tokenizer.sha256,
        training_config=config.model_dump(mode="json"), seed=config.seed, hardware={"environment": "pytest"},
        curriculum_stage="REASONING", training_token_budget=tokens, reason="Bounded locally owned Phase P capability experiment.", parent_experiment=parent,
    )


def test_phase_p_real_cycle_trains_more_than_proof_of_life_and_selects_next_experiment(tmp_path: Path) -> None:
    release_dir, tokenizer = _release(tmp_path)
    engine = PhaseOEngine(tmp_path, autonomy=AutonomyLevel.EVALUATION_AND_REJECTION)
    dataset = engine.register_dataset(dataset_id="phase-p-e2e-dataset", release_dir=release_dir, tokenizer=tokenizer, curriculum_roles=("REASONING",))
    model = UltronModelConfig(model_name="ultron-phase-p-e2e", version="v1", vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    baseline_config = TrainingConfig(run_id="phase-p-e2e-run", model_id=model.model_name, steps=1, batch_size=1, checkpoint_every=1, validation_every=1, dataset_release_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256)
    candidate_config = baseline_config.model_copy(update={"steps": 12})
    engine.development.configure(PromotionPolicy(policy_id="phase-p-e2e-policy", minimum_improvement=0.01, max_training_tokens_per_experiment=176, max_training_tokens_total=192, allow_automatic_training_start=True))
    suite = _suite(tmp_path)
    phase_p = PhasePEngine(tmp_path)
    assert phase_p.validate_dataset_isolation(release_dir=release_dir, suite=suite).status == "CLEAR"

    baseline_plan = _plan(candidate_id="phase-p-baseline", experiment_id="phase-p-exp-001", config=baseline_config, model=model, dataset_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256, tokenizer=tokenizer, tokens=16)
    engine.development.plan_candidate(baseline_plan)
    baseline_job = engine.run_job(engine.queue_job(plan=baseline_plan, dataset_id=dataset.dataset_id, tokenizer_path=tokenizer.model_path.parent, model_config=model, training_config=baseline_config).job_id)
    assert baseline_job.status is TrainingJobStatus.EVALUATING
    baseline_run, _ = engine.evaluate_capability_model_only(baseline_job.job_id, suite, max_new_tokens=4)
    engine.development.establish_initial_champion(baseline_plan.candidate_id, operator_id="phase-p-test-operator")

    candidate_plan = _plan(candidate_id="phase-p-candidate", experiment_id="phase-p-exp-002", config=candidate_config, model=model, dataset_id=dataset.release_id, dataset_manifest_sha256=dataset.release_manifest_sha256, tokenizer=tokenizer, tokens=176, parent=baseline_plan.experiment_id)
    engine.development.plan_candidate(candidate_plan)
    candidate_job = engine.run_job(engine.queue_job(plan=candidate_plan, dataset_id=dataset.dataset_id, tokenizer_path=tokenizer.model_path.parent, model_config=model, training_config=candidate_config, resume_checkpoint=Path(baseline_job.checkpoint_path or "")).job_id)
    assert candidate_job.current_tokens == 176
    assert candidate_job.status is TrainingJobStatus.EVALUATING
    assert sha256_file(Path(candidate_job.checkpoint_path or "") / "model.safetensors") != sha256_file(Path(baseline_job.checkpoint_path or "") / "model.safetensors")
    candidate_run, comparison = engine.evaluate_capability_model_only(candidate_job.job_id, suite, baseline_run=baseline_run, max_new_tokens=4)
    decision = engine.development.validate_candidate(candidate_plan.candidate_id)
    assert decision.decision == "REJECTED"
    assert "measurable improvement" in decision.rationale

    _, _, _, opportunities = phase_p.analyze(
        baseline=baseline_run, candidate=candidate_run, suite=suite,
        importance={domain: 0.8 for domain in CapabilityDomain}, expected_improvement={domain: 0.05 for domain in CapabilityDomain},
        generalization_potential={domain: 0.5 for domain in CapabilityDomain}, confidence={domain: 0.6 for domain in CapabilityDomain},
        estimated_compute_tokens={domain: 176 for domain in CapabilityDomain},
    )
    next_hypothesis = phase_p.select_next_experiment(opportunities=opportunities, dataset_ids=(dataset.dataset_id,), intervention_by_domain={item.domain: "Add independently validated, non-overlapping examples for this measured failure cluster." for item in opportunities}, maximum_acceptable_regression=0.0)
    assert next_hypothesis is not None
    outcome = phase_p.record_outcome(hypothesis=next_hypothesis, candidate_run=candidate_run, comparison=comparison, tokens_used=candidate_job.current_tokens)
    trace = phase_p.write_trace(DevelopmentTrace(trace_id="phase-p-e2e-trace", cycle_id="phase-p-cycle-001", candidate_id=candidate_plan.candidate_id, baseline_run_id=baseline_run.run_id, candidate_run_id=candidate_run.run_id, comparison_id=f"comparison-{candidate_run.run_id}", selected_opportunity_id=opportunities[0].opportunity_id, hypothesis_id=next_hypothesis.experiment_id, dataset_ids=(dataset.dataset_id,), training_tokens=candidate_job.current_tokens, decision=decision.decision, rationale=decision.rationale, next_experiment_id=next_hypothesis.experiment_id))
    assert outcome.status.value in {"INCONCLUSIVE", "REFUTED"}
    assert trace.next_experiment_id == next_hypothesis.experiment_id
    monitor = engine.monitor()
    assert monitor.latest_capability_comparison is not None
    assert monitor.target_weakness
    assert monitor.next_experiment and monitor.next_experiment.experiment_id == next_hypothesis.experiment_id
    assert monitor.resource_state["autonomy_level"] == "EVALUATION_AND_REJECTION"
