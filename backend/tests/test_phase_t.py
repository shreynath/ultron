from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    Provenance,
    SourceType,
)
from app.foundation.development import CandidatePlan, PromotionPolicy
from app.foundation.model import UltronModelConfig
from app.foundation.phase_o import PhaseOEngine
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    EvaluationPartition,
    ItemRole,
    create_capability_suite,
)
from app.foundation.phase_r import LearningAttribution, LearningEvidenceLevel
from app.foundation.phase_t import (
    DeterministicMixtureSampler,
    PhaseTEngine,
    create_campaign,
    exposure_for,
    feasibility_for,
    register_architecture,
    scaling_conclusion,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import TrainingConfig


def test_registry_counts_constructed_model_and_campaign_is_immutable(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("owned ULTRON training corpus. " * 40)
    tokenizer = UltronTokenizer.train(
        corpus, tmp_path / "tokenizer", TokenizerConfig(version="t", vocab_size=320)
    )
    config = UltronModelConfig(
        model_name="ultron-scale-1m",
        version="v1",
        vocab_size=tokenizer.vocab_size,
        hidden_size=160,
        num_layers=4,
        num_heads=4,
        intermediate_size=640,
        max_sequence_length=64,
    )
    record = register_architecture(model_config=config, tokenizer=tokenizer)
    assert record.parameter_count == 1_342_880
    feasibility = feasibility_for(
        model=config, batch_size=2, unified_memory_bytes=24 * 1024**3
    )
    assert feasibility.status == "FEASIBLE_CONSERVATIVE_ESTIMATE"
    campaign = create_campaign(
        campaign_id="scale",
        architecture_id=record.model_id,
        architecture_hash=record.configuration_sha256,
        tokenizer_sha256=tokenizer.sha256,
        dataset_release_ids=("release",),
        dataset_release_hashes=("a" * 64,),
        mixture_policy_hash="b" * 64,
        requested_tokens=1024,
        maximum_steps=8,
        batch_size=2,
        sequence_length=64,
        max_checkpoints=2,
        seed=17,
    )
    engine = PhaseTEngine(tmp_path)
    engine.register_architecture(record)
    engine.record_campaign(campaign)
    with pytest.raises(ValueError, match="campaign changed"):
        engine.record_campaign(campaign.model_copy(update={"requested_tokens": 2048}))


def test_mixture_sampling_is_seeded_and_reports_actual_exposure() -> None:
    sampler = DeterministicMixtureSampler(
        sequences_by_release={"a": (1, 2), "b": (3,)},
        weights={"a": 0.7, "b": 0.3},
        seed=9,
    )
    samples = sampler.plan(1000)
    assert samples == sampler.plan(1000)
    exposure = exposure_for(
        weights={"a": 0.7, "b": 0.3}, samples=samples, tokens_per_sample=64, seed=9
    )
    assert exposure.total_tokens == 64_000
    assert not exposure.materially_diverged
    attribution = LearningAttribution(
        attribution_id="a",
        experiment_id="e",
        baseline_run_id="b",
        candidate_run_id="c",
        weights_changed=True,
        training_loss_delta=-0.2,
        target_capability_delta=0,
        generalization_delta=0,
        regression_rate=0,
        evidence=LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
        rationale="zero",
    )
    assert scaling_conclusion(attribution=attribution) == "TRAINING_SIGNAL_ONLY"


def test_phase_t_real_larger_from_scratch_campaign(tmp_path: Path) -> None:
    docs = tuple(
        CorpusDocument(
            content=f"Owned ULTRON development language record {index}. A recursive function stops at a base case and combines child results. Structured reasoning checks assumptions before conclusions. Programming examples use distinct variable names {index}.",
            provenance=Provenance(
                source_type=SourceType.LOCAL_AUTHORED,
                source_identifier=f"scale-{index}",
                source_group=f"scale-{index % 8}",
                license="CC0-1.0",
                license_evidence="owned Phase T fixture",
            ),
        )
        for index in range(96)
    )
    release = CorpusPipeline(
        tmp_path, CorpusConfig(validation_percent=20, test_percent=20)
    ).release("phase-t-owned-v1", docs)
    corpus = tmp_path / "tokenizer.txt"
    corpus.write_text("\n".join(item.content for item in docs))
    tokenizer = UltronTokenizer.train(
        corpus,
        tmp_path / "tokenizer",
        TokenizerConfig(version="phase-t-owned-tokenizer", vocab_size=320),
    )
    model = UltronModelConfig(
        model_name="ultron-phase-t-1m",
        version="v1",
        vocab_size=tokenizer.vocab_size,
        hidden_size=160,
        num_layers=4,
        num_heads=4,
        intermediate_size=640,
        max_sequence_length=64,
    )
    phase_t = PhaseTEngine(tmp_path)
    architecture = phase_t.register_architecture(
        register_architecture(model_config=model, tokenizer=tokenizer)
    )
    engine = PhaseOEngine(tmp_path)
    dataset = engine.register_dataset(
        dataset_id="phase-t-owned",
        release_dir=tmp_path / "corpora" / "releases" / release.release_id,
        tokenizer=tokenizer,
        curriculum_roles=("LANGUAGE", "PROGRAMMING", "REASONING"),
    )
    engine.development.configure(
        PromotionPolicy(
            policy_id="phase-t-policy",
            minimum_improvement=0.01,
            max_training_tokens_per_experiment=2_048,
            max_training_tokens_total=2_304,
            allow_automatic_training_start=True,
        )
    )

    def plan(
        candidate_id: str,
        experiment_id: str,
        config: TrainingConfig,
        tokens: int,
        parent: str | None = None,
    ) -> CandidatePlan:
        return CandidatePlan(
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            model_id=model.model_name,
            model_version=model.version,
            dataset_release_id=dataset.release_id,
            dataset_manifest_sha256=dataset.release_manifest_sha256,
            tokenizer_sha256=tokenizer.sha256,
            training_config=config.model_dump(mode="json"),
            seed=config.seed,
            hardware={"environment": "phase-t-pytest"},
            curriculum_stage="DEVELOPMENT_SCALE",
            training_token_budget=tokens,
            reason="Phase T controlled from-scratch larger-model campaign.",
            parent_experiment=parent,
        )

    baseline_config = TrainingConfig(
        run_id="phase-t-baseline",
        model_id=model.model_name,
        steps=1,
        batch_size=2,
        checkpoint_every=1,
        validation_every=1,
        learning_rate=3e-4,
        dataset_release_id=dataset.release_id,
        dataset_manifest_sha256=dataset.release_manifest_sha256,
    )
    baseline_plan = plan(
        "phase-t-baseline", "phase-t-baseline-exp", baseline_config, 128
    )
    engine.development.plan_candidate(baseline_plan)
    baseline_job = engine.run_job(
        engine.queue_job(
            plan=baseline_plan,
            dataset_id=dataset.dataset_id,
            tokenizer_path=tokenizer.model_path.parent,
            model_config=model,
            training_config=baseline_config,
        ).job_id
    )
    suite = create_capability_suite(
        tmp_path / "scale-suite.json",
        suite_id="phase-t-scale-suite",
        version="v1",
        partition=EvaluationPartition.TEST,
        items=tuple(
            CapabilityItem(
                task_id=f"scale-{domain.value.lower()}",
                domain=domain,
                partition=EvaluationPartition.TEST,
                role=ItemRole.TARGET,
                prompt=f"scale probe {index}",
                expected_answer="heldout owned answer not in training corpus",
                difficulty=5,
                provenance={"source": "Phase T independently authored heldout fixture"},
            )
            for index, domain in enumerate(
                (
                    CapabilityDomain.GENERAL_KNOWLEDGE,
                    CapabilityDomain.FACTUALITY,
                    CapabilityDomain.MATHEMATICS,
                    CapabilityDomain.PROGRAMMING,
                    CapabilityDomain.REASONING,
                    CapabilityDomain.INSTRUCTION_FOLLOWING,
                    CapabilityDomain.GENERALIZATION,
                )
            )
        ),
    )
    baseline_run, _ = engine.evaluate_capability_model_only(
        baseline_job.job_id, suite, max_new_tokens=4
    )
    engine.development.establish_initial_champion(
        baseline_plan.candidate_id, operator_id="phase-t-test"
    )
    candidate_config = baseline_config.model_copy(
        update={"run_id": "phase-t-candidate", "steps": 16, "seed": 18}
    )
    candidate_plan = plan(
        "phase-t-candidate",
        "phase-t-candidate-exp",
        candidate_config,
        2_048,
        parent=baseline_plan.experiment_id,
    )
    engine.development.plan_candidate(candidate_plan)
    candidate_job = engine.run_job(
        engine.queue_job(
            plan=candidate_plan,
            dataset_id=dataset.dataset_id,
            tokenizer_path=tokenizer.model_path.parent,
            model_config=model,
            training_config=candidate_config,
            initial_weights=Path(baseline_job.checkpoint_path or "")
            / "model.safetensors",
        ).job_id
    )
    candidate_run, comparison = engine.evaluate_capability_model_only(
        candidate_job.job_id, suite, baseline_run=baseline_run, max_new_tokens=4
    )
    baseline_metrics = json.loads(
        (
            tmp_path / "models" / "base" / baseline_config.run_id / "metrics.json"
        ).read_text()
    )
    candidate_metrics = json.loads(
        (
            tmp_path / "models" / "base" / candidate_config.run_id / "metrics.json"
        ).read_text()
    )
    attribution = LearningAttribution(
        attribution_id="phase-t-learning",
        experiment_id=candidate_plan.experiment_id,
        baseline_run_id=baseline_run.run_id,
        candidate_run_id=candidate_run.run_id,
        weights_changed=(
            Path(baseline_job.checkpoint_path or "") / "model.safetensors"
        ).read_bytes()
        != (
            Path(candidate_job.checkpoint_path or "") / "model.safetensors"
        ).read_bytes(),
        training_loss_delta=candidate_metrics["training_loss"]
        - baseline_metrics["training_loss"],
        held_out_loss_delta=candidate_metrics["validation_loss"]
        - baseline_metrics["validation_loss"],
        target_capability_delta=comparison.domain_deltas[
            CapabilityDomain.PROGRAMMING.value
        ],
        generalization_delta=None,
        regression_rate=comparison.regression_rate,
        evidence=LearningEvidenceLevel.NO_LEARNING_EVIDENCE,
        rationale="Actual Phase P suite produced no target score delta.",
    )
    campaign = phase_t.record_campaign(
        create_campaign(
            campaign_id="phase-t-1m-owned",
            architecture_id=architecture.model_id,
            architecture_hash=architecture.configuration_sha256,
            tokenizer_sha256=tokenizer.sha256,
            dataset_release_ids=(release.release_id,),
            dataset_release_hashes=(dataset.release_manifest_sha256,),
            mixture_policy_hash="single-owned-release",
            requested_tokens=candidate_job.current_tokens,
            maximum_steps=candidate_config.steps,
            batch_size=2,
            sequence_length=model.max_sequence_length,
            max_checkpoints=16,
            seed=candidate_config.seed,
        )
    )
    evidence = phase_t.persist(
        "evidence",
        "phase-t-1m-owned",
        __import__(
            "app.foundation.phase_t", fromlist=["ScalingEvidence"]
        ).ScalingEvidence(
            evidence_id="phase-t-1m-owned",
            campaign_id=campaign.campaign_id,
            parameter_count=architecture.parameter_count,
            training_tokens=candidate_job.current_tokens,
            tokens_per_parameter=candidate_job.current_tokens
            / architecture.parameter_count,
            baseline_validation_loss=baseline_metrics["validation_loss"],
            candidate_validation_loss=candidate_metrics["validation_loss"],
            validation_loss_delta=candidate_metrics["validation_loss"]
            - baseline_metrics["validation_loss"],
            baseline_capability=sum(baseline_run.domain_scores.values())
            / len(baseline_run.domain_scores),
            candidate_capability=sum(candidate_run.domain_scores.values())
            / len(candidate_run.domain_scores),
            capability_delta=comparison.domain_deltas[
                CapabilityDomain.PROGRAMMING.value
            ],
            generalization_delta=None,
            regression_rate=comparison.regression_rate,
            weights_changed=attribution.weights_changed,
            throughput_tokens_per_second=candidate_metrics["tokens_per_second"],
            wall_seconds=None,
            compute_units=None,
            conclusion=scaling_conclusion(attribution=attribution),
        ),
    )
    assert architecture.parameter_count == 1_342_880
    assert candidate_job.current_tokens == 2_048
    assert attribution.weights_changed
    assert evidence.conclusion in {"NO_LEARNING_EVIDENCE", "TRAINING_SIGNAL_ONLY"}
