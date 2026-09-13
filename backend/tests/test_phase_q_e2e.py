"""A real Phase Q data-release handoff into Phase O/P; no teacher is invoked."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.checkpoints import canonical_hash, sha256_file
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
    DevelopmentTrace,
    EvaluationPartition,
    ItemRole,
    LearningOpportunity,
    PhasePEngine,
    compare_capability_runs,
    create_capability_suite,
)
from app.foundation.phase_q import (
    DataPolicy,
    DatasetContribution,
    DataSource,
    DataSourceType,
    PhaseQEngine,
    SourceEligibility,
)
from app.foundation.phase_r import (
    GeneralizationClass,
    PhaseREngine,
    assess_generalization,
    classify_learning_evidence,
    create_experiment,
    create_experiment_policy,
    create_generalization_policy,
    detect_contamination,
    diagnose_failure,
    efficiency_for,
    next_recommendation,
)
from app.foundation.phase_s import (
    CheckpointEvidence,
    DifficultyBand,
    PhaseSEngine,
    SamplingPolicy,
    adapt_intervention,
    create_early_stop_policy,
    create_intervention,
    create_learning_policy,
    recommend_next_training,
    select_checkpoint,
)
from app.foundation.phase_s import (
    efficiency_for as phase_s_efficiency_for,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import TrainingConfig


def _plan(
    *,
    candidate_id: str,
    experiment_id: str,
    model: UltronModelConfig,
    config: TrainingConfig,
    tokenizer: UltronTokenizer,
    tokens: int,
    parent: str | None = None,
) -> CandidatePlan:
    return CandidatePlan(
        candidate_id=candidate_id,
        experiment_id=experiment_id,
        model_id=model.model_name,
        model_version=model.version,
        dataset_release_id=config.dataset_release_id,
        dataset_manifest_sha256=config.dataset_manifest_sha256,
        tokenizer_sha256=tokenizer.sha256,
        training_config=config.model_dump(mode="json"),
        seed=config.seed,
        hardware={"environment": "pytest"},
        curriculum_stage="PROGRAMMING",
        training_token_budget=tokens,
        reason="Phase Q controlled target-data experiment.",
        parent_experiment=parent,
    )


def test_phase_q_real_target_data_release_hands_off_to_training_and_honest_evaluation(
    tmp_path: Path,
) -> None:
    generic_docs = tuple(
        CorpusDocument(
            content=f"Generic owned baseline document {index}. This independent prose trains a small local ULTRON decoder without targeted recursion examples.",
            provenance=Provenance(
                source_type=SourceType.LOCAL_AUTHORED,
                source_identifier=f"baseline-{index}",
                source_group=f"baseline-source-{index}",
                license="CC0-1.0",
                license_evidence="owned fixture",
            ),
        )
        for index in range(30)
    )
    baseline_release = CorpusPipeline(
        tmp_path, CorpusConfig(validation_percent=20, test_percent=20)
    ).release("phase-q-baseline-v1", generic_docs)
    tokenizer_corpus = tmp_path / "tokenizer.txt"
    tokenizer_corpus.write_text(
        "\n".join(item.content for item in generic_docs), encoding="utf-8"
    )
    tokenizer = UltronTokenizer.train(
        tokenizer_corpus,
        tmp_path / "tokenizer",
        TokenizerConfig(version="phase-q-e2e-tokenizer", vocab_size=320),
    )
    model = UltronModelConfig(
        model_name="ultron-phase-q-e2e",
        version="v1",
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_heads=4,
        num_layers=1,
        intermediate_size=32,
        max_sequence_length=16,
    )
    engine = PhaseOEngine(tmp_path)
    registered_baseline = engine.register_dataset(
        dataset_id="phase-q-baseline",
        release_dir=tmp_path / "corpora" / "releases" / baseline_release.release_id,
        tokenizer=tokenizer,
        curriculum_roles=("GENERAL_KNOWLEDGE",),
    )
    baseline_config = TrainingConfig(
        run_id="phase-q-baseline-run",
        model_id=model.model_name,
        steps=1,
        batch_size=1,
        checkpoint_every=1,
        validation_every=1,
        dataset_release_id=registered_baseline.release_id,
        dataset_manifest_sha256=registered_baseline.release_manifest_sha256,
    )
    engine.development.configure(
        PromotionPolicy(
            policy_id="phase-q-policy",
            minimum_improvement=0.01,
            max_training_tokens_per_experiment=192,
            max_training_tokens_total=400,
            allow_automatic_training_start=True,
        )
    )
    baseline_plan = _plan(
        candidate_id="phase-q-baseline",
        experiment_id="phase-q-baseline-exp",
        model=model,
        config=baseline_config,
        tokenizer=tokenizer,
        tokens=16,
    )
    engine.development.plan_candidate(baseline_plan)
    baseline_job = engine.run_job(
        engine.queue_job(
            plan=baseline_plan,
            dataset_id=registered_baseline.dataset_id,
            tokenizer_path=tokenizer.model_path.parent,
            model_config=model,
            training_config=baseline_config,
        ).job_id
    )
    suite = create_capability_suite(
        tmp_path / "phase-q-suite.json",
        suite_id="phase-q-capability",
        version="v1",
        partition=EvaluationPartition.TEST,
        items=tuple(
            CapabilityItem(
                task_id=f"phase-q-{domain.value.lower()}",
                domain=domain,
                partition=EvaluationPartition.TEST,
                role=ItemRole.TARGET,
                prompt=f"q{index}",
                expected_answer="unreachable heldout answer alpha beta gamma delta",
                difficulty=5,
                provenance={"source": "owned heldout fixture"},
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
        baseline_plan.candidate_id, operator_id="phase-q-test-operator"
    )

    opportunity = LearningOpportunity(
        opportunity_id="opportunity-phase-q-recursion",
        source_run_id=baseline_run.run_id,
        domain=CapabilityDomain.PROGRAMMING,
        task_ids=("phase-q-programming",),
        importance=0.9,
        frequency=1.0,
        expected_improvement=0.1,
        generalization_potential=0.6,
        confidence=0.7,
        estimated_compute_tokens=192,
        priority=0.0002,
        rationale="Model-only baseline failed the held-out programming item.",
    )
    phase_q = PhaseQEngine(tmp_path)
    policy = phase_q.configure(
        DataPolicy(
            policy_id="phase-q-data-policy",
            maximum_raw_bytes=20_000,
            maximum_records_per_release=20,
        )
    )
    target = phase_q.create_target(opportunity)
    requirement = phase_q.create_requirement(
        target,
        minimum_records=12,
        required_example_kinds=(
            "recursive tracing",
            "recursive implementation",
            "recursive debugging",
        ),
        required_subdomains=("recursion",),
    )
    records = []
    for index in range(12):
        source = phase_q.register_source(
            DataSource(
                source_id=f"phase-q-owned-{index}",
                source_type=DataSourceType.USER_OWNED,
                provider="fixture-owner",
                location=f"local://phase-q/{index}",
                license_status="CC0-1.0",
                usage_permissions="USER_OWNED",
                domain=CapabilityDomain.PROGRAMMING,
                trust_level=1.0,
                acquisition_method="LOCAL_FIXTURE",
                eligibility=SourceEligibility.ELIGIBLE,
                eligibility_reason="Owner-approved training fixture.",
            )
        )
        phase_q.acquire_text(
            source=source,
            content=f"Owned recursion source {index}. It documents a distinct recursive algorithm and debugging practice.",
            policy=policy,
        )
        records.append(
            phase_q.source_record(
                record_id=f"phase-q-recursion-{index}",
                target=target,
                source=source,
                instruction=f"Trace recursive accumulator exercise {index} and state its base case.",
                input=f"value {index}",
                output=f"Check the base case, reduce the argument, and combine result {index} after the recursive call.",
                difficulty=2 + index % 5,
                subdomain="recursion",
            )
        )
    curriculum = phase_q.build_curriculum(
        target=target,
        requirement=requirement,
        records=tuple(records),
        tokenizer=tokenizer,
    )
    targeted_release = phase_q.release_training_data(
        release_id="phase-q-recursion-v1",
        target=target,
        requirement=requirement,
        curriculum=curriculum,
        records=tuple(records),
        suite=suite,
    )
    registered_targeted = engine.register_dataset(
        dataset_id="phase-q-recursion",
        release_dir=tmp_path / "corpora" / "releases" / targeted_release.release_id,
        tokenizer=tokenizer,
        curriculum_roles=("PROGRAMMING",),
    )

    candidate_config = TrainingConfig(
        run_id="phase-q-recursion-run",
        model_id=model.model_name,
        steps=12,
        batch_size=1,
        checkpoint_every=1,
        validation_every=1,
        dataset_release_id=registered_targeted.release_id,
        dataset_manifest_sha256=registered_targeted.release_manifest_sha256,
    )
    candidate_plan = _plan(
        candidate_id="phase-q-candidate",
        experiment_id="phase-q-recursion-exp",
        model=model,
        config=candidate_config,
        tokenizer=tokenizer,
        tokens=192,
        parent=baseline_plan.experiment_id,
    )
    engine.development.plan_candidate(candidate_plan)
    candidate_job = engine.run_job(
        engine.queue_job(
            plan=candidate_plan,
            dataset_id=registered_targeted.dataset_id,
            tokenizer_path=tokenizer.model_path.parent,
            model_config=model,
            training_config=candidate_config,
            initial_weights=Path(baseline_job.checkpoint_path or "")
            / "model.safetensors",
        ).job_id
    )
    assert candidate_job.current_tokens == 192
    assert sha256_file(
        Path(candidate_job.checkpoint_path or "") / "model.safetensors"
    ) != sha256_file(Path(baseline_job.checkpoint_path or "") / "model.safetensors")
    candidate_run, comparison = engine.evaluate_capability_model_only(
        candidate_job.job_id, suite, baseline_run=baseline_run, max_new_tokens=4
    )
    decision = engine.development.validate_candidate(candidate_plan.candidate_id)
    contribution = phase_q.record_contribution(
        DatasetContribution(
            contribution_id="contribution-phase-q-recursion",
            dataset_release_id=targeted_release.release_id,
            target_id=target.target_id,
            experiment_id=candidate_plan.experiment_id,
            target_delta=comparison.domain_deltas["PROGRAMMING"],
            regression_rate=comparison.regression_rate,
            training_tokens=candidate_job.current_tokens,
            status=decision.decision,
        )
    )
    phase_p = PhasePEngine(tmp_path)
    _, _, _, next_opportunities = phase_p.analyze(
        baseline=baseline_run,
        candidate=candidate_run,
        suite=suite,
        importance={domain: 0.8 for domain in CapabilityDomain},
        expected_improvement={domain: 0.05 for domain in CapabilityDomain},
        generalization_potential={domain: 0.5 for domain in CapabilityDomain},
        confidence={domain: 0.6 for domain in CapabilityDomain},
        estimated_compute_tokens={domain: 192 for domain in CapabilityDomain},
    )
    next_hypothesis = phase_p.select_next_experiment(
        opportunities=next_opportunities,
        dataset_ids=(registered_targeted.dataset_id,),
        intervention_by_domain={
            item.domain: "Construct a newly validated, non-overlapping targeted release before training."
            for item in next_opportunities
        },
        maximum_acceptable_regression=0.0,
    )
    assert next_hypothesis is not None
    trace = phase_p.write_trace(
        DevelopmentTrace(
            trace_id="phase-q-e2e-trace",
            cycle_id="phase-q-cycle-001",
            candidate_id=candidate_plan.candidate_id,
            baseline_run_id=baseline_run.run_id,
            candidate_run_id=candidate_run.run_id,
            comparison_id=f"comparison-{candidate_run.run_id}",
            selected_opportunity_id=opportunity.opportunity_id,
            hypothesis_id=next_hypothesis.experiment_id,
            dataset_ids=(registered_targeted.dataset_id,),
            training_tokens=candidate_job.current_tokens,
            decision=decision.decision,
            rationale=decision.rationale,
            next_experiment_id=next_hypothesis.experiment_id,
        )
    )
    assert decision.decision == "REJECTED"
    assert contribution.target_delta == 0.0
    assert candidate_run.runtime_dependency == "NONE_LOCAL_ULTRON_CHECKPOINT_ONLY"
    assert trace.next_experiment_id == next_hypothesis.experiment_id
    assert (
        tmp_path / "phase_q" / "releases" / f"{targeted_release.release_id}.json"
    ).is_file()

    # Phase R uses a second, independently authored held-out suite.  The real
    # tiny-model run is expected to show no measurable learning; that is valid
    # evidence and must not be rewritten into a success claim.
    generalization_suite = create_capability_suite(
        tmp_path / "phase-r-generalization-suite.json",
        suite_id="phase-r-independent-generalization",
        version="v1",
        partition=EvaluationPartition.TEST,
        items=tuple(
            CapabilityItem(
                task_id=f"phase-r-generalization-{index}",
                domain=CapabilityDomain.GENERALIZATION,
                partition=EvaluationPartition.TEST,
                role=ItemRole.GENERALIZATION,
                prompt=prompt,
                expected_answer="independently authored heldout response omega sigma",
                difficulty=6,
                provenance={"source": "independently authored Phase R fixture"},
            )
            for index, prompt in enumerate(
                (
                    "apply recursion to an unfamiliar tree shape",
                    "combine recursion with an unseen parity constraint",
                    "repair a counterfactual recursive base condition",
                )
            )
        ),
    )
    phase_p_generalization = PhasePEngine(tmp_path)
    baseline_generalization = phase_p_generalization.evaluate_model_only(
        checkpoint=Path(baseline_job.checkpoint_path or ""),
        tokenizer=tokenizer,
        model_config=model,
        suite=generalization_suite,
        candidate_id=baseline_plan.candidate_id,
        max_new_tokens=4,
    )
    candidate_generalization = phase_p_generalization.evaluate_model_only(
        checkpoint=Path(candidate_job.checkpoint_path or ""),
        tokenizer=tokenizer,
        model_config=model,
        suite=generalization_suite,
        candidate_id=candidate_plan.candidate_id,
        max_new_tokens=4,
    )
    generalization_comparison = compare_capability_runs(
        baseline_generalization, candidate_generalization
    )
    phase_r = PhaseREngine(tmp_path)
    policy_r = phase_r.configure(
        create_experiment_policy(
            "phase-r-e2e-policy",
            maximum_experiments=3,
            maximum_training_tokens=192,
            maximum_compute_units=10_000,
        )
    )
    experiment_r = phase_r.record_experiment(
        create_experiment(
            experiment_id="phase-r-e2e-recursion",
            parent_model_id=baseline_plan.candidate_id,
            parent_checkpoint_sha256=sha256_file(
                Path(baseline_job.checkpoint_path or "") / "model.safetensors"
            ),
            candidate_model_id=candidate_plan.candidate_id,
            candidate_checkpoint_sha256=sha256_file(
                Path(candidate_job.checkpoint_path or "") / "model.safetensors"
            ),
            target_id=target.target_id,
            target_domain=CapabilityDomain.PROGRAMMING.value,
            dataset_release_ids=(targeted_release.release_id,),
            dataset_release_hashes=(registered_targeted.release_manifest_sha256,),
            curriculum_id=curriculum.curriculum_id,
            curriculum_ordering_hash=canonical_hash(curriculum.ordered_record_ids),
            baseline_evaluation_ids=(
                baseline_run.run_id,
                baseline_generalization.run_id,
            ),
            candidate_evaluation_ids=(
                candidate_run.run_id,
                candidate_generalization.run_id,
            ),
            training_config_sha256=canonical_hash(
                candidate_config.model_dump(mode="json")
            ),
            evaluation_config_sha256=generalization_suite.suite_sha256,
            analysis_policy_id=policy_r.policy_id,
            analysis_policy_fingerprint=policy_r.fingerprint,
            seed=candidate_config.seed,
            training_tokens=candidate_job.current_tokens,
            optimizer_steps=candidate_job.current_step,
            wall_seconds=0,
            compute_units=float(candidate_job.current_tokens * 16),
            runtime="local-MLX",
            software_version="pytest-phase-r-e2e",
        )
    )
    contamination = detect_contamination(
        experiment_id=experiment_r.experiment_id,
        suite=generalization_suite,
        training_records={
            record.record_id: record.canonical_text for record in records
        },
    )
    assessment = assess_generalization(
        experiment_id=experiment_r.experiment_id,
        baseline=baseline_generalization,
        candidate=candidate_generalization,
        suite=generalization_suite,
        classes={
            "phase-r-generalization-0": GeneralizationClass.STRUCTURAL,
            "phase-r-generalization-1": GeneralizationClass.COMPOSITIONAL,
            "phase-r-generalization-2": GeneralizationClass.COUNTERFACTUAL,
        },
        policy=create_generalization_policy(
            "phase-r-e2e-generalization", require_all_nonempty_classes=False
        ),
        quarantined_task_ids=tuple({finding.task_id for finding in contamination}),
    )
    learning = classify_learning_evidence(
        experiment_id=experiment_r.experiment_id,
        baseline=baseline_run,
        candidate=candidate_run,
        comparison=comparison,
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=sha256_file(
            Path(baseline_job.checkpoint_path or "") / "model.safetensors"
        )
        != sha256_file(Path(candidate_job.checkpoint_path or "") / "model.safetensors"),
        training_loss_delta=None,
        held_out_loss_delta=None,
        generalization=assessment,
    )
    failure = diagnose_failure(
        experiment_id=experiment_r.experiment_id, attribution=learning
    )
    recommendation = next_recommendation(
        experiment=experiment_r,
        attribution=learning,
        failure=failure,
        expected_value=0.0,
        confidence=0.7,
    )
    efficiency = efficiency_for(
        experiment=experiment_r,
        capability_gain=comparison.domain_deltas[CapabilityDomain.PROGRAMMING.value],
        generalization_gain=assessment.generalization_score,
        newly_solved_items=0,
    )
    for category, item_id, artifact in (
        (
            "contamination",
            "phase-r-e2e-contamination",
            contamination[0] if contamination else None,
        ),
        ("generalization", assessment.assessment_id, assessment),
        ("learning", learning.attribution_id, learning),
        ("efficiency", efficiency.efficiency_id, efficiency),
        ("failures", failure.failure_id, failure),
        ("recommendations", recommendation.recommendation_id, recommendation),
    ):
        if artifact is not None:
            phase_r.persist(category, item_id, artifact)
    assert (
        generalization_comparison.domain_deltas[CapabilityDomain.GENERALIZATION.value]
        == 0.0
    )
    assert learning.target_capability_delta == 0.0  # No measurable learning.
    assert learning.evidence.value == "NO_LEARNING_EVIDENCE"
    assert failure.diagnosis.value == "NO_CAPABILITY_GAIN"
    assert recommendation.action.value == "ACQUIRE_MORE_DATA"
    assert (
        tmp_path / "phase_r" / "experiments" / "phase-r-e2e-recursion.json"
    ).is_file()
    monitor = engine.monitor()
    assert (
        monitor.latest_learning_attribution
        and monitor.latest_learning_attribution.evidence.value == "NO_LEARNING_EVIDENCE"
    )
    assert (
        monitor.next_learning_recommendation
        and monitor.next_learning_recommendation.action.value == "ACQUIRE_MORE_DATA"
    )

    # Phase S consumes the persisted zero-gain evidence, then makes a bounded
    # configuration change: a release with 25% newly authored structural
    # variants, a higher target weight, and a controlled 2x LR. Both candidate
    # runs start from the same proprietary baseline checkpoint.
    diverse_records = list(records)
    for index in range(4):
        source = phase_q.register_source(
            DataSource(
                source_id=f"phase-s-owned-{index}",
                source_type=DataSourceType.USER_OWNED,
                provider="fixture-owner",
                location=f"local://phase-s/{index}",
                license_status="CC0-1.0",
                usage_permissions="USER_OWNED",
                domain=CapabilityDomain.PROGRAMMING,
                trust_level=1.0,
                acquisition_method="LOCAL_FIXTURE",
                eligibility=SourceEligibility.ELIGIBLE,
                eligibility_reason="Owner-approved structurally diverse fixture.",
            )
        )
        phase_q.acquire_text(
            source=source,
            content=f"Owned structural recursion variant {index}. It uses a tree, parity, or counterfactual base case without copying evaluation items.",
            policy=policy,
        )
        diverse_records.append(
            phase_q.source_record(
                record_id=f"phase-s-recursion-{index}",
                target=target,
                source=source,
                instruction=f"Explain a structurally distinct recursive tree traversal variant {index} and identify its termination invariant.",
                input=f"tree-shape-{index}",
                output=f"State a distinct base condition, recurse over child {index}, and combine only after validating the invariant.",
                difficulty=6 + index % 2,
                subdomain="structural-recursion",
            )
        )
    diverse_curriculum = phase_q.build_curriculum(
        target=target,
        requirement=requirement,
        records=tuple(diverse_records),
        tokenizer=tokenizer,
        curriculum_id="curriculum-phase-s-recursion-diverse",
    )
    diverse_release = phase_q.release_training_data(
        release_id="phase-s-recursion-diverse-v1",
        target=target,
        requirement=requirement,
        curriculum=diverse_curriculum,
        records=tuple(diverse_records),
        suite=suite,
    )
    registered_diverse = engine.register_dataset(
        dataset_id="phase-s-recursion-diverse",
        release_dir=tmp_path / "corpora" / "releases" / diverse_release.release_id,
        tokenizer=tokenizer,
        curriculum_roles=("PROGRAMMING",),
    )
    phase_s = PhaseSEngine(tmp_path)
    policy_s = phase_s.configure(
        create_learning_policy(
            "phase-s-e2e-policy",
            maximum_experiments=2,
            maximum_tokens_per_intervention=192,
            maximum_tokens_total=384,
            maximum_adaptive_iterations=1,
        )
    )
    stop_s = create_early_stop_policy(
        "phase-s-e2e-stop", capability_evaluation_available=True
    )
    intervention_one = phase_s.record_intervention(
        create_intervention(
            intervention_id="phase-s-intervention-1",
            parent_model_id=baseline_plan.candidate_id,
            parent_checkpoint_sha256=sha256_file(
                Path(baseline_job.checkpoint_path or "") / "model.safetensors"
            ),
            target_id=target.target_id,
            dataset_release_ids=(targeted_release.release_id,),
            dataset_release_hashes=(registered_targeted.release_manifest_sha256,),
            curriculum_id=curriculum.curriculum_id,
            curriculum_ordering=(
                DifficultyBand.FOUNDATIONAL,
                DifficultyBand.EASY,
                DifficultyBand.MODERATE,
            ),
            mixture_weights={"original_target": 1.0},
            sampling_policy=SamplingPolicy.TARGET_DENSITY,
            dataset_weights={targeted_release.release_id: 1.0},
            target_weight=1.0,
            difficulty_weights={"MODERATE": 1.0},
            learning_rate=candidate_config.learning_rate,
            scheduler=candidate_config.schedule,
            optimizer="adamw",
            batch_size=candidate_config.batch_size,
            sequence_length=model.max_sequence_length,
            gradient_accumulation=candidate_config.gradient_accumulation_steps,
            weight_decay=candidate_config.weight_decay,
            warmup_steps=candidate_config.warmup_steps,
            max_tokens=candidate_job.current_tokens,
            max_steps=candidate_config.steps,
            seed=candidate_config.seed,
            checkpoint_interval=candidate_config.checkpoint_every,
            early_stop_policy=stop_s,
        ),
        policy_s,
    )
    intervention_two = create_intervention(
        intervention_id="phase-s-intervention-2",
        parent_model_id=baseline_plan.candidate_id,
        parent_checkpoint_sha256=sha256_file(
            Path(baseline_job.checkpoint_path or "") / "model.safetensors"
        ),
        target_id=target.target_id,
        dataset_release_ids=(diverse_release.release_id,),
        dataset_release_hashes=(registered_diverse.release_manifest_sha256,),
        curriculum_id=diverse_curriculum.curriculum_id,
        curriculum_ordering=(
            DifficultyBand.FOUNDATIONAL,
            DifficultyBand.MODERATE,
            DifficultyBand.HARD,
            DifficultyBand.COMPOSITIONAL,
        ),
        mixture_weights={"original_target": 0.75, "structural_variant": 0.25},
        sampling_policy=SamplingPolicy.TARGET_DENSITY,
        dataset_weights={
            targeted_release.release_id: 0.75,
            diverse_release.release_id: 0.25,
        },
        target_weight=2.0,
        difficulty_weights={"MODERATE": 0.75, "HARD": 0.25},
        learning_rate=candidate_config.learning_rate * 2,
        scheduler=candidate_config.schedule,
        optimizer="adamw",
        batch_size=candidate_config.batch_size,
        sequence_length=model.max_sequence_length,
        gradient_accumulation=candidate_config.gradient_accumulation_steps,
        weight_decay=candidate_config.weight_decay,
        warmup_steps=candidate_config.warmup_steps,
        max_tokens=192,
        max_steps=12,
        seed=candidate_config.seed + 1,
        checkpoint_interval=1,
        early_stop_policy=stop_s,
    )
    adaptive = adapt_intervention(
        previous=intervention_one,
        next_intervention=intervention_two,
        attribution=learning,
        policy=policy_s,
        iteration=1,
    )
    phase_s.persist("decisions", adaptive.decision_id, adaptive)
    intervention_two = phase_s.record_intervention(intervention_two, policy_s)
    adaptive_config = candidate_config.model_copy(
        update={
            "run_id": "phase-s-recursion-run",
            "learning_rate": intervention_two.learning_rate,
            "seed": intervention_two.seed,
            "dataset_release_id": registered_diverse.release_id,
            "dataset_manifest_sha256": registered_diverse.release_manifest_sha256,
        }
    )
    adaptive_plan = _plan(
        candidate_id="phase-s-candidate",
        experiment_id="phase-s-recursion-exp",
        model=model,
        config=adaptive_config,
        tokenizer=tokenizer,
        tokens=192,
        parent=baseline_plan.experiment_id,
    )
    engine.development.plan_candidate(adaptive_plan)
    adaptive_job = engine.run_job(
        engine.queue_job(
            plan=adaptive_plan,
            dataset_id=registered_diverse.dataset_id,
            tokenizer_path=tokenizer.model_path.parent,
            model_config=model,
            training_config=adaptive_config,
            initial_weights=Path(baseline_job.checkpoint_path or "")
            / "model.safetensors",
        ).job_id
    )
    adaptive_run, adaptive_comparison = engine.evaluate_capability_model_only(
        adaptive_job.job_id, suite, baseline_run=baseline_run, max_new_tokens=4
    )
    adaptive_generalization = phase_p_generalization.evaluate_model_only(
        checkpoint=Path(adaptive_job.checkpoint_path or ""),
        tokenizer=tokenizer,
        model_config=model,
        suite=generalization_suite,
        candidate_id=adaptive_plan.candidate_id,
        max_new_tokens=4,
    )
    adaptive_generalization_assessment = assess_generalization(
        experiment_id="phase-s-recursion-exp",
        baseline=baseline_generalization,
        candidate=adaptive_generalization,
        suite=generalization_suite,
        classes={
            "phase-r-generalization-0": GeneralizationClass.STRUCTURAL,
            "phase-r-generalization-1": GeneralizationClass.COMPOSITIONAL,
            "phase-r-generalization-2": GeneralizationClass.COUNTERFACTUAL,
        },
        policy=create_generalization_policy(
            "phase-s-generalization", require_all_nonempty_classes=False
        ),
    )
    adaptive_learning = classify_learning_evidence(
        experiment_id="phase-s-recursion-exp",
        baseline=baseline_run,
        candidate=adaptive_run,
        comparison=adaptive_comparison,
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=sha256_file(
            Path(baseline_job.checkpoint_path or "") / "model.safetensors"
        )
        != sha256_file(Path(adaptive_job.checkpoint_path or "") / "model.safetensors"),
        training_loss_delta=None,
        held_out_loss_delta=None,
        generalization=adaptive_generalization_assessment,
    )
    phase_r.persist("learning", adaptive_learning.attribution_id, adaptive_learning)
    checkpoint = CheckpointEvidence(
        checkpoint_id=Path(adaptive_job.checkpoint_path or "").name,
        checkpoint_sha256=sha256_file(
            Path(adaptive_job.checkpoint_path or "") / "model.safetensors"
        ),
        target_capability_delta=adaptive_comparison.domain_deltas[
            CapabilityDomain.PROGRAMMING.value
        ],
        generalization_delta=adaptive_generalization_assessment.generalization_score,
        regression_rate=adaptive_comparison.regression_rate,
        training_stable=True,
        tokens_processed=adaptive_job.current_tokens,
    )
    selection = select_checkpoint(
        intervention_id=intervention_two.intervention_id,
        candidates=(checkpoint,),
        maximum_regression_rate=policy_s.maximum_regression_rate,
    )
    phase_s.persist("checkpoints", selection.selection_id, selection)
    adaptive_efficiency = phase_s_efficiency_for(
        intervention_id=intervention_two.intervention_id,
        capability_gain=checkpoint.target_capability_delta,
        generalization_gain=checkpoint.generalization_delta,
        tokens=adaptive_job.current_tokens,
        compute_units=float(adaptive_job.current_tokens * 16),
    )
    phase_s.persist("efficiency", intervention_two.intervention_id, adaptive_efficiency)
    next_training = recommend_next_training(
        intervention_id=intervention_two.intervention_id,
        attribution=adaptive_learning,
        repeated_no_learning=True,
    )
    phase_s.persist("recommendations", next_training.recommendation_id, next_training)
    assert adaptive.action.value == "INCREASE_TARGET_DENSITY"
    assert adaptive_job.current_tokens == 192
    assert adaptive_learning.target_capability_delta == 0.0
    assert adaptive_learning.evidence.value == "NO_LEARNING_EVIDENCE"
    assert (
        selection.selected_checkpoint_id
        == Path(adaptive_job.checkpoint_path or "").name
    )
    assert next_training.action.value == "PAUSE_FOR_REVIEW"
    assert (
        tmp_path / "phase_s" / "decisions" / f"{adaptive.decision_id}.json"
    ).is_file()
    monitor = engine.monitor()
    assert (
        monitor.latest_training_intervention
        and monitor.latest_training_intervention.intervention_id
        == "phase-s-intervention-2"
    )
    assert (
        monitor.next_training_recommendation
        and monitor.next_training_recommendation.action.value == "PAUSE_FOR_REVIEW"
    )
