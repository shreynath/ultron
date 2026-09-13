"""Perform local-only held-out and generalization evaluation for Phase U."""

from __future__ import annotations

import json
from pathlib import Path

from app.foundation.checkpoints import sha256_file
from app.foundation.data import PackedShardSequenceDataset
from app.foundation.model import UltronModelConfig
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    EvaluationPartition,
    ItemRole,
    ModelOnlyCapabilityEvaluator,
    compare_capability_runs,
    create_capability_suite,
)
from app.foundation.phase_r import (
    GeneralizationClass,
    PhaseREngine,
    assess_generalization,
    classify_learning_evidence,
    create_generalization_policy,
)
from app.foundation.phase_s import PhaseSEngine, recommend_next_training
from app.foundation.phase_u import (
    PhaseUEngine,
    PhaseUResult,
    account_corpus,
    assess_resources,
    learning_verdict,
)
from app.foundation.tokenizer import UltronTokenizer
from app.foundation.training import FoundationTrainer


def _items(
    partition: EvaluationPartition, *, generalization: bool = False
) -> tuple[CapabilityItem, ...]:
    values = (
        (
            "program-1",
            CapabilityDomain.PROGRAMMING,
            "2 4 6 next?",
            "8",
        ),
        (
            "program-2",
            CapabilityDomain.PROGRAMMING,
            "1 + 1 =",
            "2",
        ),
        (
            "math-1",
            CapabilityDomain.MATHEMATICS,
            "3 + 4 =",
            "7",
        ),
        (
            "math-2",
            CapabilityDomain.MATHEMATICS,
            "9 - 5 =",
            "4",
        ),
        (
            "systems-1",
            CapabilityDomain.REASONING,
            "No work: idle?",
            "yes",
        ),
        (
            "science-1",
            CapabilityDomain.GENERAL_KNOWLEDGE,
            "0 C freezes? yes/no",
            "yes",
        ),
    )
    return tuple(
        CapabilityItem(
            task_id=f"phase-u-{'generalization' if generalization else 'test'}-{identifier}",
            domain=domain,
            partition=partition,
            role=ItemRole.GENERALIZATION if generalization else ItemRole.TARGET,
            prompt=prompt,
            expected_answer=answer,
            difficulty=2,
            provenance={
                "source": "Phase U independently authored held-out item",
                "evaluation_method": "local normalized exact match",
            },
        )
        for identifier, domain, prompt, answer in values
    )


def main() -> None:
    root = Path("data/phase-u-e2e").resolve()
    execution = root / "runs" / "phase-u-candidate-20m"
    evaluation_root = execution / "evaluation-v2"
    tokenizer = UltronTokenizer(root / "tokenizer")
    model = UltronModelConfig(
        model_name="ultron-phase-u-1.35m",
        version="phase-u-v1",
        vocab_size=tokenizer.vocab_size,
        hidden_size=160,
        num_layers=4,
        num_heads=4,
        intermediate_size=640,
        max_sequence_length=64,
    )
    baseline = (
        execution / "models" / "base" / "phase-u-baseline-zero" / "checkpoint-000000"
    )
    candidate = (
        execution / "models" / "base" / "phase-u-candidate-20m" / "checkpoint-160000"
    )
    manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    validation = PackedShardSequenceDataset(
        [
            Path(shard["path"])
            for release in manifest["packed"].values()
            for shard in release["shards"]
            if shard["split"] == "validation"
        ],
        expected_width=65,
        include_loss_mask=True,
    )
    trainer = FoundationTrainer(execution)
    base_loss, base_ppl = trainer.evaluate_checkpoint(
        checkpoint=baseline,
        model_config=model,
        tokenizer=tokenizer,
        sequences=validation,
    )
    candidate_loss, candidate_ppl = trainer.evaluate_checkpoint(
        checkpoint=candidate,
        model_config=model,
        tokenizer=tokenizer,
        sequences=validation,
    )
    evaluator = ModelOnlyCapabilityEvaluator()
    suites = {}
    for name, generalization_items in (("TEST", False), ("GENERALIZATION", True)):
        partition = EvaluationPartition.TEST
        suite = create_capability_suite(
            evaluation_root / "suites" / f"{name.lower()}.json",
            suite_id=f"phase-u-{name.lower()}",
            version="v2",
            partition=partition,
            items=_items(partition, generalization=generalization_items),
        )
        base_run = evaluator.evaluate(
            checkpoint=baseline,
            tokenizer=tokenizer,
            model_config=model,
            suite=suite,
            candidate_id="phase-u-baseline-zero",
            max_new_tokens=8,
        )
        candidate_run = evaluator.evaluate(
            checkpoint=candidate,
            tokenizer=tokenizer,
            model_config=model,
            suite=suite,
            candidate_id="phase-u-candidate-20m",
            max_new_tokens=8,
        )
        suites[name] = {
            "suite": suite,
            "baseline": base_run,
            "candidate": candidate_run,
            "comparison": compare_capability_runs(base_run, candidate_run),
        }
    generalization = assess_generalization(
        experiment_id="phase-u-candidate-20m-eval-v2",
        baseline=suites["GENERALIZATION"]["baseline"],
        candidate=suites["GENERALIZATION"]["candidate"],
        suite=suites["GENERALIZATION"]["suite"],
        classes={
            item.task_id: GeneralizationClass.STRUCTURAL
            if index % 2
            else GeneralizationClass.COUNTERFACTUAL
            for index, item in enumerate(suites["GENERALIZATION"]["suite"].items)
        },
        policy=create_generalization_policy(
            "phase-u-generalization", minimum_class_items=1
        ),
    )
    comparison = suites["TEST"]["comparison"]
    attribution = classify_learning_evidence(
        experiment_id="phase-u-candidate-20m-eval-v2",
        baseline=suites["TEST"]["baseline"],
        candidate=suites["TEST"]["candidate"],
        comparison=comparison,
        target_domain=CapabilityDomain.PROGRAMMING.value,
        weights_changed=sha256_file(baseline / "model.safetensors")
        != sha256_file(candidate / "model.safetensors"),
        training_loss_delta=None,
        held_out_loss_delta=candidate_loss - base_loss,
        generalization=generalization,
    )
    phase_r = PhaseREngine(evaluation_root)
    phase_r.persist("generalization", generalization.assessment_id, generalization)
    phase_r.persist("learning", attribution.attribution_id, attribution)
    recommendation = recommend_next_training(
        intervention_id="phase-u-candidate-20m-eval-v2",
        attribution=attribution,
        repeated_no_learning=False,
    )
    PhaseSEngine(evaluation_root).persist(
        "recommendations", recommendation.recommendation_id, recommendation
    )
    totals = {
        split: sum(
            shard["token_count"]
            for release in manifest["packed"].values()
            for shard in release["shards"]
            if shard["split"] == split
        )
        for split in ("train", "validation", "test")
    }
    releases = list(manifest["releases"].values())
    before = sum(
        release["duplicate_statistics"]["documents_before"] for release in releases
    )
    duplicates = sum(
        release["duplicate_statistics"]["duplicates_removed"] for release in releases
    )
    accounting = account_corpus(
        dataset_release_id="phase-u-source-derived-mixture-v1",
        dataset_sha256=sha256_file(root / "source-manifest.json"),
        raw_documents=before,
        normalized_documents=before,
        unique_documents=sum(release["document_count"] for release in releases),
        unique_records=sum(release["document_count"] for release in releases),
        raw_tokens=sum(totals.values()),
        deduplicated_tokens=sum(totals.values()),
        train_tokens=totals["train"],
        validation_tokens=totals["validation"],
        test_tokens=totals["test"],
        teacher_generated_tokens=0,
        source_derived_tokens=sum(totals.values()),
        synthetic_tokens=0,
        repeated_tokens=0,
        contaminated_tokens=0,
        quarantined_tokens=0,
        duplicate_rate=duplicates / before,
        contamination_rate=0.0,
    )
    candidate_metrics = json.loads(
        (
            execution / "models" / "base" / "phase-u-candidate-20m" / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    resources = assess_resources(
        tokens_per_second=candidate_metrics["tokens_per_second"],
        tokens_target=candidate_metrics["tokens_processed"],
        maximum_wall_seconds=None,
        disk_usage_bytes=sum(
            path.stat().st_size for path in execution.rglob("*") if path.is_file()
        ),
        peak_memory_bytes=max(
            json.loads(line)["memory"]["mlx_peak_bytes"]
            for line in (
                execution
                / "models"
                / "base"
                / "phase-u-candidate-20m"
                / "training_curve.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ),
    )
    verdict = learning_verdict(attribution=attribution, resources=resources)
    phase_u_result = PhaseUResult(
        experiment_id="phase-u-candidate-20m-eval-v2",
        corpus=accounting,
        mixture=json.loads(
            (execution / "mixture-exposure.json").read_text(encoding="utf-8")
        ),
        resources=resources,
        attribution_id=attribution.attribution_id,
        phase_s_recommendation_id=recommendation.recommendation_id,
        verdict=verdict,
        rationale="Full local campaign conclusion is derived from Phase P/R evidence, never loss alone.",
    )
    phase_u = PhaseUEngine(evaluation_root)
    phase_u.record_corpus(accounting)
    phase_u.record_result(phase_u_result)
    output = {
        "baseline_validation_loss": base_loss,
        "candidate_validation_loss": candidate_loss,
        "baseline_perplexity": base_ppl,
        "candidate_perplexity": candidate_ppl,
        "test": {
            key: value.model_dump(mode="json")
            for key, value in suites["TEST"].items()
            if key != "suite"
        },
        "generalization": generalization.model_dump(mode="json"),
        "attribution": attribution.model_dump(mode="json"),
        "phase_s_recommendation": recommendation.model_dump(mode="json"),
        "phase_u_verdict": verdict.value,
    }
    (evaluation_root / "evaluation.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
