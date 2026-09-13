"""Offline developer CLI for ULTRON foundation research artifacts."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.brain.acquisition import (
    AcquisitionEngine,
    AcquisitionItem,
    AcquisitionJob,
    CuratedSource,
)
from app.foundation.capabilities import (
    Capability,
    CapabilitySuite,
    EvaluationTask,
    TaskKind,
    create_suite,
    evaluate_generator,
)
from app.foundation.checkpoints import CheckpointStore
from app.foundation.data import (
    CorpusDocument,
    CorpusPipeline,
    TokenPacker,
    resolve_artifact_path,
)
from app.foundation.diagnostics import (
    analyze_learning_curve,
    corpus_diagnostics,
    initialization_diagnostics,
    tokenizer_ab_release,
)
from app.foundation.evaluation import EvaluationReport, FoundationEvaluator
from app.foundation.generation import (
    GenerationConfig,
    UltronGenerator,
    degeneration_metrics,
)
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.phase_k import (
    PhaseKRegistry,
    corpus_balance_report,
    parameter_breakdown,
    phase_k_preflight,
    scaling_analysis,
    scaling_ladder,
)
from app.foundation.phase_l import (
    FunctionalFailure,
    OneBFunctionalStatus,
    PhaseLTrainingRecipeRecord,
    PhaseLTrainingRecipeRegistry,
    aggregate_generation_health,
    checkpoint_portability,
    corpus_deficit_plan,
    one_b_functional_gate,
    phase_l_corpus_gate,
    registry_corpus_candidate,
    summarize_sustained_run,
)
from app.foundation.phase_l_diagnostics import (
    branch_evidence,
    compare_branches,
    configuration_audit,
    data_audit,
    objective_sanity,
    packing_audit,
    synthetic_overfit,
    tokenizer_audit,
)
from app.foundation.phase_m import (
    audit_phase_m_release,
    build_phase_m_release,
    phase_m_rung_plan,
    phase_m_source_inventory,
    phase_m_status,
)
from app.foundation.runs import TrainingRunRegistry
from app.foundation.scaling import (
    ScalingConfig,
    ScalingHarness,
    candidate_configs,
    resource_preflight,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


def emit(value: object) -> None:
    print(json_payload(value))


def json_payload(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, default=lambda item: item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item))


def emit_or_write(value: object, output: str | None) -> None:
    """Write a versioned machine-readable experiment artifact when requested."""
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_payload(value) + "\n", encoding="utf-8")
        return
    emit(value)


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_train(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    trainer = FoundationTrainer(Path(args.root))
    result = trainer.train(model_config=model, tokenizer=UltronTokenizer(Path(args.tokenizer)), train_text=Path(args.train_text).read_text(encoding="utf-8"), validation_text=Path(args.validation_text).read_text(encoding="utf-8"), config=TrainingConfig.model_validate(read_json(args.training_config)), resume_checkpoint=Path(args.resume) if args.resume else None)
    emit(result)


def command_train_packed(args: argparse.Namespace) -> None:
    """Train only deterministic packed shards; do not silently cross documents."""
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    config = TrainingConfig.model_validate(read_json(args.training_config))
    manifest = read_json(str(Path(args.tokenized_dir) / "manifest.json"))
    if manifest["release_id"] != config.dataset_release_id:
        raise ValueError("Training config dataset_release_id does not match the packed corpus manifest.")
    train = [resolve_artifact_path(item["path"]) for item in manifest["shards"] if item["split"] == "train"]
    validation = [resolve_artifact_path(item["path"]) for item in manifest["shards"] if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Packed training requires train and validation shards.")
    result = FoundationTrainer(Path(args.root)).train_shards(
        model_config=model,
        tokenizer=UltronTokenizer(Path(args.tokenizer)),
        train_shards=train,
        validation_shards=validation,
        config=config,
        resume_checkpoint=Path(args.resume) if args.resume else None,
    )
    emit(result)


def _packed_training_inputs(args: argparse.Namespace):
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    config = TrainingConfig.model_validate(read_json(args.training_config))
    manifest = read_json(str(Path(args.tokenized_dir) / "manifest.json"))
    if manifest["release_id"] != config.dataset_release_id:
        raise ValueError("Training config dataset_release_id does not match the packed corpus manifest.")
    train = [resolve_artifact_path(item["path"]) for item in manifest["shards"] if item["split"] == "train"]
    validation = [resolve_artifact_path(item["path"]) for item in manifest["shards"] if item["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Packed training requires train and validation shards.")
    return model, config, train, validation


def command_phase_l_train(args: argparse.Namespace) -> None:
    """Run one Phase L duration rung and append its immutable metadata record."""
    model, config, train, validation = _packed_training_inputs(args)
    root = Path(args.root)
    result = FoundationTrainer(root).train_shards(model_config=model, tokenizer=UltronTokenizer(Path(args.tokenizer)), train_shards=train, validation_shards=validation, config=config, resume_checkpoint=Path(args.resume) if args.resume else None)
    run = TrainingRunRegistry(root).get(config.run_id)
    curve = [json.loads(line) for line in (root / "models" / "base" / config.run_id / "training_curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    memory_peak = max((int(item["memory"]["mlx_peak_bytes"]) for item in curve), default=0)
    PhaseLTrainingRecipeRegistry(root).record(PhaseLTrainingRecipeRecord(
        # A resumable run has one stable run_id but can produce several duration
        # rungs.  Evidence must identify the completed rung, rather than collide
        # with the prior immutable record when the run is extended.
        experiment_id=args.experiment_id or f"{config.run_id}-training-{result.steps_completed:06d}",
        evidence_class=config.evidence_class,
        architecture_config=model.model_dump(mode="json"),
        actual_parameter_count=run.model_parameters,
        tokenizer_id=UltronTokenizer(Path(args.tokenizer)).model_path.name,
        tokenizer_vocab_size=model.vocab_size,
        corpus_release_id=config.dataset_release_id,
        corpus_token_count=config.corpus_token_count,
        train_token_count=config.corpus_token_count - config.validation_token_count - config.test_token_count,
        validation_token_count=config.validation_token_count,
        test_token_count=config.test_token_count,
        optimizer="AdamW",
        learning_rate=config.learning_rate,
        scheduler=config.schedule,
        warmup=config.warmup_steps,
        batch_size=config.batch_size,
        sequence_length=model.max_sequence_length,
        gradient_accumulation=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        initialization=model.initialization,
        precision=model.dtype,
        seed=config.seed,
        training_steps=result.steps_completed,
        training_tokens=result.tokens_processed,
        checkpoint_interval=config.checkpoint_every,
        hardware=run.hardware,
        memory_peak=memory_peak or "INSUFFICIENT_EVIDENCE",
        throughput=result.tokens_per_second,
        wall_time=(result.completed_at - result.started_at).total_seconds(),
        train_loss=result.training_loss,
        validation_loss=result.validation_loss,
        parent_experiment=args.parent_experiment,
    ))
    emit(result)


def command_dataset_build(args: argparse.Namespace) -> None:
    documents = [CorpusDocument.model_validate_json(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    emit(CorpusPipeline(Path(args.root)).release(args.release_id, documents))


def command_dataset_inspect(args: argparse.Namespace) -> None:
    emit(read_json(str(Path(args.release_dir) / "manifest.json")))


def command_tokenize(args: argparse.Namespace) -> None:
    emit({"model_path": str(UltronTokenizer.train(Path(args.corpus), Path(args.output_dir), TokenizerConfig(version=args.version, vocab_size=args.vocab_size)).model_path)})


def command_pack(args: argparse.Namespace) -> None:
    emit(TokenPacker(Path(args.root)).pack(Path(args.release_dir), UltronTokenizer(Path(args.tokenizer)), context_length=args.context, packing_strategy=args.packing_strategy))


def command_benchmark(args: argparse.Namespace) -> None:
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    model = candidate_configs(tokenizer.vocab_size, args.context)[args.profile]
    emit(ScalingHarness(Path(args.root)).benchmark(model, ScalingConfig(warmup_steps=args.warmup, measurement_steps=args.steps, batch_size=args.batch, sequence_length=args.context)))


def command_evaluate(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx import nn

    model_config = UltronModelConfig.model_validate(read_json(args.model_config))
    model = build_model(model_config)
    checkpoint = Path(args.checkpoint)
    model.load_weights(str(checkpoint / "model.safetensors"))
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    ids = tokenizer.encode(Path(args.validation_text).read_text(encoding="utf-8"))
    if len(ids) < 2:
        raise ValueError("Validation text has no token targets.")
    batch = mx.array([ids[: model_config.max_sequence_length]])
    loss = float(mx.mean(nn.losses.cross_entropy(model(batch), mx.array([ids[1:]]))))
    emit({"checkpoint": str(checkpoint), "validation_loss": loss, "perplexity": float(mx.exp(mx.array(loss)))})


def command_evaluate_packed(args: argparse.Namespace) -> None:
    """Evaluate a sealed packed split without rebuilding cross-document text."""
    import mlx.core as mx
    from mlx import nn

    model_config = UltronModelConfig.model_validate(read_json(args.model_config))
    model = build_model(model_config)
    checkpoint = Path(args.checkpoint)
    model.load_weights(str(checkpoint / "model.safetensors"))
    manifest = read_json(str(Path(args.tokenized_dir) / "manifest.json"))
    shards = [resolve_artifact_path(item["path"]) for item in manifest["shards"] if item["split"] == args.split]
    from app.foundation.data import PackedShardSequenceDataset
    dataset = PackedShardSequenceDataset(shards, expected_width=model_config.max_sequence_length + 1, include_loss_mask=True)
    if not dataset:
        raise ValueError(f"Packed release has no {args.split} sequences.")
    limit = min(len(dataset), args.max_batches) if args.max_batches else len(dataset)
    losses = []
    for index in range(limit):
        tokens, mask = dataset[index]
        batch, loss_mask = mx.array([tokens]), mx.array([mask])
        per_token = nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:])
        loss = mx.sum(per_token * loss_mask) / mx.maximum(mx.sum(loss_mask), mx.array(1.0))
        mx.eval(loss)
        losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    emit_or_write({"checkpoint": str(checkpoint), "split": args.split, "sequences": len(losses), "loss": mean_loss, "perplexity": float(mx.exp(mx.array(mean_loss))), "mode": "PACKED_SAME_DOCUMENT_EVALUATION"}, args.output)


def command_preflight(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    emit(resource_preflight(model, batch_size=args.batch, sequence_length=args.context))


def command_count_parameters(args: argparse.Namespace) -> None:
    """Instantiate and inspect an ULTRON model; no checkpoint is loaded."""
    emit(parameter_breakdown(UltronModelConfig.model_validate(read_json(args.model_config))))


def command_phase_k_preflight(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    emit(phase_k_preflight(
        model,
        batch_size=args.batch,
        sequence_length=args.context,
        target_tokens=args.target_tokens,
        measured_tokens_per_second=args.tokens_per_second,
        checkpoint_count=args.checkpoint_count,
        instantiate_count=args.instantiate_count,
        disk_path=Path(args.disk_path) if args.disk_path else None,
    ))


def command_corpus_report(args: argparse.Namespace) -> None:
    emit(corpus_balance_report(Path(args.release_dir), Path(args.tokenized_dir), UltronTokenizer(Path(args.tokenizer)), maximum_major_domain_percent=args.maximum_domain_percent))


def command_phase_k_ladder(args: argparse.Namespace) -> None:
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    emit({name: model.model_dump(mode="json") for name, model in scaling_ladder(vocab_size=tokenizer.vocab_size, context_length=args.context).items()})


def command_phase_k_experiments(args: argparse.Namespace) -> None:
    records = PhaseKRegistry(Path(args.root)).records()
    emit(scaling_analysis(records) if args.analyze else records)


def command_phase_l_corpus_gate(args: argparse.Namespace) -> None:
    emit(phase_l_corpus_gate(Path(args.release_dir), Path(args.tokenized_dir), UltronTokenizer(Path(args.tokenizer)), target_tokens=args.target_tokens))


def command_phase_l_corpus_plan(args: argparse.Namespace) -> None:
    emit(corpus_deficit_plan(
        Path(args.release_dir),
        Path(args.tokenized_dir),
        UltronTokenizer(Path(args.tokenizer)),
        formal_target_tokens=args.formal_target_tokens,
        development_target_tokens=args.development_target_tokens,
    ))


def command_phase_l_sustained(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    result = ScalingHarness(Path(args.root)).benchmark(model, ScalingConfig(warmup_steps=args.warmup, measurement_steps=args.steps, batch_size=args.batch, sequence_length=args.context, window_steps=args.window_steps))
    emit(summarize_sustained_run(result))


def command_phase_l_generation_trajectory(args: argparse.Namespace) -> None:
    """Evaluate raw MODEL_ONLY generation at every supplied checkpoint."""
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    prompts = [json.loads(line)["prompt"] for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Generation trajectory needs at least one explicit prompt.")
    records = []
    for value in args.checkpoints.split(","):
        checkpoint = Path(value)
        generator = UltronGenerator(checkpoint, tokenizer, model)
        results = [generator.generate(prompt, GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, seed=args.seed, timeout_seconds=args.timeout)) for prompt in prompts]
        metrics = [degeneration_metrics(result) for result in results]
        numeric = ("average_length", "repeated_unigram_rate", "repeated_bigram_rate", "repetition_3gram", "distinct_1", "distinct_2", "distinct_3", "entropy_bits", "eos_rate")
        aggregate = {name: sum(float(item[name]) for item in metrics if item[name] is not None) / max(1, sum(item[name] is not None for item in metrics)) for name in numeric}
        records.append({"checkpoint_id": generator.manifest.checkpoint_id, "tokens_consumed": generator.manifest.tokens_consumed, "generation": [result.model_dump(mode="json") for result in results], "aggregate": aggregate, "generation_health": aggregate_generation_health(metrics).model_dump(mode="json")})
    emit_or_write({"mode": "MODEL_ONLY_RAW_GENERATION", "evidence_class": "DEVELOPMENT_SCALE", "prompts": len(prompts), "records": records, "note": "Development-scale diagnostic evidence only; no formal Phase L capability claim."}, args.output)


def command_phase_l_status(args: argparse.Namespace) -> None:
    root = Path(args.root)
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    gate = phase_l_corpus_gate(Path(args.release_dir), Path(args.tokenized_dir), tokenizer)
    recipes = PhaseLTrainingRecipeRegistry(root).records()
    final_path = root / "reports" / "phase-l-final-disposition-v1.json"
    final_disposition = read_json(str(final_path)) if final_path.exists() else None
    emit({"phase_l_1": "TRAINING_RECIPE_VALIDATION", "phase_l_2": one_b_functional_gate(recipes), "phase_l_3": "POLICY_ONLY_DO_NOT_EXECUTE", "corpus_gate": gate, "recipe_count": len(recipes), "final_disposition": final_disposition})


def command_phase_l_capability(args: argparse.Namespace) -> None:
    path = Path(args.suite)
    if path.exists():
        suite = CapabilitySuite.model_validate_json(path.read_text(encoding="utf-8"))
    else:
        tasks = (
            EvaluationTask(task_id="math-add", capability=Capability.MATH, kind=TaskKind.EXACT, prompt="What is 2 + 2?", expected="4"),
            EvaluationTask(task_id="reasoning-order", capability=Capability.REASONING, kind=TaskKind.EXACT, prompt="If A is before B and B is before C, what is first?", expected="A"),
            EvaluationTask(task_id="instruction-count", capability=Capability.INSTRUCTION, kind=TaskKind.EXACT, prompt="Reply with exactly three words: red blue green", expected="red blue green"),
            EvaluationTask(task_id="coding-function", capability=Capability.CODING, kind=TaskKind.EXACT, prompt="Write Python code that returns x plus one.", expected="def add_one(x):\n    return x + 1"),
            EvaluationTask(task_id="technical-binary", capability=Capability.COMPUTER_SCIENCE, kind=TaskKind.EXACT, prompt="What algorithm searches a sorted list by halving it?", expected="binary search"),
            EvaluationTask(task_id="language-capital", capability=Capability.LANGUAGE, kind=TaskKind.EXACT, prompt="What is the capital of France?", expected="Paris"),
        )
        suite = create_suite(path, suite_id="phase-l-model-only-basics", version="v1", tasks=tasks, tokenizer_version=UltronTokenizer(Path(args.tokenizer)).version, source_manifest_hash=args.corpus_manifest_sha256, preprocessing_version="phase-l-model-only-v1")
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    generator = UltronGenerator(Path(args.checkpoint), UltronTokenizer(Path(args.tokenizer)), model)
    emit_or_write(evaluate_generator(suite, candidate_id=generator.manifest.checkpoint_id, generator=generator, generation_config=GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, seed=args.seed)), args.output)


def command_phase_l_gate(args: argparse.Namespace) -> None:
    emit(one_b_functional_gate(PhaseLTrainingRecipeRegistry(Path(args.root)).records()))


def command_phase_l_report(args: argparse.Namespace) -> None:
    recipes = PhaseLTrainingRecipeRegistry(Path(args.root)).records()
    emit({"recipes": recipes, "one_b_functional_gate": one_b_functional_gate(recipes)})


def command_phase_l_record_evidence(args: argparse.Namespace) -> None:
    """Append final evaluated evidence without mutating a prior training record."""
    root = Path(args.root)
    config = TrainingConfig.model_validate(read_json(args.training_config))
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    run = TrainingRunRegistry(root).get(config.run_id)
    result = json.loads((root / "models" / "base" / config.run_id / "metrics.json").read_text(encoding="utf-8"))
    curve = [json.loads(line) for line in (root / "models" / "base" / config.run_id / "training_curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test = read_json(args.test_report)
    generation = read_json(args.generation_report)
    capability = read_json(args.capability_report)
    latest_generation = generation["records"][-1]
    record = PhaseLTrainingRecipeRecord(
        experiment_id=args.experiment_id,
        evidence_class=config.evidence_class,
        architecture_config=model.model_dump(mode="json"),
        actual_parameter_count=run.model_parameters,
        tokenizer_id=UltronTokenizer(Path(args.tokenizer)).model_path.name,
        tokenizer_vocab_size=model.vocab_size,
        corpus_release_id=config.dataset_release_id,
        corpus_token_count=config.corpus_token_count,
        train_token_count=config.corpus_token_count - config.validation_token_count - config.test_token_count,
        validation_token_count=config.validation_token_count,
        test_token_count=config.test_token_count,
        optimizer="AdamW", learning_rate=config.learning_rate, scheduler=config.schedule, warmup=config.warmup_steps,
        batch_size=config.batch_size, sequence_length=model.max_sequence_length, gradient_accumulation=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, initialization=model.initialization, precision=model.dtype, seed=config.seed,
        training_steps=result["steps_completed"], training_tokens=result["tokens_processed"], checkpoint_interval=config.checkpoint_every,
        hardware=run.hardware, memory_peak=max(int(item["memory"]["mlx_peak_bytes"]) for item in curve), throughput=result["tokens_per_second"],
        wall_time=(__import__("datetime").datetime.fromisoformat(result["completed_at"]) - __import__("datetime").datetime.fromisoformat(result["started_at"])).total_seconds(),
        train_loss=result["training_loss"], validation_loss=result["validation_loss"], test_loss=test["loss"], generation_metrics=latest_generation,
        capability_metrics={"scores": capability["scores"], "results": capability["results"]}, gate_status=OneBFunctionalStatus.NOT_STARTED,
        failure_reasons=(FunctionalFailure.TRAINING_DURATION_INSUFFICIENT, FunctionalFailure.GENERATION_FAILURE, FunctionalFailure.CAPABILITY_INSUFFICIENT), parent_experiment=config.run_id,
    )
    emit(PhaseLTrainingRecipeRegistry(root).record(record))


def command_phase_l_compare(args: argparse.Namespace) -> None:
    """Compare two immutable evaluation reports from the same versioned suite."""
    baseline = EvaluationReport.model_validate_json(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = EvaluationReport.model_validate_json(Path(args.candidate).read_text(encoding="utf-8"))
    emit(FoundationEvaluator.compare(baseline, candidate))


def command_phase_l_diagnose(args: argparse.Namespace) -> None:
    emit_or_write(configuration_audit(model_config=UltronModelConfig.model_validate(read_json(args.model_config)), training_config=TrainingConfig.model_validate(read_json(args.training_config)), tokenizer=UltronTokenizer(Path(args.tokenizer)), release_dir=Path(args.release_dir), tokenized_dir=Path(args.tokenized_dir), checkpoint=Path(args.checkpoint) if args.checkpoint else None), args.output)


def command_phase_l_diagnose_tokenizer(args: argparse.Namespace) -> None:
    emit_or_write(tokenizer_audit(tokenizer=UltronTokenizer(Path(args.tokenizer)), release_dir=Path(args.release_dir), sample_count=args.sample_count), args.output)


def command_phase_l_diagnose_data(args: argparse.Namespace) -> None:
    emit_or_write(data_audit(release_dir=Path(args.release_dir), tokenized_dir=Path(args.tokenized_dir), tokenizer=UltronTokenizer(Path(args.tokenizer)), split=args.split), args.output)


def command_phase_l_diagnose_packing(args: argparse.Namespace) -> None:
    emit_or_write(packing_audit(release_dir=Path(args.release_dir), tokenized_dir=Path(args.tokenized_dir), tokenizer=UltronTokenizer(Path(args.tokenizer)), split=args.split, sample_limit=args.sample_limit), args.output)


def command_phase_l_diagnose_objective(args: argparse.Namespace) -> None:
    emit_or_write(objective_sanity(), args.output)


def command_phase_l_synthetic_overfit(args: argparse.Namespace) -> None:
    emit_or_write(synthetic_overfit(model_config=UltronModelConfig.model_validate(read_json(args.model_config)), tokenizer=UltronTokenizer(Path(args.tokenizer)), steps=args.steps), args.output)


def command_phase_l_diagnostic_report(args: argparse.Namespace) -> None:
    reports = [read_json(path) for path in args.reports.split(",")]
    emit_or_write({"experiment_id": args.experiment_id, "parent_experiment": args.parent_experiment, "git_revision": "see constituent reports", "result": "PASS" if all(report.get("result") == "PASS" for report in reports) else "FAIL", "reports": reports, "next_action": args.next_action}, args.output)


def command_phase_l_branch_evidence(args: argparse.Namespace) -> None:
    emit_or_write(branch_evidence(branch_id=args.branch_id, parent_experiment=args.parent_experiment, audit=read_json(args.audit), training=read_json(args.training), test=read_json(args.test), generation=read_json(args.generation), capability=read_json(args.capability)), args.output)


def command_phase_l_branch_compare(args: argparse.Namespace) -> None:
    emit_or_write(compare_branches([read_json(path) for path in args.branch_reports.split(",")]), args.output)


def command_checkpoint_portability(args: argparse.Namespace) -> None:
    emit(checkpoint_portability(Path(args.checkpoint)))


def command_phase_l_corpus_candidate(args: argparse.Namespace) -> None:
    tokenizer = UltronTokenizer(Path(args.tokenizer))
    release, tokenized, gate = registry_corpus_candidate(Path(args.root), release_id=args.release_id, tokenizer=tokenizer, context_length=args.context, privacy_action=args.privacy_action)
    emit({"release": release.model_dump(mode="json"), "tokenized": tokenized.model_dump(mode="json"), "gate": gate.model_dump(mode="json")})


def command_phase_l_acquire(args: argparse.Namespace) -> None:
    """Run only explicitly curated, license-evidenced source entries."""
    sources = [CuratedSource.model_validate_json(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({source.source_identifier for source in sources}) != len(sources):
        raise ValueError("Acquisition manifest has duplicate source identifiers.")
    result = asyncio.run(AcquisitionEngine(Path(args.root)).run(AcquisitionJob(items=[AcquisitionItem(source=source) for source in sources], max_retries=args.max_retries)))
    emit(result)


def command_phase_m_build(args: argparse.Namespace) -> None:
    """Publish and audit a Phase M corpus from registered cache entries only."""
    emit(build_phase_m_release(
        Path(args.root), release_id=args.release_id, tokenizer=UltronTokenizer(Path(args.tokenizer)),
        target_tokens=args.target_tokens, context_length=args.context,
    ))


def command_phase_m_audit(args: argparse.Namespace) -> None:
    emit(audit_phase_m_release(
        Path(args.release_dir), Path(args.tokenized_dir), UltronTokenizer(Path(args.tokenizer)),
    ))


def command_phase_m_status(args: argparse.Namespace) -> None:
    emit(phase_m_status(Path(args.root)))


def command_phase_m_inventory(args: argparse.Namespace) -> None:
    emit(phase_m_source_inventory(Path(args.root), UltronTokenizer(Path(args.tokenizer))))


def command_phase_m_rungs(args: argparse.Namespace) -> None:
    development = read_json(str(Path(args.development_tokenized_dir) / "manifest.json"))
    formal = read_json(str(Path(args.formal_tokenized_dir) / "manifest.json")) if args.formal_tokenized_dir else None
    emit(phase_m_rung_plan(
        development_release_id=development["release_id"], development_corpus_tokens=development["token_count"],
        formal_release_id=formal["release_id"] if formal else None,
        formal_corpus_tokens=formal["token_count"] if formal else None,
        parent_checkpoint=args.parent_checkpoint,
    ))


def command_diagnostics(args: argparse.Namespace) -> None:
    emit(analyze_learning_curve(Path(args.curve)))


def command_corpus_diagnostics(args: argparse.Namespace) -> None:
    emit(corpus_diagnostics(Path(args.release_dir), UltronTokenizer(Path(args.tokenizer)), split=args.split))


def command_tokenizer_ab(args: argparse.Namespace) -> None:
    sizes = tuple(int(item) for item in args.vocab_sizes.split(",") if item.strip())
    emit(tokenizer_ab_release(Path(args.release_dir), Path(args.output_root), experiment_id=args.experiment_id, vocab_sizes=sizes, context_length=args.context))


def command_initialization_diagnostics(args: argparse.Namespace) -> None:
    emit(initialization_diagnostics(UltronModelConfig.model_validate(read_json(args.model_config))))


def command_generate(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    generator = UltronGenerator(Path(args.checkpoint), UltronTokenizer(Path(args.tokenizer)), model)
    result = generator.generate(args.prompt, GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed, timeout_seconds=args.timeout))
    print(result.text)
    emit(result)


def command_generate_file(args: argparse.Namespace) -> None:
    model = UltronModelConfig.model_validate(read_json(args.model_config))
    generator = UltronGenerator(Path(args.checkpoint), UltronTokenizer(Path(args.tokenizer)), model)
    config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed, timeout_seconds=args.timeout)
    records = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise TypeError("generate-file input must be JSONL objects containing a string prompt.")
        records.append({"id": item.get("id"), "prompt": item["prompt"], "generation": generator.generate(item["prompt"], config).model_dump(mode="json")})
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train or resume a local random-init model.")
    for name in ("root", "model_config", "training_config", "tokenizer", "train_text", "validation_text"):
        train.add_argument(f"--{name.replace('_', '-')}", required=True)
    train.add_argument("--resume")
    train.set_defaults(func=command_train)
    train_packed = commands.add_parser("train-packed", help="Train from a tokenized release without crossing document boundaries.")
    for name in ("root", "model_config", "training_config", "tokenizer", "tokenized_dir"):
        train_packed.add_argument(f"--{name.replace('_', '-')}", required=True)
    train_packed.add_argument("--resume")
    train_packed.set_defaults(func=command_train_packed)
    phase_l_train = commands.add_parser("phase-l-train", help="Train and immutably register a Phase L recipe from packed shards.")
    for name in ("root", "model_config", "training_config", "tokenizer", "tokenized_dir"):
        phase_l_train.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_train.add_argument("--resume")
    phase_l_train.add_argument("--parent-experiment")
    phase_l_train.add_argument("--experiment-id", help="Optional immutable evidence ID for this completed duration rung.")
    phase_l_train.add_argument("--formal-target-tokens", type=int, default=10_000_000)
    phase_l_train.add_argument("--development-target-tokens", type=int, default=1_000_000)
    phase_l_train.set_defaults(func=command_phase_l_train)
    resume = commands.add_parser("resume", help="Alias for train with --resume.")
    for name in ("root", "model_config", "training_config", "tokenizer", "train_text", "validation_text", "resume"):
        resume.add_argument(f"--{name.replace('_', '-')}", required=True)
    resume.set_defaults(func=command_train)
    inspect = commands.add_parser("checkpoint-inspect")
    inspect.add_argument("checkpoint")
    inspect.set_defaults(func=lambda args: emit(CheckpointStore.inspect(Path(args.checkpoint))))
    build = commands.add_parser("dataset-build")
    build.add_argument("--root", required=True)
    build.add_argument("--release-id", required=True)
    build.add_argument("--input", required=True, help="Explicit CorpusDocument JSONL only.")
    build.set_defaults(func=command_dataset_build)
    dataset = commands.add_parser("dataset-inspect")
    dataset.add_argument("release_dir")
    dataset.set_defaults(func=command_dataset_inspect)
    tokenize = commands.add_parser("tokenize")
    tokenize.add_argument("--corpus", required=True)
    tokenize.add_argument("--output-dir", required=True)
    tokenize.add_argument("--version", default="0.0.1")
    tokenize.add_argument("--vocab-size", type=int, default=320)
    tokenize.set_defaults(func=command_tokenize)
    pack = commands.add_parser("pack")
    pack.add_argument("--root", required=True)
    pack.add_argument("--release-dir", required=True)
    pack.add_argument("--tokenizer", required=True)
    pack.add_argument("--context", type=int, required=True)
    pack.add_argument("--packing-strategy", choices=("FIXED_TERMINAL_EOS_OVERLAP", "VARIABLE_EOS_MASKED", "VARIABLE_EOS_MASKED_V2", "NATURAL_BOUNDARY_EOS_MASKED", "SHORT_NATURAL_EOS_MASKED", "UNPACKED_NATURAL_EOS_MASKED", "DENSE_TEXT_NATURAL_EOS_MASKED", "DENSE_PROSE_NATURAL_EOS_MASKED"), default="FIXED_TERMINAL_EOS_OVERLAP")
    pack.set_defaults(func=command_pack)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--root", required=True)
    benchmark.add_argument("--tokenizer", required=True)
    benchmark.add_argument("--profile", choices=tuple(candidate_configs(320)), default="100k")
    benchmark.add_argument("--context", type=int, default=32)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--steps", type=int, default=3)
    benchmark.add_argument("--batch", type=int, default=1)
    benchmark.set_defaults(func=command_benchmark)
    scaling = commands.add_parser("scaling", help="Alias for benchmark; larger profiles require explicit selection.")
    scaling.add_argument("--root", required=True)
    scaling.add_argument("--tokenizer", required=True)
    scaling.add_argument("--profile", choices=tuple(candidate_configs(320)), default="100k")
    scaling.add_argument("--context", type=int, default=32)
    scaling.add_argument("--warmup", type=int, default=1)
    scaling.add_argument("--steps", type=int, default=3)
    scaling.add_argument("--batch", type=int, default=1)
    scaling.set_defaults(func=command_benchmark)
    evaluate = commands.add_parser("evaluate")
    for name in ("checkpoint", "model_config", "tokenizer", "validation_text"):
        evaluate.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluate.set_defaults(func=command_evaluate)
    evaluate_packed = commands.add_parser("evaluate-packed", help="Evaluate a packed validation or sealed-test split.")
    for name in ("checkpoint", "model_config", "tokenized_dir"):
        evaluate_packed.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluate_packed.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate_packed.add_argument("--max-batches", type=int)
    evaluate_packed.add_argument("--output", help="Write the immutable evaluation artifact to this JSON file.")
    evaluate_packed.set_defaults(func=command_evaluate_packed)
    preflight = commands.add_parser("train-preflight", help="Report a conservative local resource estimate.")
    preflight.add_argument("--model-config", required=True)
    preflight.add_argument("--batch", type=int, default=1)
    preflight.add_argument("--context", type=int, required=True)
    preflight.set_defaults(func=command_preflight)
    count_parameters = commands.add_parser("model-count-parameters", help="Instantiate an ULTRON model and report its exact parameter tree.")
    count_parameters.add_argument("--model-config", required=True)
    count_parameters.set_defaults(func=command_count_parameters)
    phase_k_preflight_parser = commands.add_parser("phase-k-preflight", help="Fail-closed Phase K memory, disk, and approximate-compute preflight.")
    phase_k_preflight_parser.add_argument("--model-config", required=True)
    phase_k_preflight_parser.add_argument("--batch", type=int, default=1)
    phase_k_preflight_parser.add_argument("--context", type=int, required=True)
    phase_k_preflight_parser.add_argument("--target-tokens", type=int)
    phase_k_preflight_parser.add_argument("--tokens-per-second", type=float)
    phase_k_preflight_parser.add_argument("--checkpoint-count", type=int, default=3)
    phase_k_preflight_parser.add_argument("--instantiate-count", action="store_true")
    phase_k_preflight_parser.add_argument("--disk-path")
    phase_k_preflight_parser.set_defaults(func=command_phase_k_preflight)
    phase_l_gate = commands.add_parser("phase-l-corpus-gate")
    phase_l_gate.add_argument("--release-dir", required=True)
    phase_l_gate.add_argument("--tokenized-dir", required=True)
    phase_l_gate.add_argument("--tokenizer", required=True)
    phase_l_gate.add_argument("--target-tokens", type=int, default=10_000_000)
    phase_l_gate.set_defaults(func=command_phase_l_corpus_gate)
    phase_l_status = commands.add_parser("phase-l-status", help="Report Phase L.1/L.2 status without changing evidence.")
    for name in ("root", "release_dir", "tokenized_dir", "tokenizer"):
        phase_l_status.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_status.set_defaults(func=command_phase_l_status)
    phase_l_plan = commands.add_parser("phase-l-corpus-plan", help="Report token-realized Phase L corpus deficits and targeted acquisition cohorts.")
    phase_l_plan.add_argument("--release-dir", required=True)
    phase_l_plan.add_argument("--tokenized-dir", required=True)
    phase_l_plan.add_argument("--tokenizer", required=True)
    phase_l_plan.add_argument("--formal-target-tokens", type=int, default=10_000_000)
    phase_l_plan.add_argument("--development-target-tokens", type=int, default=1_000_000)
    phase_l_plan.set_defaults(func=command_phase_l_corpus_plan)
    phase_l_candidate = commands.add_parser("phase-l-corpus-candidate", help="Build an immutable candidate from deduplicated acquired-source cache entries.")
    phase_l_candidate.add_argument("--root", required=True)
    phase_l_candidate.add_argument("--release-id", required=True)
    phase_l_candidate.add_argument("--tokenizer", required=True)
    phase_l_candidate.add_argument("--context", type=int, default=512)
    phase_l_candidate.add_argument("--privacy-action", choices=("REJECT", "REDACT"), default="REJECT")
    phase_l_candidate.set_defaults(func=command_phase_l_corpus_candidate)
    phase_l_acquire = commands.add_parser("phase-l-acquire", help="Acquire an explicit CuratedSource JSONL manifest through the existing provenance cache.")
    phase_l_acquire.add_argument("--root", required=True)
    phase_l_acquire.add_argument("--manifest", required=True)
    phase_l_acquire.add_argument("--max-retries", type=int, default=2)
    phase_l_acquire.set_defaults(func=command_phase_l_acquire)
    phase_m_build = commands.add_parser("phase-m-build", help="Build and fully audit an immutable registered-source Phase M release.")
    phase_m_build.add_argument("--root", required=True)
    phase_m_build.add_argument("--release-id", required=True)
    phase_m_build.add_argument("--tokenizer", required=True)
    phase_m_build.add_argument("--target-tokens", type=int, choices=(5_000_000, 10_000_000), required=True)
    phase_m_build.add_argument("--context", type=int, default=512)
    phase_m_build.set_defaults(func=command_phase_m_build)
    phase_m_audit = commands.add_parser("phase-m-audit", help="Audit an immutable Phase M release and packed shards.")
    for name in ("release_dir", "tokenized_dir", "tokenizer"):
        phase_m_audit.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_m_audit.set_defaults(func=command_phase_m_audit)
    phase_m_status_parser = commands.add_parser("phase-m-status", help="Report Phase M release and scaling readiness without changing evidence.")
    phase_m_status_parser.add_argument("--root", required=True)
    phase_m_status_parser.set_defaults(func=command_phase_m_status)
    phase_m_inventory = commands.add_parser("phase-m-inventory", help="Measure source-pool scale and emit every inclusion/exclusion decision.")
    phase_m_inventory.add_argument("--root", required=True)
    phase_m_inventory.add_argument("--tokenizer", required=True)
    phase_m_inventory.set_defaults(func=command_phase_m_inventory)
    phase_m_rungs = commands.add_parser("phase-m-rungs", help="Write the controlled 1M/5M/10M exposure ladder without executing it.")
    phase_m_rungs.add_argument("--development-tokenized-dir", required=True)
    phase_m_rungs.add_argument("--formal-tokenized-dir")
    phase_m_rungs.add_argument("--parent-checkpoint")
    phase_m_rungs.set_defaults(func=command_phase_m_rungs)
    phase_l_sustained = commands.add_parser("phase-l-sustained")
    for name in ("root", "model_config"):
        phase_l_sustained.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_sustained.add_argument("--context", type=int, default=512)
    phase_l_sustained.add_argument("--steps", type=int, default=100)
    phase_l_sustained.add_argument("--warmup", type=int, default=1)
    phase_l_sustained.add_argument("--window-steps", type=int, default=10)
    phase_l_sustained.add_argument("--batch", type=int, default=1)
    phase_l_sustained.set_defaults(func=command_phase_l_sustained)
    phase_l_trajectory = commands.add_parser("phase-l-generation-trajectory", help="Measure raw model-only generation across Phase L checkpoints.")
    for name in ("model_config", "tokenizer", "prompts", "checkpoints"):
        phase_l_trajectory.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_trajectory.add_argument("--max-new-tokens", type=int, default=64)
    phase_l_trajectory.add_argument("--seed", type=int, default=17)
    phase_l_trajectory.add_argument("--timeout", type=float, default=30.0)
    phase_l_trajectory.add_argument("--output", help="Write the immutable generation artifact to this JSON file.")
    phase_l_trajectory.set_defaults(func=command_phase_l_generation_trajectory)
    phase_l_generate = commands.add_parser("phase-l-generate", help="Alias for fixed prompt-set Phase L generation trajectory.")
    for name in ("model_config", "tokenizer", "prompts", "checkpoints"):
        phase_l_generate.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_generate.add_argument("--max-new-tokens", type=int, default=64)
    phase_l_generate.add_argument("--seed", type=int, default=17)
    phase_l_generate.add_argument("--timeout", type=float, default=30.0)
    phase_l_generate.add_argument("--output", help="Write the immutable generation artifact to this JSON file.")
    phase_l_generate.set_defaults(func=command_phase_l_generation_trajectory)
    phase_l_evaluate = commands.add_parser("phase-l-evaluate", help="Evaluate an immutable packed validation or test split.")
    for name in ("checkpoint", "model_config", "tokenized_dir"):
        phase_l_evaluate.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    phase_l_evaluate.add_argument("--max-batches", type=int)
    phase_l_evaluate.add_argument("--output", help="Write the immutable evaluation artifact to this JSON file.")
    phase_l_evaluate.set_defaults(func=command_evaluate_packed)
    phase_l_capability = commands.add_parser("phase-l-capability", help="Run the fixed, model-only capability suite.")
    for name in ("checkpoint", "model_config", "tokenizer", "suite", "corpus_manifest_sha256"):
        phase_l_capability.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_capability.add_argument("--max-new-tokens", type=int, default=64)
    phase_l_capability.add_argument("--seed", type=int, default=17)
    phase_l_capability.add_argument("--output", help="Write the immutable capability artifact to this JSON file.")
    phase_l_capability.set_defaults(func=command_phase_l_capability)
    phase_l_gate_status = commands.add_parser("phase-l-gate", help="Evaluate the fail-closed 1B functional-foundation gate.")
    phase_l_gate_status.add_argument("--root", required=True)
    phase_l_gate_status.set_defaults(func=command_phase_l_gate)
    phase_l_report = commands.add_parser("phase-l-report", help="Serialize immutable Phase L recipe and 1B gate evidence.")
    phase_l_report.add_argument("--root", required=True)
    phase_l_report.set_defaults(func=command_phase_l_report)
    phase_l_record = commands.add_parser("phase-l-record-evidence", help="Append immutable evaluated Phase L recipe evidence.")
    for name in ("root", "experiment_id", "model_config", "training_config", "tokenizer", "test_report", "generation_report", "capability_report"):
        phase_l_record.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_record.set_defaults(func=command_phase_l_record_evidence)
    phase_l_compare = commands.add_parser("phase-l-compare", help="Compare two same-suite immutable Phase L evaluation reports.")
    phase_l_compare.add_argument("--baseline", required=True)
    phase_l_compare.add_argument("--candidate", required=True)
    phase_l_compare.set_defaults(func=command_phase_l_compare)
    phase_l_diagnose = commands.add_parser("phase-l-diagnose", help="Audit active Phase L configuration against runtime artifacts.")
    for name in ("model_config", "training_config", "tokenizer", "release_dir", "tokenized_dir"):
        phase_l_diagnose.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_diagnose.add_argument("--checkpoint")
    phase_l_diagnose.add_argument("--output")
    phase_l_diagnose.set_defaults(func=command_phase_l_diagnose)
    phase_l_diagnose_tokenizer = commands.add_parser("phase-l-diagnose-tokenizer", help="Validate tokenizer identity, specials, punctuation, and round trips.")
    for name in ("tokenizer", "release_dir"):
        phase_l_diagnose_tokenizer.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_diagnose_tokenizer.add_argument("--sample-count", type=int, default=3)
    phase_l_diagnose_tokenizer.add_argument("--output")
    phase_l_diagnose_tokenizer.set_defaults(func=command_phase_l_diagnose_tokenizer)
    phase_l_diagnose_data = commands.add_parser("phase-l-diagnose-data", help="Measure actual corpus and packed EOS/token distributions.")
    for name in ("tokenizer", "release_dir", "tokenized_dir"):
        phase_l_diagnose_data.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_diagnose_data.add_argument("--split", choices=("train", "validation", "test"), default="train")
    phase_l_diagnose_data.add_argument("--output")
    phase_l_diagnose_data.set_defaults(func=command_phase_l_diagnose_data)
    phase_l_diagnose_eos = commands.add_parser("phase-l-diagnose-eos", help="Alias for EOS frequency and packed-position diagnostics.")
    for name in ("tokenizer", "release_dir", "tokenized_dir"):
        phase_l_diagnose_eos.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_diagnose_eos.add_argument("--split", choices=("train", "validation", "test"), default="train")
    phase_l_diagnose_eos.add_argument("--output")
    phase_l_diagnose_eos.set_defaults(func=command_phase_l_diagnose_data)
    phase_l_diagnose_packing = commands.add_parser("phase-l-diagnose-packing", help="Verify packed target alignment, masks, and EOS positions.")
    for name in ("tokenizer", "release_dir", "tokenized_dir"):
        phase_l_diagnose_packing.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_diagnose_packing.add_argument("--split", choices=("train", "validation", "test"), default="train")
    phase_l_diagnose_packing.add_argument("--sample-limit", type=int, default=32)
    phase_l_diagnose_packing.add_argument("--output")
    phase_l_diagnose_packing.set_defaults(func=command_phase_l_diagnose_packing)
    phase_l_diagnose_objective = commands.add_parser("phase-l-diagnose-objective", help="Numerically verify next-token cross-entropy and masking semantics.")
    phase_l_diagnose_objective.add_argument("--output")
    phase_l_diagnose_objective.set_defaults(func=command_phase_l_diagnose_objective)
    phase_l_synthetic = commands.add_parser("phase-l-synthetic-overfit", help="Run bounded production-pipeline synthetic overfit validation.")
    for name in ("model_config", "tokenizer"):
        phase_l_synthetic.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_synthetic.add_argument("--steps", type=int, default=80)
    phase_l_synthetic.add_argument("--output")
    phase_l_synthetic.set_defaults(func=command_phase_l_synthetic_overfit)
    phase_l_smoke = commands.add_parser("phase-l-smoke-train", help="Alias for a bounded packed training smoke run.")
    for name in ("root", "model_config", "training_config", "tokenizer", "tokenized_dir"):
        phase_l_smoke.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_smoke.add_argument("--resume")
    phase_l_smoke.set_defaults(func=command_train_packed)
    phase_l_recovery = commands.add_parser("phase-l-recovery", help="Run a versioned Phase L recovery recipe and register its rung.")
    for name in ("root", "model_config", "training_config", "tokenizer", "tokenized_dir"):
        phase_l_recovery.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_recovery.add_argument("--resume")
    phase_l_recovery.add_argument("--parent-experiment")
    phase_l_recovery.add_argument("--experiment-id")
    phase_l_recovery.add_argument("--formal-target-tokens", type=int, default=10_000_000)
    phase_l_recovery.add_argument("--development-target-tokens", type=int, default=1_000_000)
    phase_l_recovery.set_defaults(func=command_phase_l_train)
    phase_l_progress = commands.add_parser("phase-l-progress", help="Summarize an ordered set of Phase L diagnostic artifacts.")
    phase_l_progress.add_argument("--reports", required=True, help="Comma-separated diagnostic JSON report paths.")
    phase_l_progress.add_argument("--experiment-id", required=True)
    phase_l_progress.add_argument("--parent-experiment")
    phase_l_progress.add_argument("--next-action", required=True)
    phase_l_progress.add_argument("--output")
    phase_l_progress.set_defaults(func=command_phase_l_diagnostic_report)
    phase_l_report_diagnostic = commands.add_parser("phase-l-diagnostic-report", help="Alias for phase-l-progress with explicit durable report output.")
    phase_l_report_diagnostic.add_argument("--reports", required=True)
    phase_l_report_diagnostic.add_argument("--experiment-id", required=True)
    phase_l_report_diagnostic.add_argument("--parent-experiment")
    phase_l_report_diagnostic.add_argument("--next-action", required=True)
    phase_l_report_diagnostic.add_argument("--output")
    phase_l_report_diagnostic.set_defaults(func=command_phase_l_diagnostic_report)
    phase_l_generation_sanity = commands.add_parser("phase-l-generation-sanity", help="Alias for deterministic raw model-only generation trajectory checks.")
    for name in ("model_config", "tokenizer", "prompts", "checkpoints"):
        phase_l_generation_sanity.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_generation_sanity.add_argument("--max-new-tokens", type=int, default=64)
    phase_l_generation_sanity.add_argument("--seed", type=int, default=17)
    phase_l_generation_sanity.add_argument("--timeout", type=float, default=30.0)
    phase_l_generation_sanity.add_argument("--output")
    phase_l_generation_sanity.set_defaults(func=command_phase_l_generation_trajectory)
    phase_l_branch_evidence = commands.add_parser("phase-l-branch-evidence", help="Assemble auditable matched-branch evidence from frozen reports.")
    for name in ("branch_id", "audit", "training", "test", "generation", "capability"):
        phase_l_branch_evidence.add_argument(f"--{name.replace('_', '-')}", required=True)
    phase_l_branch_evidence.add_argument("--parent-experiment")
    phase_l_branch_evidence.add_argument("--output")
    phase_l_branch_evidence.set_defaults(func=command_phase_l_branch_evidence)
    phase_l_branch_compare = commands.add_parser("phase-l-branch-compare", help="Rank matched branches by functional trajectory before loss.")
    phase_l_branch_compare.add_argument("--branch-reports", required=True, help="Comma-separated branch-evidence JSON paths.")
    phase_l_branch_compare.add_argument("--output")
    phase_l_branch_compare.set_defaults(func=command_phase_l_branch_compare)
    portability = commands.add_parser("checkpoint-portability")
    portability.add_argument("checkpoint")
    portability.set_defaults(func=command_checkpoint_portability)
    corpus_report = commands.add_parser("corpus-report", help="Report realized Phase K token/domain composition and isolation.")
    corpus_report.add_argument("--release-dir", required=True)
    corpus_report.add_argument("--tokenized-dir", required=True)
    corpus_report.add_argument("--tokenizer", required=True)
    corpus_report.add_argument("--maximum-domain-percent", type=float, default=0.45)
    corpus_report.set_defaults(func=command_corpus_report)
    phase_k_ladder_parser = commands.add_parser("phase-k-ladder", help="Derive Phase K parameter-scale candidate configurations.")
    phase_k_ladder_parser.add_argument("--tokenizer", required=True)
    phase_k_ladder_parser.add_argument("--context", type=int, default=512)
    phase_k_ladder_parser.set_defaults(func=command_phase_k_ladder)
    phase_k_experiments = commands.add_parser("phase-k-experiments", help="List or analyze append-only Phase K experiments.")
    phase_k_experiments.add_argument("--root", required=True)
    phase_k_experiments.add_argument("--analyze", action="store_true")
    phase_k_experiments.set_defaults(func=command_phase_k_experiments)
    diagnostics = commands.add_parser("diagnostics", help="Analyze an append-only training curve.")
    diagnostics.add_argument("--curve", required=True)
    diagnostics.set_defaults(func=command_diagnostics)
    corpus_diagnostic = commands.add_parser("corpus-diagnostics", help="Measure one immutable corpus split.")
    corpus_diagnostic.add_argument("--release-dir", required=True)
    corpus_diagnostic.add_argument("--tokenizer", required=True)
    corpus_diagnostic.add_argument("--split", choices=("train", "validation", "test"), default="train")
    corpus_diagnostic.set_defaults(func=command_corpus_diagnostics)
    tokenizer_ab = commands.add_parser("tokenizer-ab", help="Train development-split-only ULTRON tokenizer candidates.")
    tokenizer_ab.add_argument("--release-dir", required=True)
    tokenizer_ab.add_argument("--output-root", required=True)
    tokenizer_ab.add_argument("--experiment-id", required=True)
    tokenizer_ab.add_argument("--vocab-sizes", default="320,512,1024,2048,4096,8192")
    tokenizer_ab.add_argument("--context", type=int, default=128)
    tokenizer_ab.set_defaults(func=command_tokenizer_ab)
    init_diagnostic = commands.add_parser("initialization-diagnostics", help="Measure a fresh random ULTRON initialization.")
    init_diagnostic.add_argument("--model-config", required=True)
    init_diagnostic.set_defaults(func=command_initialization_diagnostics)
    generate = commands.add_parser("generate", help="Generate with an ULTRON from-scratch checkpoint only.")
    for name in ("checkpoint", "model_config", "tokenizer", "prompt"):
        generate.add_argument(f"--{name.replace('_', '-')}", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--top-p", type=float)
    generate.add_argument("--seed", type=int, default=17)
    generate.add_argument("--timeout", type=float, default=30.0)
    generate.set_defaults(func=command_generate)
    generate_file = commands.add_parser("generate-file", help="Run JSONL evaluation prompts through an ULTRON checkpoint.")
    for name in ("checkpoint", "model_config", "tokenizer", "input"):
        generate_file.add_argument(f"--{name.replace('_', '-')}", required=True)
    generate_file.add_argument("--output")
    generate_file.add_argument("--max-new-tokens", type=int, default=64)
    generate_file.add_argument("--temperature", type=float, default=0.0)
    generate_file.add_argument("--top-k", type=int)
    generate_file.add_argument("--top-p", type=float)
    generate_file.add_argument("--seed", type=int, default=17)
    generate_file.add_argument("--timeout", type=float, default=30.0)
    generate_file.set_defaults(func=command_generate_file)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
