"""Execute a measured Phase-U native-mixture training campaign through Phase O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.foundation.data import PackedShardSequenceDataset
from app.foundation.development import CandidatePlan, PromotionPolicy
from app.foundation.model import UltronModelConfig
from app.foundation.phase_o import PhaseOEngine
from app.foundation.phase_u import (
    NativeMixtureSequenceDataset,
    create_mixture_configuration,
)
from app.foundation.tokenizer import UltronTokenizer
from app.foundation.training import TrainingConfig

WEIGHTS = {
    "natural-language": 0.15,
    "programming": 0.45,
    "mathematics": 0.08,
    "systems": 0.22,
    "science-engineering": 0.10,
}


def _split(manifest: dict[str, object], domain: str, split: str) -> list[Path]:
    packed = manifest["packed"]
    assert isinstance(packed, dict)
    release = packed[domain]
    assert isinstance(release, dict)
    shards = release["shards"]
    assert isinstance(shards, list)
    return [Path(item["path"]) for item in shards if item["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/phase-u-e2e"))
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--maximum-wall-seconds", type=float, default=None)
    parser.add_argument("--weights-json", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mixture-seed", type=int, default=20260912)
    args = parser.parse_args()
    root = args.root.resolve()
    weights = json.loads(args.weights_json) if args.weights_json else WEIGHTS
    execution_root = root / "runs" / args.run_id
    manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
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
        random_seed=args.seed,
    )
    train = {
        domain: PackedShardSequenceDataset(
            _split(manifest, domain, "train"),
            expected_width=65,
            include_loss_mask=True,
        )
        for domain in weights
    }
    validation = PackedShardSequenceDataset(
        [path for domain in weights for path in _split(manifest, domain, "validation")],
        expected_width=65,
        include_loss_mask=True,
    )
    mixture = create_mixture_configuration(
        f"{args.run_id}-mixture", release_weights=weights, seed=args.mixture_seed
    )
    mixed = NativeMixtureSequenceDataset(
        datasets=train,
        configuration=mixture,
        samples=args.steps * 2,
        tokens_per_sample=64,
    )
    releases = manifest["releases"]
    assert isinstance(releases, dict)
    primary_release = releases.get("general")
    if primary_release is None:
        primary_release = releases["programming"]
    assert isinstance(primary_release, dict)
    release_dir = root / "corpora" / "releases" / primary_release["release_id"]
    engine = PhaseOEngine(execution_root)
    registered = engine.register_dataset(
        dataset_id="phase-u-source-derived-primary",
        release_dir=release_dir,
        tokenizer=tokenizer,
        curriculum_roles=tuple(weights),
    )
    engine.development.configure(
        PromotionPolicy(
            policy_id=f"{args.run_id}-policy",
            minimum_improvement=0.01,
            max_training_tokens_per_experiment=args.steps * 128,
            max_training_tokens_total=args.steps * 128,
            allow_automatic_training_start=True,
        )
    )
    checkpoints = tuple(
        sorted(
            {
                max(1, round(args.steps * fraction))
                for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
            }
        )
    )
    config = TrainingConfig(
        run_id=args.run_id,
        model_id=model.model_name,
        steps=args.steps,
        batch_size=2,
        learning_rate=3e-4,
        warmup_steps=min(1_600, max(1, args.steps // 10)),
        checkpoint_every=args.steps,
        checkpoint_steps=checkpoints,
        validation_every=min(1_600, max(1, args.steps // 10)),
        validation_batches=4,
        maximum_wall_seconds=args.maximum_wall_seconds,
        seed=args.seed,
        tokens_target=args.steps * 128,
        dataset_release_id=registered.release_id,
        dataset_manifest_sha256=registered.release_manifest_sha256,
        mixture_config_sha256=mixture.configuration_sha256,
        mixture_seed=mixture.seed,
        sampling_algorithm=mixture.sampling_algorithm,
        sampling_version=mixture.sampling_version,
    )
    plan = CandidatePlan(
        candidate_id=args.run_id,
        experiment_id=f"{args.run_id}-experiment",
        model_id=model.model_name,
        model_version=model.version,
        dataset_release_id=registered.release_id,
        dataset_manifest_sha256=registered.release_manifest_sha256,
        tokenizer_sha256=tokenizer.sha256,
        training_config=config.model_dump(mode="json"),
        seed=config.seed,
        hardware={"campaign": "phase-u-native-mixture"},
        curriculum_stage="SOURCE_DERIVED_MIXTURE",
        training_token_budget=args.steps * 128,
        reason="Measured Phase U native-mixture proprietary model campaign.",
    )
    engine.development.plan_candidate(plan)
    job = engine.queue_job(
        plan=plan,
        dataset_id=registered.dataset_id,
        tokenizer_path=tokenizer.model_path.parent,
        model_config=model,
        training_config=config,
    )
    completed = engine.run_job(
        job.job_id, train_sequences=mixed, validation_sequences=validation
    )
    (execution_root / "mixture-exposure.json").write_text(
        mixed.exposure.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (execution_root / "sampler-state.json").write_text(
        json.dumps(mixed.sampler_state(completed.current_step * 2), indent=2) + "\n",
        encoding="utf-8",
    )
    print(completed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
