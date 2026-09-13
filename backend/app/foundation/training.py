"""Deterministic local pretraining with complete, inspectable checkpoints."""

from __future__ import annotations

import ast
import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.foundation.checkpoints import (
    CheckpointManifest,
    CheckpointStore,
    canonical_hash,
    runtime_metadata,
)
from app.foundation.data import PackedShardSequenceDataset
from app.foundation.model import UltronModelConfig, build_model
from app.foundation.runs import (
    CorpusTier,
    DevelopmentStatus,
    EvidenceClass,
    ExperimentTier,
    FormalGateStatus,
    RunStatus,
    TrainingRun,
    TrainingRunRegistry,
)
from app.foundation.tokenizer import UltronTokenizer


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = "ultron-tiny-0.0.1"
    model_id: str = "ultron-foundation-0.0.1"
    steps: int = Field(default=100, ge=1, description="Absolute target optimizer step.")
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    gradient_clip_norm: float | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    epsilon: float = Field(default=1e-8, gt=0)
    warmup_steps: int = Field(default=10, ge=0)
    schedule: str = "linear_warmup_constant"
    checkpoint_every: int = Field(default=50, ge=1)
    checkpoint_steps: tuple[int, ...] = ()
    validation_every: int = Field(default=25, ge=1)
    validation_batches: int = Field(
        default=1,
        ge=1,
        description="Deterministic held-out batches per validation point.",
    )
    early_stopping_patience: int | None = Field(
        default=None,
        ge=1,
        description="Validation evaluations without a material improvement before preserving and stopping a pilot.",
    )
    early_stopping_min_delta: float = Field(default=0.0, ge=0)
    maximum_wall_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Hard local wall-clock boundary checked between optimizer steps; a complete checkpoint is saved before stopping.",
    )
    seed: int = 17
    tokens_target: int | None = Field(default=None, ge=1)
    dataset_release_id: str = "unreleased-development-text"
    dataset_manifest_sha256: str = "unreleased-development-text"
    corpus_tier: CorpusTier = CorpusTier.INFRASTRUCTURE
    experiment_tier: ExperimentTier = ExperimentTier.INFRASTRUCTURE
    evidence_class: EvidenceClass = EvidenceClass.INFRASTRUCTURE_ONLY
    formal_gate_status: FormalGateStatus = FormalGateStatus.NOT_APPLICABLE
    development_status: DevelopmentStatus = DevelopmentStatus.NOT_APPLICABLE
    corpus_token_count: int = Field(default=0, ge=0)
    validation_token_count: int = Field(default=0, ge=0)
    test_token_count: int = Field(default=0, ge=0)
    mixture_config_sha256: str | None = None
    mixture_seed: int | None = None
    sampling_algorithm: str | None = None
    sampling_version: str | None = None

    def model_post_init(self, __context: Any, /) -> None:
        expected = {
            ExperimentTier.INFRASTRUCTURE: EvidenceClass.INFRASTRUCTURE_ONLY,
            ExperimentTier.DEVELOPMENT: EvidenceClass.DEVELOPMENT_SCALE,
            ExperimentTier.FORMAL: EvidenceClass.FORMAL_PHASE_L,
        }[self.experiment_tier]
        if self.evidence_class is not expected:
            raise ValueError("experiment_tier and evidence_class must agree.")
        if self.corpus_tier.value != self.experiment_tier.value:
            raise ValueError("corpus_tier and experiment_tier must agree for training.")
        if self.experiment_tier is ExperimentTier.FORMAL:
            if self.formal_gate_status is not FormalGateStatus.READY:
                raise ValueError(
                    "Formal Phase L training requires a READY 10M corpus gate."
                )
            if self.corpus_token_count < 10_000_000:
                raise ValueError(
                    "Formal Phase L training requires at least 10,000,000 tokenizer tokens."
                )
        if self.experiment_tier is ExperimentTier.DEVELOPMENT:
            if self.development_status is not DevelopmentStatus.READY:
                raise ValueError(
                    "Development-scale training requires a READY development corpus."
                )
            if self.corpus_token_count < 1_000_000:
                raise ValueError(
                    "Development-scale training requires at least 1,000,000 tokenizer tokens."
                )


class TrainingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    started_at: datetime
    completed_at: datetime
    steps_completed: int
    tokens_processed: int
    training_loss: float
    validation_loss: float
    perplexity: float
    checkpoint_path: str
    tokens_per_second: float
    maximum_gradient_norm: float | None = None
    gradient_norm_p50: float | None = None
    gradient_norm_p95: float | None = None
    clipping_frequency: float = 0.0
    early_stop_reason: str | None = None
    tokens_target: int | None = None
    tokens_remaining: int | None = None


class LinearWarmupConstant:
    """Stateful linear-warmup constant/cosine schedules.

    ``linear_warmup_constant`` remains the default for backwards-compatible
    Phase H experiments.  The cosine variant is a bounded Phase I pilot option,
    not an implicit optimizer change.
    """

    def __init__(self, config: TrainingConfig, *, step: int = 0) -> None:
        self.config, self.step = config, step

    @property
    def learning_rate(self) -> float:
        if self.config.schedule not in {
            "linear_warmup_constant",
            "linear_warmup_cosine",
        }:
            raise ValueError(
                "schedule must be linear_warmup_constant or linear_warmup_cosine"
            )
        warmup = (
            1.0
            if not self.config.warmup_steps
            else min(1.0, self.step / self.config.warmup_steps)
        )
        if (
            self.config.schedule == "linear_warmup_constant"
            or self.step <= self.config.warmup_steps
        ):
            return self.config.learning_rate * warmup
        progress = (self.step - self.config.warmup_steps) / max(
            1, self.config.steps - self.config.warmup_steps
        )
        return (
            self.config.learning_rate
            * 0.5
            * (1.0 + math.cos(math.pi * min(1.0, progress)))
        )

    def advance(self) -> float:
        self.step += 1
        return self.learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.config.schedule,
            "step": self.step,
            "warmup_steps": self.config.warmup_steps,
            "base_learning_rate": self.config.learning_rate,
        }

    @classmethod
    def from_state(
        cls, config: TrainingConfig, state: dict[str, Any]
    ) -> LinearWarmupConstant:
        if (
            state.get("kind") != config.schedule
            or state.get("warmup_steps") != config.warmup_steps
            or state.get("base_learning_rate") != config.learning_rate
        ):
            raise ValueError(
                "Checkpoint scheduler does not match training configuration."
            )
        return cls(config, step=int(state["step"]))


class ResumableAdamW:
    """Minimal AdamW with a portable tensor state instead of opaque optimizer state."""

    def __init__(
        self, config: TrainingConfig, state: dict[str, Any] | None = None
    ) -> None:
        self.config = config
        self.step = int((state or {}).get("step", 0))
        self.m: dict[str, Any] = {}
        self.v: dict[str, Any] = {}

    def update(self, model: Any, gradients: Any, learning_rate: float) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten

        parameters = dict(tree_flatten(model.parameters()))
        updated: list[tuple[str, Any]] = []
        next_step = self.step + 1
        for name, grad in tree_flatten(gradients):
            current = parameters[name]
            previous_m = self.m.setdefault(name, mx.zeros_like(grad))
            previous_v = self.v.setdefault(name, mx.zeros_like(grad))
            moment = self.config.beta1 * previous_m + (1.0 - self.config.beta1) * grad
            variance = self.config.beta2 * previous_v + (
                1.0 - self.config.beta2
            ) * mx.square(grad)
            self.m[name], self.v[name] = moment, variance
            corrected_m = moment / (1.0 - self.config.beta1**next_step)
            corrected_v = variance / (1.0 - self.config.beta2**next_step)
            decayed = current * (1.0 - learning_rate * self.config.weight_decay)
            updated.append(
                (
                    name,
                    decayed
                    - learning_rate
                    * corrected_m
                    / (mx.sqrt(corrected_v) + self.config.epsilon),
                )
            )
        model.update(tree_unflatten(updated))
        self.step = next_step

    def arrays(self) -> dict[str, Any]:
        return {
            **{f"m.{name}": value for name, value in self.m.items()},
            **{f"v.{name}": value for name, value in self.v.items()},
        }

    def load_arrays(self, arrays: dict[str, Any]) -> None:
        self.m = {
            name.removeprefix("m."): value
            for name, value in arrays.items()
            if name.startswith("m.")
        }
        self.v = {
            name.removeprefix("v."): value
            for name, value in arrays.items()
            if name.startswith("v.")
        }
        if not self.m or set(self.m) != set(self.v):
            raise ValueError("Checkpoint optimizer state is incomplete.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "adamw",
            "step": self.step,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "epsilon": self.config.epsilon,
            "weight_decay": self.config.weight_decay,
        }


def _hardware() -> dict[str, str | int | float | None]:
    machine: dict[str, str | int | float | None] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "system": platform.system(),
    }
    if platform.system() == "Darwin":
        try:
            machine["memory_bytes"] = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True
                ).strip()
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            machine["memory_bytes"] = None
    return machine


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python_random": repr(random.getstate()),
        "numpy_random": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "mlx_random": "UNAVAILABLE: MLX exposes seed() but no readable/restorable generator state. This trainer uses no MLX RNG after deterministic model construction.",
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(ast.literal_eval(state["python_random"]))
    numpy = state["numpy_random"]
    np.random.set_state(
        (
            numpy["name"],
            np.asarray(numpy["keys"], dtype=np.uint32),
            numpy["pos"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


class FoundationTrainer:
    """Trains only fresh ULTRON models; no foreign model or tokenizer route exists."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _samples(ids: list[int], *, context: int, batch_size: int, offset: int):
        import mlx.core as mx

        width = context + 1
        starts = [
            (offset + item * context) % (len(ids) - width) for item in range(batch_size)
        ]
        return mx.array([ids[start : start + width] for start in starts])

    @staticmethod
    def _gradient_norm(gradients: Any) -> float:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        squared = sum(
            (mx.sum(mx.square(value)) for _, value in tree_flatten(gradients)),
            mx.array(0.0),
        )
        return float(mx.sqrt(squared))

    def _load_checkpoint(
        self, path: Path, model: Any, config: TrainingConfig
    ) -> tuple[
        CheckpointManifest, ResumableAdamW, LinearWarmupConstant, dict[str, Any]
    ]:
        import mlx.core as mx

        manifest = CheckpointStore.inspect(path)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError(
                "Legacy weight-only checkpoints are loadable for inference but cannot resume exactly."
            )
        if (
            manifest.training_run_id != config.run_id
            or manifest.model_id != config.model_id
        ):
            raise ValueError(
                "Checkpoint does not belong to this training run/model identity."
            )
        checkpoint_config = dict(manifest.training_config)
        requested_config = config.model_dump(mode="json")
        # The target step and checkpoint cadence may change when continuing a
        # run; optimizer/data-affecting settings may not.
        for mutable in (
            "steps",
            "tokens_target",
            "checkpoint_every",
            "checkpoint_steps",
            "validation_every",
            "validation_batches",
            "early_stopping_patience",
            "maximum_wall_seconds",
        ):
            checkpoint_config.pop(mutable, None)
            requested_config.pop(mutable, None)
        if checkpoint_config != requested_config:
            raise ValueError(
                "Checkpoint training configuration differs from requested resume configuration."
            )
        model.load_weights(str(path / "model.safetensors"))
        trainer_state = json_load(path / "trainer_state.json")
        optimizer = ResumableAdamW(config, trainer_state["optimizer"])
        optimizer.load_arrays(mx.load(str(path / "optimizer.safetensors")))
        scheduler = LinearWarmupConstant.from_state(
            config, json_load(path / "scheduler.json")
        )
        _restore_rng(trainer_state["rng"])
        return manifest, optimizer, scheduler, trainer_state

    def train(
        self,
        *,
        model_config: UltronModelConfig,
        tokenizer: UltronTokenizer,
        config: TrainingConfig,
        train_text: str = "",
        validation_text: str = "",
        train_sequences: Sequence[list[int] | tuple[list[int], list[int]]]
        | None = None,
        validation_sequences: Sequence[list[int] | tuple[list[int], list[int]]]
        | None = None,
        resume_weights: Path | None = None,
        resume_checkpoint: Path | None = None,
        parent_checkpoint: str | None = None,
    ) -> TrainingResult:
        import mlx.core as mx
        from mlx import nn

        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary must exactly match model configuration."
            )
        train_ids, validation_ids = (
            tokenizer.encode(train_text),
            tokenizer.encode(validation_text),
        )
        packed = train_sequences is not None or validation_sequences is not None
        if packed:
            if not train_sequences or not validation_sequences:
                raise ValueError(
                    "Packed training requires both train and validation sequences."
                )
            expected_width = model_config.max_sequence_length + 1
            for sequences in (train_sequences, validation_sequences):
                if any(
                    len(sequence[0] if isinstance(sequence, tuple) else sequence)
                    != expected_width
                    for sequence in sequences
                ):
                    raise ValueError(
                        "Packed sequences must exactly match context length plus one target token."
                    )
        elif (
            min(len(train_ids), len(validation_ids))
            <= model_config.max_sequence_length + 1
        ):
            raise ValueError(
                "Training and validation corpora need more than one context window."
            )
        random.seed(config.seed)
        np.random.seed(config.seed)
        model = build_model(model_config)
        run_dir = self.root / "models" / "base" / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        model_record = run_dir / "model_config.json"
        model_serialized = (
            json.dumps(model_config.model_dump(mode="json"), sort_keys=True, indent=2)
            + "\n"
        )
        if (
            model_record.exists()
            and model_record.read_text(encoding="utf-8") != model_serialized
        ):
            raise ValueError(
                "Training run model metadata differs; choose a new run ID."
            )
        if not model_record.exists():
            model_record.write_text(model_serialized, encoding="utf-8")
        # Preserve the initial configuration. Resume may extend only target and
        # cadence fields, which are checked separately by ``_load_checkpoint``.
        training_record = run_dir / "training_config.json"
        if not training_record.exists():
            training_record.write_text(
                json.dumps(config.model_dump(mode="json"), sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
        from mlx.utils import tree_flatten

        model_parameters = sum(
            int(value.size) for _, value in tree_flatten(model.parameters())
        )
        registry = TrainingRunRegistry(self.root)
        try:
            registry.get(config.run_id)
        except KeyError:
            registry.create(
                TrainingRun(
                    run_id=config.run_id,
                    model_id=config.model_id,
                    model_version=model_config.version,
                    dataset_release=config.dataset_release_id,
                    tokenizer_version=tokenizer.version,
                    training_config=config.model_dump(mode="json"),
                    corpus_id=config.dataset_release_id,
                    corpus_tier=config.corpus_tier,
                    token_count=config.corpus_token_count,
                    evidence_class=config.evidence_class,
                    experiment_tier=config.experiment_tier,
                    formal_gate_status=config.formal_gate_status,
                    development_status=config.development_status,
                    model_parameters=model_parameters,
                    validation_tokens=config.validation_token_count,
                    test_tokens=config.test_token_count,
                    hardware=_hardware(),
                    runtime=runtime_metadata(),
                )
            )
        registry.update(config.run_id, status=RunStatus.RUNNING)

        if resume_checkpoint and resume_weights:
            raise ValueError(
                "Use either a resumable checkpoint or initial weights, never both."
            )
        start_step, checkpoint_parent = 0, parent_checkpoint
        if resume_checkpoint:
            manifest, optimizer, scheduler, _ = self._load_checkpoint(
                resume_checkpoint, model, config
            )
            start_step, checkpoint_parent, tokens_processed, metrics = (
                manifest.global_step,
                manifest.checkpoint_id,
                manifest.tokens_consumed,
                manifest.metrics,
            )
        else:
            if resume_weights:
                model.load_weights(str(resume_weights))
            optimizer, scheduler, tokens_processed, metrics = (
                ResumableAdamW(config),
                LinearWarmupConstant(config),
                0,
                {},
            )
        if start_step >= config.steps:
            raise ValueError(
                "Requested training target must be greater than checkpoint step."
            )
        mx.eval(model.parameters())

        def loss_fn(current_model, sample):
            batch, loss_mask = sample
            losses = nn.losses.cross_entropy(current_model(batch[:, :-1]), batch[:, 1:])
            if loss_mask is None:
                return mx.mean(losses)
            return mx.sum(losses * loss_mask) / mx.maximum(
                mx.sum(loss_mask), mx.array(1.0)
            )

        def batch_for(
            sequences: Sequence[list[int] | tuple[list[int], list[int]]] | None,
            ids: list[int],
            *,
            offset: int,
        ):
            if sequences is None:
                return self._samples(
                    ids,
                    context=model_config.max_sequence_length,
                    batch_size=config.batch_size,
                    offset=offset,
                ), None
            selected = [
                sequences[(offset + index) % len(sequences)]
                for index in range(config.batch_size)
            ]
            tokens = [item[0] if isinstance(item, tuple) else item for item in selected]
            masks = [
                item[1] if isinstance(item, tuple) else [1] * (len(item) - 1)
                for item in selected
            ]
            return mx.array(tokens), mx.array(masks)

        value_and_grad = nn.value_and_grad(model, loss_fn)
        started_at, started = datetime.now(UTC), time.perf_counter()
        curve_path = run_dir / "training_curve.jsonl"
        final_loss, validation_loss, maximum_gradient_norm = (
            float(metrics.get("train_loss") or 0),
            float(metrics.get("validation_loss") or 0),
            0.0,
        )
        gradient_norms: list[float] = []
        clipped_steps = 0
        best_validation = float("inf")
        validations_without_improvement = 0
        early_stop_reason: str | None = None
        completed_steps = start_step
        store = CheckpointStore(self.root)
        try:
            for step in range(start_step + 1, config.steps + 1):
                from mlx.utils import tree_flatten, tree_unflatten

                step_started = time.perf_counter()
                losses = []
                accumulated: dict[str, Any] = {}
                for microbatch in range(config.gradient_accumulation_steps):
                    batch = batch_for(
                        train_sequences,
                        train_ids,
                        offset=(
                            (step - 1) * config.gradient_accumulation_steps + microbatch
                        )
                        * config.batch_size,
                    )
                    loss, micro_grads = value_and_grad(model, batch)
                    losses.append(loss)
                    for name, value in tree_flatten(micro_grads):
                        accumulated[name] = (
                            accumulated[name] + value if name in accumulated else value
                        )
                grads = tree_unflatten(
                    [
                        (name, value / config.gradient_accumulation_steps)
                        for name, value in accumulated.items()
                    ]
                )
                final_loss_array = sum(losses) / config.gradient_accumulation_steps
                gradient_norm = self._gradient_norm(grads)
                if not math.isfinite(gradient_norm) or not math.isfinite(
                    float(final_loss_array)
                ):
                    raise FloatingPointError(
                        "Non-finite training loss or gradient norm."
                    )
                clipped = False
                if (
                    config.gradient_clip_norm is not None
                    and gradient_norm > config.gradient_clip_norm
                ):
                    scale = config.gradient_clip_norm / gradient_norm
                    grads = tree_unflatten(
                        [(name, value * scale) for name, value in tree_flatten(grads)]
                    )
                    clipped = True
                learning_rate = scheduler.advance()
                optimizer.update(model, grads, learning_rate)
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                gradient_norms.append(gradient_norm)
                clipped_steps += int(clipped)
                mx.eval(model.parameters(), optimizer.arrays(), final_loss_array)
                final_loss = float(final_loss_array)
                tokens_processed += (
                    config.batch_size
                    * model_config.max_sequence_length
                    * config.gradient_accumulation_steps
                )
                if step % config.validation_every == 0 or step == config.steps:
                    values = []
                    for validation_index in range(config.validation_batches):
                        validation = batch_for(
                            validation_sequences,
                            validation_ids,
                            offset=validation_index * config.batch_size,
                        )
                        values.append(float(loss_fn(model, validation)))
                    validation_loss = sum(values) / len(values)
                    validation_for_curve: float | None = validation_loss
                    if (
                        validation_loss
                        < best_validation - config.early_stopping_min_delta
                    ):
                        best_validation, validations_without_improvement = (
                            validation_loss,
                            0,
                        )
                    else:
                        validations_without_improvement += 1
                else:
                    validation_for_curve = None
                curve = {
                    "step": step,
                    "tokens_seen": tokens_processed,
                    "train_loss": final_loss,
                    "validation_loss": validation_for_curve,
                    "validation_batches": config.validation_batches
                    if validation_for_curve is not None
                    else None,
                    "perplexity": math.exp(min(validation_for_curve, 20.0))
                    if validation_for_curve is not None
                    else None,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "maximum_gradient_norm": maximum_gradient_norm,
                    "gradient_clipped": clipped,
                    "parameter_update_norm_proxy": gradient_norm * learning_rate,
                    "tokens_per_second": config.batch_size
                    * model_config.max_sequence_length
                    * config.gradient_accumulation_steps
                    / (time.perf_counter() - step_started),
                    "step_time": time.perf_counter() - step_started,
                    "memory": {
                        "mlx_active_bytes": int(mx.get_active_memory()),
                        "mlx_peak_bytes": int(mx.get_peak_memory()),
                    },
                }
                with curve_path.open("a", encoding="utf-8") as curve_file:
                    curve_file.write(json.dumps(curve, sort_keys=True) + "\n")
                should_early_stop = bool(
                    validation_for_curve is not None
                    and config.early_stopping_patience is not None
                    and validations_without_improvement
                    >= config.early_stopping_patience
                )
                wall_clock_reached = (
                    config.maximum_wall_seconds is not None
                    and time.perf_counter() - started >= config.maximum_wall_seconds
                )
                if (
                    step % config.checkpoint_every == 0
                    or step in config.checkpoint_steps
                    or step == config.steps
                    or should_early_stop
                    or wall_clock_reached
                ):
                    checkpoint_id = f"checkpoint-{step:06d}"
                    state = {
                        "global_step": step,
                        "epoch": 0,
                        "tokens_consumed": tokens_processed,
                        "dataset_position": {"shard": 0, "batch": step},
                        "optimizer": optimizer.state_dict(),
                        "rng": _rng_state(),
                        "metrics": {
                            "train_loss": final_loss,
                            "validation_loss": validation_loss,
                            "perplexity": math.exp(min(validation_loss, 20.0)),
                            "learning_rate": learning_rate,
                            "gradient_norm": maximum_gradient_norm,
                        },
                    }
                    manifest = CheckpointManifest(
                        checkpoint_id=checkpoint_id,
                        model_id=config.model_id,
                        model_version=model_config.version,
                        architecture_config_sha256=canonical_hash(
                            model_config.model_dump(mode="json")
                        ),
                        tokenizer_id=tokenizer.model_path.stem,
                        tokenizer_version=tokenizer.version,
                        tokenizer_sha256=tokenizer.sha256,
                        dataset_release_id=config.dataset_release_id,
                        dataset_manifest_sha256=config.dataset_manifest_sha256,
                        training_run_id=config.run_id,
                        parent_checkpoint=checkpoint_parent,
                        global_step=step,
                        epoch=0,
                        tokens_consumed=tokens_processed,
                        dataset_position=state["dataset_position"],
                        metrics=state["metrics"],
                        training_config=config.model_dump(mode="json"),
                        optimizer_config=optimizer.state_dict(),
                        scheduler_config=scheduler.state_dict(),
                        runtime=runtime_metadata(),
                        hardware=_hardware(),
                        rng={
                            "mlx_random": "UNAVAILABLE: no MLX getter",
                            "python_numpy": "serialized in trainer_state.json",
                        },
                        lineage={
                            "origin": "ULTRON_FROM_SCRATCH",
                            "initialization": model_config.initialization,
                            "pretrained_weights": model_config.pretrained_weights,
                            "parent_model": model_config.parent_model,
                        },
                    )
                    latest = store.save(
                        directory=run_dir / checkpoint_id,
                        model=model,
                        optimizer_arrays=optimizer.arrays(),
                        scheduler=scheduler.state_dict(),
                        trainer_state=state,
                        manifest=manifest,
                    )
                    checkpoint_parent = checkpoint_id
                completed_steps = step
                if should_early_stop:
                    early_stop_reason = f"NO_VALIDATION_IMPROVEMENT:{validations_without_improvement}_EVALUATIONS"
                    break
                if wall_clock_reached:
                    early_stop_reason = (
                        f"WALL_CLOCK_LIMIT:{config.maximum_wall_seconds:.6f}_SECONDS"
                    )
                    break
        except Exception:
            registry.update(
                config.run_id,
                status=RunStatus.FAILED,
                end_time=datetime.now(UTC),
                steps=optimizer.step,
                tokens=tokens_processed,
                training_tokens=tokens_processed,
            )
            raise
        elapsed = time.perf_counter() - started
        ordered_norms = sorted(gradient_norms)
        p50 = ordered_norms[len(ordered_norms) // 2] if ordered_norms else None
        p95 = (
            ordered_norms[
                min(len(ordered_norms) - 1, math.ceil(len(ordered_norms) * 0.95) - 1)
            ]
            if ordered_norms
            else None
        )
        result = TrainingResult(
            run_id=config.run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            steps_completed=completed_steps,
            tokens_processed=tokens_processed,
            training_loss=final_loss,
            validation_loss=validation_loss,
            perplexity=math.exp(min(validation_loss, 20.0)),
            checkpoint_path=str(latest),
            tokens_per_second=(
                tokens_processed
                - start_step
                * config.batch_size
                * model_config.max_sequence_length
                * config.gradient_accumulation_steps
            )
            / elapsed,
            maximum_gradient_norm=maximum_gradient_norm,
            gradient_norm_p50=p50,
            gradient_norm_p95=p95,
            clipping_frequency=clipped_steps / max(1, len(gradient_norms)),
            early_stop_reason=early_stop_reason,
            tokens_target=config.tokens_target,
            tokens_remaining=max(0, config.tokens_target - tokens_processed)
            if config.tokens_target
            else None,
        )
        (run_dir / "metrics.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        registry.update(
            config.run_id,
            status=RunStatus.COMPLETED,
            end_time=result.completed_at,
            steps=result.steps_completed,
            tokens=result.tokens_processed,
            training_tokens=result.tokens_processed,
            train_loss=result.training_loss,
            validation_loss=result.validation_loss,
            perplexity=result.perplexity,
            tokens_per_second=result.tokens_per_second,
        )
        return result

    def evaluate_checkpoint(
        self,
        *,
        checkpoint: Path,
        model_config: UltronModelConfig,
        tokenizer: UltronTokenizer,
        sequences: Sequence[list[int] | tuple[list[int], list[int]]],
        batch_size: int = 2,
        batches: int = 4,
    ) -> tuple[float, float]:
        """Evaluate a complete local checkpoint without performing an update."""
        import mlx.core as mx
        from mlx import nn

        manifest = CheckpointStore.inspect(checkpoint)
        if not isinstance(manifest, CheckpointManifest):
            raise TypeError("Evaluation requires a complete local checkpoint.")
        if tokenizer.vocab_size != model_config.vocab_size or not sequences:
            raise ValueError(
                "Checkpoint evaluation needs matching model, tokenizer, and data."
            )
        model = build_model(model_config)
        model.load_weights(str(checkpoint / "model.safetensors"))
        mx.eval(model.parameters())
        values: list[float] = []
        for offset in range(batches):
            selected = [
                sequences[(offset * batch_size + index) % len(sequences)]
                for index in range(batch_size)
            ]
            tokens = [item[0] if isinstance(item, tuple) else item for item in selected]
            masks = [
                item[1] if isinstance(item, tuple) else [1] * (len(item) - 1)
                for item in selected
            ]
            batch, mask = mx.array(tokens), mx.array(masks)
            losses = nn.losses.cross_entropy(model(batch[:, :-1]), batch[:, 1:])
            values.append(
                float(mx.sum(losses * mask) / mx.maximum(mx.sum(mask), mx.array(1.0)))
            )
        validation_loss = sum(values) / len(values)
        return validation_loss, math.exp(min(validation_loss, 20.0))

    def initialize_from_scratch(
        self,
        *,
        model_config: UltronModelConfig,
        tokenizer: UltronTokenizer,
        config: TrainingConfig,
    ) -> Path:
        """Persist an evaluable, true random-initialization checkpoint at step zero."""
        import mlx.core as mx

        if tokenizer.vocab_size != model_config.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary must exactly match model configuration."
            )
        if (
            model_config.initialization != "RANDOM"
            or model_config.pretrained_weights != "NONE"
            or model_config.parent_model != "NONE"
        ):
            raise ValueError("A zero-step baseline must be a fresh proprietary model.")
        random.seed(config.seed)
        np.random.seed(config.seed)
        mx.random.seed(config.seed)
        model = build_model(model_config)
        mx.eval(model.parameters())
        run_dir = self.root / "models" / "base" / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        model_record = run_dir / "model_config.json"
        serialized = (
            json.dumps(model_config.model_dump(mode="json"), sort_keys=True, indent=2)
            + "\n"
        )
        if (
            model_record.exists()
            and model_record.read_text(encoding="utf-8") != serialized
        ):
            raise ValueError(
                "Training run model metadata differs; choose a new run ID."
            )
        if not model_record.exists():
            model_record.write_text(serialized, encoding="utf-8")
        checkpoint = run_dir / "checkpoint-000000"
        if checkpoint.exists():
            existing = CheckpointStore.inspect(checkpoint)
            if (
                isinstance(existing, CheckpointManifest)
                and existing.training_run_id == config.run_id
            ):
                return checkpoint
            raise ValueError(
                "Zero-step checkpoint already exists with another identity."
            )
        optimizer = ResumableAdamW(config)
        scheduler = LinearWarmupConstant(config)
        state = {
            "global_step": 0,
            "epoch": 0,
            "tokens_consumed": 0,
            "dataset_position": {"shard": 0, "batch": 0},
            "optimizer": optimizer.state_dict(),
            "rng": _rng_state(),
            "metrics": {
                "train_loss": None,
                "validation_loss": None,
                "perplexity": None,
                "learning_rate": scheduler.learning_rate,
                "gradient_norm": None,
            },
        }
        manifest = CheckpointManifest(
            checkpoint_id="checkpoint-000000",
            model_id=config.model_id,
            model_version=model_config.version,
            architecture_config_sha256=canonical_hash(
                model_config.model_dump(mode="json")
            ),
            tokenizer_id=tokenizer.model_path.stem,
            tokenizer_version=tokenizer.version,
            tokenizer_sha256=tokenizer.sha256,
            dataset_release_id=config.dataset_release_id,
            dataset_manifest_sha256=config.dataset_manifest_sha256,
            training_run_id=config.run_id,
            parent_checkpoint=None,
            global_step=0,
            epoch=0,
            tokens_consumed=0,
            dataset_position=state["dataset_position"],
            metrics=state["metrics"],
            training_config=config.model_dump(mode="json"),
            optimizer_config=optimizer.state_dict(),
            scheduler_config=scheduler.state_dict(),
            runtime=runtime_metadata(),
            hardware=_hardware(),
            rng={
                "mlx_random": "seeded at construction",
                "python_numpy": "serialized in trainer_state.json",
            },
            lineage={
                "origin": "ULTRON_FROM_SCRATCH",
                "initialization": model_config.initialization,
                "pretrained_weights": model_config.pretrained_weights,
                "parent_model": model_config.parent_model,
            },
        )
        return CheckpointStore(self.root).save(
            directory=checkpoint,
            model=model,
            optimizer_arrays=optimizer.arrays(),
            scheduler=scheduler.state_dict(),
            trainer_state=state,
            manifest=manifest,
        )

    def train_shards(
        self,
        *,
        model_config: UltronModelConfig,
        tokenizer: UltronTokenizer,
        train_shards: list[Path],
        validation_shards: list[Path],
        config: TrainingConfig,
        resume_checkpoint: Path | None = None,
    ) -> TrainingResult:
        """Train from deterministic JSONL packed shards without crossing document boundaries."""
        width = model_config.max_sequence_length + 1
        return self.train(
            model_config=model_config,
            tokenizer=tokenizer,
            train_sequences=PackedShardSequenceDataset(
                train_shards, expected_width=width, include_loss_mask=True
            ),
            validation_sequences=PackedShardSequenceDataset(
                validation_shards, expected_width=width, include_loss_mask=True
            ),
            config=config,
            resume_checkpoint=resume_checkpoint,
        )
