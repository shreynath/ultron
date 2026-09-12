"""Measured local MLX training-scale experiments; no feasibility is guessed."""
from __future__ import annotations

import json
import math
import os
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.model import UltronModelConfig, build_model
from app.foundation.training import ResumableAdamW, TrainingConfig


class Feasibility(str, Enum):
    FEASIBLE = "FEASIBLE"
    FEASIBLE_WITH_MEMORY_PRESSURE = "FEASIBLE_WITH_MEMORY_PRESSURE"
    PRACTICALLY_UNUSABLE = "PRACTICALLY_UNUSABLE"
    OOM = "OOM"
    FAILED = "FAILED"


class Recommendation(str, Enum):
    RECOMMENDED = "RECOMMENDED"
    POSSIBLE = "POSSIBLE"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    PRACTICALLY_UNUSABLE = "PRACTICALLY_UNUSABLE"
    IMPOSSIBLE = "IMPOSSIBLE"


class ResourceRisk(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH_RISK = "HIGH_RISK"
    UNSUPPORTED = "UNSUPPORTED"


class ResourcePreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes_estimate: int
    temporary_tensor_bytes_estimate: int
    checkpoint_bytes_estimate: int
    disk_bytes_estimate: int
    total_bytes_estimate: int
    available_memory_bytes: int | None
    risk: ResourceRisk
    rationale: str


class ScalingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    warmup_steps: int = Field(default=1, ge=0)
    measurement_steps: int = Field(default=3, ge=1)
    measurement_token_budget: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=1, ge=1)
    sequence_length: int = Field(default=32, ge=8)
    seed: int = 17
    minimum_practical_tokens_per_second: float = Field(default=5.0, gt=0)
    swap_pressure_bytes: int = Field(default=512 * 1024 * 1024, ge=0)
    window_steps: int = Field(default=10, ge=1)


class ScalingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_id: str
    created_at: datetime
    measured: bool = True
    model: dict[str, int | str]
    training: dict[str, Any]
    memory: dict[str, Any]
    hardware: dict[str, Any]
    status: Feasibility
    failure: str | None = None


class ScalingRecommendation(BaseModel):
    """A transparent recommendation derived only from persisted measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    parameter_estimate: int
    recommendation: Recommendation
    reason: str
    tokens_per_second: float | None
    tokens_per_hour_estimate: float | None
    peak_mlx_bytes: int | None
    swap_delta_bytes: int | None
    measured: bool


def hardware_snapshot() -> dict[str, str | int | None]:
    result: dict[str, str | int | None] = {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor() or None, "cpu_count": os.cpu_count()}
    if platform.system() == "Darwin":
        for name, command in (("memory_bytes", ["sysctl", "-n", "hw.memsize"]), ("chip", ["sysctl", "-n", "machdep.cpu.brand_string"])):
            try:
                value = subprocess.check_output(command, text=True).strip()
                result[name] = int(value) if name == "memory_bytes" else value
            except (OSError, subprocess.SubprocessError, ValueError):
                result[name] = None
    return result


def swap_used_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        used = output.split("used = ", 1)[1].split("M", 1)[0]
        return int(float(used) * 1024 * 1024)
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None


def process_peak_rss_bytes() -> int | None:
    try:
        # macOS ru_maxrss is bytes; Linux is KiB.
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (ValueError, OSError):
        return None


def candidate_configs(vocab_size: int, context_length: int = 64) -> dict[str, UltronModelConfig]:
    """Valid local candidates; running the larger entries is always explicit."""
    shapes = {
        "100k": (64, 2, 4, 192),
        "1m": (128, 4, 4, 384),
        "8m": (384, 6, 6, 1024),
        "15m": (512, 6, 8, 1280),
        "25m": (576, 8, 8, 1344),
        "50m": (640, 10, 10, 1728),
        "100m": (896, 12, 14, 2432),
        "250m": (1280, 18, 16, 3456),
        "500m": (1600, 24, 20, 4288),
        "1b": (2304, 30, 36, 6144),
    }
    return {name: UltronModelConfig(model_name="ultron-foundation", version=f"scaling-{name}", vocab_size=vocab_size, hidden_size=hidden, num_layers=layers, num_heads=heads, intermediate_size=intermediate, max_sequence_length=context_length) for name, (hidden, layers, heads, intermediate) in shapes.items()}


def resource_preflight(model_config: UltronModelConfig, *, batch_size: int, sequence_length: int, memory_bytes: int | None = None) -> ResourcePreflight:
    """Conservative float32 AdamW estimate; a planning aid, never an OOM claim."""
    parameter_bytes = model_config.parameter_estimate() * 4
    gradient_bytes = parameter_bytes
    optimizer_bytes = parameter_bytes * 2  # Adam first/second moments.
    activation_bytes = batch_size * sequence_length * model_config.hidden_size * model_config.num_layers * 20
    temporary_bytes = max(parameter_bytes // 2, activation_bytes)
    checkpoint_bytes = parameter_bytes * 3  # model + optimizer state, excluding metadata.
    total = parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes + temporary_bytes
    available = memory_bytes if memory_bytes is not None else hardware_snapshot().get("memory_bytes")
    ratio = total / available if isinstance(available, int) and available else None
    if ratio is None:
        risk, rationale = ResourceRisk.CAUTION, "Machine memory was unavailable; run a measured short benchmark first."
    elif ratio <= 0.25:
        risk, rationale = ResourceRisk.SAFE, "Conservative resident estimate is at most 25% of reported machine memory."
    elif ratio <= 0.50:
        risk, rationale = ResourceRisk.CAUTION, "Conservative resident estimate is 25–50% of reported machine memory."
    elif ratio <= 0.75:
        risk, rationale = ResourceRisk.HIGH_RISK, "Conservative resident estimate exceeds 50% of reported machine memory; monitor swap."
    else:
        risk, rationale = ResourceRisk.UNSUPPORTED, "Conservative resident estimate exceeds 75% of reported machine memory."
    return ResourcePreflight(parameter_bytes=parameter_bytes, gradient_bytes=gradient_bytes, optimizer_bytes=optimizer_bytes, activation_bytes_estimate=activation_bytes, temporary_tensor_bytes_estimate=temporary_bytes, checkpoint_bytes_estimate=checkpoint_bytes, disk_bytes_estimate=checkpoint_bytes * 2, total_bytes_estimate=total, available_memory_bytes=available if isinstance(available, int) else None, risk=risk, rationale=rationale)


def recommend(results: list[ScalingResult] | tuple[ScalingResult, ...], *, minimum_tokens_per_second: float = 5.0) -> tuple[ScalingRecommendation, ...]:
    """Classify measurements without extrapolating unmeasured model sizes."""
    recommendations: list[ScalingRecommendation] = []
    feasible = [
        item for item in results
        if item.status is Feasibility.FEASIBLE
        and isinstance(item.training.get("tokens_per_second"), (int, float))
        and item.training["tokens_per_second"] >= minimum_tokens_per_second
    ]
    recommended_size = max((int(item.model["parameter_estimate"]) for item in feasible), default=None)
    for item in sorted(results, key=lambda candidate: int(candidate.model["parameter_estimate"])):
        tps = item.training.get("tokens_per_second")
        throughput = float(tps) if isinstance(tps, (int, float)) else None
        if item.status is Feasibility.OOM:
            classification, reason = Recommendation.IMPOSSIBLE, "The measured run exhausted memory."
        elif item.status is Feasibility.FEASIBLE_WITH_MEMORY_PRESSURE:
            classification, reason = Recommendation.MEMORY_PRESSURE, "The measured run completed with swap pressure."
        elif item.status is Feasibility.PRACTICALLY_UNUSABLE or (throughput is not None and throughput < minimum_tokens_per_second):
            classification, reason = Recommendation.PRACTICALLY_UNUSABLE, "Measured throughput is below the configured practical threshold."
        elif item.status is not Feasibility.FEASIBLE:
            classification, reason = Recommendation.IMPOSSIBLE, item.failure or "The measured run failed."
        elif int(item.model["parameter_estimate"]) == recommended_size:
            classification, reason = Recommendation.RECOMMENDED, "Largest measured configuration that completed without recorded memory pressure."
        else:
            classification, reason = Recommendation.POSSIBLE, "Measured feasible, but a larger measured feasible configuration exists."
        recommendations.append(ScalingRecommendation(parameter_estimate=int(item.model["parameter_estimate"]), recommendation=classification, reason=reason, tokens_per_second=throughput, tokens_per_hour_estimate=throughput * 3600 if throughput is not None else None, peak_mlx_bytes=item.memory.get("mlx_peak_bytes"), swap_delta_bytes=item.memory.get("swap_delta_bytes"), measured=item.measured))
    return tuple(recommendations)


class ScalingHarness:
    def __init__(self, root: Path) -> None:
        self.root = root

    def benchmark(self, model_config: UltronModelConfig, config: ScalingConfig) -> ScalingResult:
        import mlx.core as mx
        from mlx import nn

        experiment_id = f"experiment-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{model_config.parameter_estimate()}"
        started = datetime.now(UTC)
        before_swap, before_rss = swap_used_bytes(), process_peak_rss_bytes()
        peak_swap = before_swap
        mx.reset_peak_memory()
        try:
            model = build_model(model_config)
            optimizer = ResumableAdamW(TrainingConfig(steps=1, batch_size=config.batch_size, warmup_steps=0))
            batch = mx.random.randint(0, model_config.vocab_size, (config.batch_size, config.sequence_length + 1))
            def loss_fn(current, current_batch):
                return mx.mean(nn.losses.cross_entropy(current(current_batch[:, :-1]), current_batch[:, 1:]))
            value_and_grad = nn.value_and_grad(model, loss_fn)
            for _ in range(config.warmup_steps):
                loss, grads = value_and_grad(model, batch)
                optimizer.update(model, grads, 3e-4)
                mx.eval(model.parameters(), optimizer.arrays(), loss)
                observed_swap = swap_used_bytes()
                if observed_swap is not None:
                    peak_swap = max(peak_swap or 0, observed_swap)
            began = time.perf_counter()
            loss = None
            measurement_steps = config.measurement_steps if config.measurement_token_budget is None else max(1, math.ceil(config.measurement_token_budget / (config.batch_size * config.sequence_length)))
            windows: list[dict[str, int | float | None]] = []
            window_started = time.perf_counter()
            window_swap_before = swap_used_bytes()
            losses: list[float] = []
            for index in range(measurement_steps):
                loss, grads = value_and_grad(model, batch)
                optimizer.update(model, grads, 3e-4)
                mx.eval(model.parameters(), optimizer.arrays(), loss)
                losses.append(float(loss))
                observed_swap = swap_used_bytes()
                if observed_swap is not None:
                    peak_swap = max(peak_swap or 0, observed_swap)
                if (index + 1) % config.window_steps == 0 or index + 1 == measurement_steps:
                    window_tokens = ((index + 1) % config.window_steps or config.window_steps) * config.batch_size * config.sequence_length
                    window_elapsed = time.perf_counter() - window_started
                    windows.append({"start_step": index + 2 - (window_tokens // (config.batch_size * config.sequence_length)), "end_step": index + 1, "tokens": window_tokens, "tokens_per_second": window_tokens / window_elapsed if window_elapsed else None, "mean_loss": sum(losses[-(window_tokens // (config.batch_size * config.sequence_length)):]) / (window_tokens // (config.batch_size * config.sequence_length)), "swap_before_bytes": window_swap_before, "swap_after_bytes": observed_swap, "mlx_peak_bytes": int(mx.get_peak_memory())})
                    window_started, window_swap_before = time.perf_counter(), observed_swap
            elapsed = time.perf_counter() - began
            tokens = measurement_steps * config.batch_size * config.sequence_length
            peak_mlx = int(mx.get_peak_memory())
            after_swap = swap_used_bytes()
            swap_delta = after_swap - before_swap if after_swap is not None and before_swap is not None else None
            throughput = tokens / elapsed
            status = Feasibility.FEASIBLE_WITH_MEMORY_PRESSURE if swap_delta is not None and swap_delta > config.swap_pressure_bytes else (Feasibility.PRACTICALLY_UNUSABLE if throughput < config.minimum_practical_tokens_per_second else Feasibility.FEASIBLE)
            result = ScalingResult(experiment_id=experiment_id, created_at=started, model={"parameter_estimate": model_config.parameter_estimate(), "layers": model_config.num_layers, "hidden_size": model_config.hidden_size, "num_heads": model_config.num_heads, "context_length": config.sequence_length, "vocab_size": model_config.vocab_size}, training={"batch_size": config.batch_size, "steps": measurement_steps, "tokens": tokens, "tokens_per_second": throughput, "step_time_seconds": elapsed / measurement_steps, "loss": float(loss), "loss_min": min(losses), "loss_max": max(losses), "measurement_token_budget": config.measurement_token_budget, "windows": windows}, memory={"mlx_peak_bytes": peak_mlx, "process_peak_rss_bytes": process_peak_rss_bytes() or before_rss, "swap_used_before_bytes": before_swap, "swap_used_after_bytes": after_swap, "swap_peak_bytes": peak_swap, "swap_delta_bytes": swap_delta}, hardware=hardware_snapshot(), status=status)
        except (MemoryError, OSError, RuntimeError, ValueError) as exc:
            result = ScalingResult(experiment_id=experiment_id, created_at=started, model={"parameter_estimate": model_config.parameter_estimate(), "layers": model_config.num_layers, "hidden_size": model_config.hidden_size, "num_heads": model_config.num_heads, "context_length": config.sequence_length, "vocab_size": model_config.vocab_size}, training={"batch_size": config.batch_size, "steps": config.measurement_steps, "tokens": 0, "tokens_per_second": None, "step_time_seconds": None, "loss": None}, memory={"mlx_peak_bytes": None, "process_peak_rss_bytes": process_peak_rss_bytes(), "swap_used_before_bytes": before_swap, "swap_used_after_bytes": swap_used_bytes(), "swap_peak_bytes": peak_swap, "swap_delta_bytes": None}, hardware=hardware_snapshot(), status=Feasibility.OOM if "memory" in str(exc).lower() else Feasibility.FAILED, failure=f"{type(exc).__name__}: {exc}")
        target = self.root / "scaling" / experiment_id
        target.mkdir(parents=True, exist_ok=False)
        (target / "experiment.json").write_text(json.dumps({"scaling_config": config.model_dump(mode="json"), "model_config": model_config.model_dump(mode="json")}, indent=2) + "\n", encoding="utf-8")
        (target / "results.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (target / "hardware.json").write_text(json.dumps(result.hardware, indent=2) + "\n", encoding="utf-8")
        (target / "README.md").write_text("MEASURED values come from this short local run. Any future token-budget calculation is an estimate, not a measurement.\n", encoding="utf-8")
        return result
