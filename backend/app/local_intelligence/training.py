"""Controlled local LoRA training and evaluation orchestration.

The runner starts no training automatically.  It executes MLX-LM's local CLI
against a versioned curated dataset, records the exact command/configuration,
and registers an adapter only after the process produced files on disk.
"""
from __future__ import annotations

import asyncio
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.local_intelligence.core import (
    LocalModelError,
    LocalModelRegistry,
    ModelRecord,
    ModelStatus,
    ensure_local_layout,
    sha256_tree,
)


class TrainingRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    base_model_id: str
    dataset_path: str
    adapter_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: str = "created"
    command: tuple[str, ...]
    return_code: int | None = None
    output_tail: str = ""
    hardware: str = Field(default_factory=lambda: f"{platform.machine()} {platform.platform()}")


class LocalTrainingRunner:
    def __init__(self, registry: LocalModelRegistry | None = None, root: Path | None = None) -> None:
        self.root = ensure_local_layout(root)
        self.registry = registry or LocalModelRegistry(self.root)

    def plan(self, *, base_model_id: str, dataset_path: Path, adapter_version: str) -> TrainingRun:
        base = self.registry.get(base_model_id)
        if base is None:
            raise LocalModelError("Base model is not registered.")
        if base.status not in {ModelStatus.BASE, ModelStatus.PROMOTED, ModelStatus.ARCHIVED}:
            raise LocalModelError("Training requires a registered immutable base/promoted model.")
        dataset = dataset_path.resolve()
        try:
            dataset.relative_to(self.root / "learning" / "curated")
        except ValueError as exc:
            raise LocalModelError("Training data must be a curated dataset under ULTRON_HOME.") from exc
        if not dataset.is_file():
            raise LocalModelError("Curated training dataset does not exist.")
        adapter = self.root / "models" / "adapters" / adapter_version
        if adapter.exists():
            raise LocalModelError("Adapter versions are immutable; choose a new version.")
        # MLX-LM accepts a local model directory and a local data directory.
        # Its runner performs no remote lookup when both are absolute paths.
        command = (
            os.environ.get("PYTHON", "python3"), "-m", "mlx_lm.lora",
            "--model", base.location, "--train", "--data", str(dataset.parent),
            "--adapter-path", str(adapter),
        )
        return TrainingRun(base_model_id=base_model_id, dataset_path=str(dataset), adapter_path=str(adapter), command=command)

    async def run(self, plan: TrainingRun) -> TrainingRun:
        if plan.status != "created":
            raise LocalModelError("A training plan can only run once.")
        try:
            result = await asyncio.to_thread(subprocess.run, list(plan.command), capture_output=True, text=True, timeout=86_400, check=False)
        except FileNotFoundError as exc:
            raise LocalModelError("Local Python/MLX training runtime is unavailable.") from exc
        except subprocess.TimeoutExpired as exc:
            return plan.model_copy(update={"status": "failed", "completed_at": datetime.now(UTC), "output_tail": (exc.stderr or "")[-4_000:]})
        finished = plan.model_copy(update={"status": "completed" if result.returncode == 0 and Path(plan.adapter_path).exists() else "failed", "completed_at": datetime.now(UTC), "return_code": result.returncode, "output_tail": (result.stdout + "\n" + result.stderr)[-4_000:]})
        self._write_run(finished)
        return finished

    async def register_candidate(self, run: TrainingRun, *, model_id: str, version: str) -> ModelRecord:
        if run.status != "completed":
            raise LocalModelError("Only an actually completed local training run can register a candidate.")
        base = self.registry.get(run.base_model_id)
        assert base is not None
        adapter = Path(run.adapter_path)
        if not adapter.exists():
            raise LocalModelError("Training reported success but produced no adapter artifact.")
        return await self.registry.register(ModelRecord(
            model_id=model_id, base_model=base.model_id, version=version,
            location=base.location, runtime="mlx", training_state="lora",
            adapter_stack=(str(adapter.resolve()),), adapter_hashes=(sha256_tree(adapter),),
            artifact_sha256=sha256_tree(Path(base.location)), capabilities=base.capabilities,
            status=ModelStatus.CANDIDATE,
        ))

    def _write_run(self, run: TrainingRun) -> None:
        path = self.root / "learning" / "training" / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
