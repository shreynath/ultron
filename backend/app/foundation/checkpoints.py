"""Inspectable, atomic checkpoints for ULTRON-owned training runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CHECKPOINT_SCHEMA_VERSION = "ultron.checkpoint.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class CheckpointError(RuntimeError):
    pass


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str
    model_id: str
    model_version: str
    model_origin: str = "ULTRON_FROM_SCRATCH"
    architecture_config_sha256: str
    tokenizer_id: str
    tokenizer_version: str
    tokenizer_sha256: str
    dataset_release_id: str
    dataset_manifest_sha256: str
    training_run_id: str
    parent_checkpoint: str | None = None
    global_step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    tokens_consumed: int = Field(ge=0)
    dataset_position: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    training_config: dict[str, Any]
    optimizer_config: dict[str, Any]
    scheduler_config: dict[str, Any]
    runtime: dict[str, str | None] = Field(default_factory=dict)
    hardware: dict[str, str | int | float | None] = Field(default_factory=dict)
    rng: dict[str, Any] = Field(default_factory=dict)
    lineage: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    files: dict[str, dict[str, str | int]] = Field(default_factory=dict)


def runtime_metadata() -> dict[str, str | None]:
    try:
        import mlx
        mlx_version = getattr(mlx, "__version__", "UNKNOWN")
    except ImportError:
        mlx_version = None
    return {"python": platform.python_version(), "mlx": mlx_version, "platform": platform.platform()}


class CheckpointStore:
    """Publishes a complete checkpoint directory atomically after verification."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def inspect(path: Path) -> CheckpointManifest | dict[str, Any]:
        if path.is_file() and path.suffix == ".safetensors":
            return {"schema_version": "legacy-weights-only", "path": str(path), "resumable": False}
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise CheckpointError("Checkpoint manifest is missing.")
        try:
            manifest = CheckpointManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError("Checkpoint manifest is invalid.") from exc
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(f"Unsupported checkpoint schema: {manifest.schema_version}")
        required = {"model.safetensors", "optimizer.safetensors", "scheduler.json", "trainer_state.json"}
        if not required <= set(manifest.files):
            raise CheckpointError("Checkpoint is incomplete: required state artifact missing.")
        for name, record in manifest.files.items():
            artifact = path / name
            if not artifact.is_file() or artifact.stat().st_size != int(record["size"]):
                raise CheckpointError(f"Checkpoint artifact is missing or truncated: {name}")
            if sha256_file(artifact) != record["sha256"]:
                raise CheckpointError(f"Checkpoint artifact hash mismatch: {name}")
        return manifest

    def save(
        self,
        *,
        directory: Path,
        model: Any,
        optimizer_arrays: dict[str, Any],
        scheduler: dict[str, Any],
        trainer_state: dict[str, Any],
        manifest: CheckpointManifest,
    ) -> Path:
        import mlx.core as mx

        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise CheckpointError(f"Refusing to overwrite immutable checkpoint {directory}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            model.save_weights(str(temporary / "model.safetensors"))
            mx.save_safetensors(str(temporary / "optimizer.safetensors"), optimizer_arrays)
            (temporary / "scheduler.json").write_text(json.dumps(scheduler, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (temporary / "trainer_state.json").write_text(json.dumps(trainer_state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            for artifact in temporary.iterdir():
                if artifact.is_file():
                    with artifact.open("rb") as stream:
                        os.fsync(stream.fileno())
            files = {item.name: {"size": item.stat().st_size, "sha256": sha256_file(item)} for item in sorted(temporary.iterdir()) if item.is_file()}
            published = manifest.model_copy(update={"files": files})
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(published.model_dump_json(indent=2) + "\n", encoding="utf-8")
            with manifest_path.open("rb") as stream:
                os.fsync(stream.fileno())
            self.inspect(temporary)
            os.replace(temporary, directory)
            return directory
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
