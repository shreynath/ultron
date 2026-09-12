"""Native local model runtime and auditable model registry.

There is intentionally no remote-provider adapter in this module.  A model is
loadable only when its recorded path is on local disk under ``ULTRON_HOME``.
MLX-LM is loaded lazily so a normal API boot remains useful while an operator
installs the optional Apple-Silicon runtime or provisions model weights.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import settings


class LocalModelError(RuntimeError):
    """An honest local-runtime failure; callers must not substitute cloud IO."""


class LocalModelUnavailable(LocalModelError):
    """The selected local runtime, weights, or active model is unavailable."""


class NetworkMode(str, Enum):
    OFFLINE = "offline"
    LOCAL_ONLY = "local_only"
    LOCAL_PLUS_TOOLS = "local_plus_tools"


class ModelCapability(str, Enum):
    FAST = "fast"
    GENERAL = "general"
    REASONING = "reasoning"
    CODING = "coding"


class ModelStatus(str, Enum):
    BASE = "base"
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class ModelOrigin(str, Enum):
    ULTRON_FROM_SCRATCH = "from_scratch"
    THIRD_PARTY_REFERENCE_MODEL = "third_party_reference"
    DERIVED_FROM_THIRD_PARTY = "derived"
    # Compatibility aliases for registries written before the lineage contract.
    FROM_SCRATCH = ULTRON_FROM_SCRATCH
    THIRD_PARTY_REFERENCE = THIRD_PARTY_REFERENCE_MODEL
    DERIVED = DERIVED_FROM_THIRD_PARTY


class GenerationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_load_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_latency_ms: float
    tokens_per_second: float | None = None


class ModelRecord(BaseModel):
    """Registry metadata for an immutable local weight or adapter artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    origin: ModelOrigin = ModelOrigin.THIRD_PARTY_REFERENCE_MODEL
    base_model: str | None = Field(default=None, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    location: str = Field(min_length=1, max_length=2048)
    runtime: str = Field(default="mlx", min_length=1, max_length=64)
    quantization: str | None = Field(default=None, max_length=128)
    parameter_count: str | None = Field(default=None, max_length=64)
    context_length: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    training_state: str = Field(default="base", max_length=64)
    adapter_stack: tuple[str, ...] = ()
    adapter_hashes: tuple[str, ...] = ()
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    configuration_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    dataset_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    lineage: dict[str, str] = Field(default_factory=dict)
    capabilities: tuple[ModelCapability, ...] = (ModelCapability.GENERAL,)
    status: ModelStatus = ModelStatus.CANDIDATE

    @property
    def is_ultron_foundation(self) -> bool:
        return self.origin is ModelOrigin.ULTRON_FROM_SCRATCH and self.base_model is None


class RegistryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str
    model_id: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    detail: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Measured local performance; fields are absent rather than invented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_tokens: int = Field(ge=1)
    generation_tokens: int = Field(ge=1)
    prompt_tokens_per_second: float = Field(gt=0)
    generation_tokens_per_second: float = Field(gt=0)
    peak_memory_gb: float = Field(gt=0)
    runtime_version: str = Field(min_length=1)
    hardware: str = Field(min_length=1)


def ultron_home() -> Path:
    root = Path(settings.ultron_home).expanduser()
    return root.resolve()


def ensure_local_layout(root: Path | None = None) -> Path:
    root = (root or ultron_home()).resolve()
    for name in (
        "models/base", "models/adapters", "models/promoted", "models/archived",
        "memory/conversations", "memory/semantic", "memory/episodic", "memory/procedural", "memory/user",
        "learning/raw", "learning/curated", "learning/feedback", "learning/evaluations", "learning/training", "learning/checkpoints",
        "registry", "benchmarks", "cache", "logs",
    ):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            path.chmod(0o700)
    return root


def _safe_local_path(location: str, root: Path) -> Path:
    if "://" in location:
        raise LocalModelError("Model locations must be local filesystem paths.")
    candidate = Path(location).expanduser().resolve()
    try:
        candidate.relative_to(root / "models")
    except ValueError as exc:
        raise LocalModelError("Model location must be inside ULTRON_HOME/models.") from exc
    return candidate


def sha256_tree(path: Path) -> str:
    """Hash local model contents deterministically without following symlinks."""
    digest = hashlib.sha256()
    paths: Iterable[Path] = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in paths:
        digest.update(str(item.relative_to(path if path.is_dir() else path.parent)).encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class LocalModelRegistry:
    """Append-audited local registry; promotion and rollback never delete weights."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_local_layout(root)
        self.path = self.root / "registry" / "models.json"
        self.audit_path = self.root / "registry" / "events.jsonl"
        self._lock = asyncio.Lock()
        if not self.path.exists():
            self._write({"models": {}, "active": {}})

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalModelError("Local model registry is unreadable.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
            raise LocalModelError("Local model registry has an invalid format.")
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(self.path)

    def _append_event(self, event: RegistryEvent) -> None:
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        if os.name != "nt":
            self.audit_path.chmod(0o600)

    def list(self) -> tuple[ModelRecord, ...]:
        data = self._read()
        return tuple(ModelRecord.model_validate(value) for value in data["models"].values())

    def get(self, model_id: str) -> ModelRecord | None:
        raw = self._read()["models"].get(model_id)
        return ModelRecord.model_validate(raw) if raw else None

    def active(self, capability: ModelCapability) -> ModelRecord | None:
        data = self._read()
        model_id = data.get("active", {}).get(capability.value)
        return ModelRecord.model_validate(data["models"][model_id]) if model_id in data["models"] else None

    async def register(self, record: ModelRecord, *, verify_hash: bool = True) -> ModelRecord:
        location = _safe_local_path(record.location, self.root)
        if not location.exists():
            raise LocalModelError(f"Local model artifact does not exist: {location}")
        actual_hash = await asyncio.to_thread(sha256_tree, location) if verify_hash else record.artifact_sha256
        if record.artifact_sha256 and actual_hash != record.artifact_sha256:
            raise LocalModelError("Local model hash does not match the registry record.")
        if record.adapter_hashes and len(record.adapter_hashes) != len(record.adapter_stack):
            raise LocalModelError("Every registered adapter needs a corresponding integrity hash.")
        for adapter_path, expected_hash in zip(record.adapter_stack, record.adapter_hashes, strict=True):
            adapter = _safe_local_path(adapter_path, self.root)
            if not adapter.exists():
                raise LocalModelError("Registered local adapter artifact does not exist.")
            if verify_hash and await asyncio.to_thread(sha256_tree, adapter) != expected_hash:
                raise LocalModelError("Local adapter hash does not match the registry record.")
        registered = record.model_copy(update={"location": str(location), "artifact_sha256": actual_hash})
        async with self._lock:
            data = self._read()
            prior = data["models"].get(registered.model_id)
            if prior and ModelRecord.model_validate(prior) != registered:
                raise LocalModelError("Model IDs are immutable; register a new version identifier.")
            data["models"][registered.model_id] = registered.model_dump(mode="json")
            self._write(data)
            self._append_event(RegistryEvent(event="registered", model_id=registered.model_id, detail={"sha256": actual_hash}))
        return registered

    async def promote(self, model_id: str, *, capabilities: tuple[ModelCapability, ...] | None = None) -> ModelRecord:
        async with self._lock:
            data = self._read()
            raw = data["models"].get(model_id)
            if raw is None:
                raise LocalModelError("Unknown local model.")
            record = ModelRecord.model_validate(raw)
            if record.status not in {ModelStatus.CANDIDATE, ModelStatus.EVALUATING, ModelStatus.PROMOTED}:
                raise LocalModelError("Only candidate/evaluating models can be promoted.")
            if record.origin is ModelOrigin.ULTRON_FROM_SCRATCH:
                required_lineage = {"initialization": "RANDOM", "pretrained_weights": "NONE", "parent_model": "NONE"}
                if not record.is_ultron_foundation or not record.tokenizer_sha256 or not record.dataset_manifest_sha256:
                    raise LocalModelError("From-scratch promotion requires tokenizer and dataset provenance.")
                if any(record.lineage.get(key) != value for key, value in required_lineage.items()):
                    raise LocalModelError("From-scratch promotion requires auditable random-initialization lineage.")
            selected = capabilities or record.capabilities
            if not selected:
                raise LocalModelError("A promoted model needs at least one capability class.")
            promoted = record.model_copy(update={"status": ModelStatus.PROMOTED, "capabilities": selected})
            data["models"][model_id] = promoted.model_dump(mode="json")
            active = data.setdefault("active", {})
            for capability in selected:
                old_id = active.get(capability.value)
                active[capability.value] = model_id
                if old_id and old_id != model_id and old_id in data["models"]:
                    old = ModelRecord.model_validate(data["models"][old_id])
                    data["models"][old_id] = old.model_copy(update={"status": ModelStatus.ARCHIVED}).model_dump(mode="json")
            self._write(data)
            self._append_event(RegistryEvent(event="promoted", model_id=model_id, detail={"capabilities": [item.value for item in selected]}))
        return promoted

    async def rollback(self, capability: ModelCapability, target_model_id: str) -> ModelRecord:
        target = self.get(target_model_id)
        if target is None:
            raise LocalModelError("Rollback target is not registered.")
        if capability not in target.capabilities:
            raise LocalModelError("Rollback target does not satisfy the requested capability.")
        return await self.promote(target_model_id, capabilities=tuple(set(target.capabilities) | {capability}))

    async def record_benchmark(self, result: BenchmarkResult) -> Path:
        """Persist one measured run without rewriting immutable model metadata."""
        async with self._lock:
            data = self._read()
            if result.model_id not in data["models"]:
                raise LocalModelError("A benchmark requires a registered local model.")
            path = self.root / "benchmarks" / f"{result.benchmark_id}.json"
            if path.exists():
                raise LocalModelError("Benchmark IDs are immutable; choose a new identifier.")
            path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
            self._append_event(RegistryEvent(event="benchmark_recorded", model_id=result.model_id, detail={"benchmark_id": result.benchmark_id, "generation_tps": result.generation_tokens_per_second, "peak_memory_gb": result.peak_memory_gb}))
        return path


class LocalModelRuntime(ABC):
    """Provider-agnostic contract for code which must never invoke a cloud model."""

    @abstractmethod
    async def load(self, record: ModelRecord) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.2) -> tuple[str, GenerationMetrics]: ...

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.2) -> AsyncIterator[str]: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...


class MlxLocalRuntime(LocalModelRuntime):
    """MLX-LM adapter for models which have already been placed on local disk."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._record: ModelRecord | None = None
        self._load_ms = 0.0
        self._lock = asyncio.Lock()

    def _modules(self) -> tuple[Any, Any, Any, Any]:
        try:
            from mlx_lm import (  # type: ignore[import-not-found]
                generate,
                load,
                stream_generate,
            )
            from mlx_lm.sample_utils import (
                make_sampler,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise LocalModelUnavailable(
                "MLX-LM is not installed. Install the `local` backend extra and provision local model weights."
            ) from exc
        return load, generate, stream_generate, make_sampler

    async def load(self, record: ModelRecord) -> None:
        if record.runtime != "mlx":
            raise LocalModelUnavailable(f"MLX runtime cannot load runtime={record.runtime!r}.")
        location = _safe_local_path(record.location, ultron_home())
        if not location.exists():
            raise LocalModelUnavailable("Registered local model files are absent.")
        if self._record and self._record.model_id == record.model_id:
            return
        async with self._lock:
            if self._record and self._record.model_id == record.model_id:
                return
            load, _, _, _ = self._modules()
            started = time.perf_counter()
            adapter_path = record.adapter_stack[-1] if record.adapter_stack else None
            if adapter_path:
                adapter = _safe_local_path(adapter_path, ultron_home())
                if not adapter.exists():
                    raise LocalModelUnavailable("Registered local adapter files are absent.")
                self._model, self._tokenizer = await asyncio.to_thread(
                    load, str(location), adapter_path=str(adapter)
                )
            else:
                self._model, self._tokenizer = await asyncio.to_thread(load, str(location))
            self._record = record
            self._load_ms = (time.perf_counter() - started) * 1000

    async def unload(self) -> None:
        async with self._lock:
            self._model = None
            self._tokenizer = None
            self._record = None
            self._load_ms = 0.0

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        if not self._tokenizer:
            raise LocalModelUnavailable("No local model is loaded.")
        try:
            # Qwen3's default template starts an exposed reasoning block.
            # Application cognition already owns planning/verification and the
            # Executive requires directly parseable JSON, so disable optional
            # visible thinking where the tokenizer supports the switch.
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except (AttributeError, TypeError, ValueError):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except (AttributeError, TypeError, ValueError):
                return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"

    async def generate(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.2) -> tuple[str, GenerationMetrics]:
        if self._model is None:
            raise LocalModelUnavailable("No local model is loaded.")
        _, generate, _, make_sampler = self._modules()
        prompt = self._prompt(messages)
        started = time.perf_counter()
        text = await asyncio.to_thread(
            generate, self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=make_sampler(temp=temperature), verbose=False,
        )
        elapsed = (time.perf_counter() - started) * 1000
        completion = len(self._tokenizer.encode(text)) if hasattr(self._tokenizer, "encode") else None
        return str(text).strip(), GenerationMetrics(model_load_ms=self._load_ms, completion_tokens=completion, total_latency_ms=elapsed, tokens_per_second=round((completion or 0) / (elapsed / 1000), 2) if completion and elapsed else None)

    async def stream(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float = 0.2) -> AsyncIterator[str]:
        if self._model is None:
            raise LocalModelUnavailable("No local model is loaded.")
        _, _, stream_generate, make_sampler = self._modules()
        prompt = self._prompt(messages)
        iterator = await asyncio.to_thread(lambda: iter(stream_generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
        )))
        while True:
            item = await asyncio.to_thread(lambda: next(iterator, None))
            if item is None:
                break
            text = getattr(item, "text", item)
            if text:
                yield str(text)

    async def health(self) -> dict[str, Any]:
        try:
            self._modules()
            installed = True
        except LocalModelUnavailable:
            installed = False
        return {"runtime": "mlx", "installed": installed, "loaded_model": self._record.model_id if self._record else None, "local": True}


class LocalModelOrchestrator:
    """Capability routing, lifecycle, metrics, and the no-cloud failure boundary."""

    def __init__(self, registry: LocalModelRegistry | None = None, runtime: LocalModelRuntime | None = None) -> None:
        self.registry = registry or LocalModelRegistry()
        self.runtime = runtime or MlxLocalRuntime()

    def _messages(self, *, system_prompt: str, user_message: str, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        for message in history or []:
            if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str):
                messages.append({"role": message["role"], "content": message["content"]})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def generate(self, *, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, capability: ModelCapability = ModelCapability.GENERAL, max_tokens: int | None = None, temperature: float = 0.2) -> tuple[str, GenerationMetrics]:
        record = self.registry.active(capability)
        if record is None:
            raise LocalModelUnavailable(f"No promoted local model is configured for {capability.value} generation.")
        await self.runtime.load(record)
        return await self.runtime.generate(self._messages(system_prompt=system_prompt, user_message=user_message, history=history), max_tokens=max_tokens or settings.local_model_max_tokens, temperature=temperature)

    async def stream(self, *, system_prompt: str, user_message: str, history: list[dict[str, str]] | None = None, capability: ModelCapability = ModelCapability.GENERAL, max_tokens: int | None = None) -> AsyncIterator[str]:
        record = self.registry.active(capability)
        if record is None:
            raise LocalModelUnavailable(f"No promoted local model is configured for {capability.value} generation.")
        await self.runtime.load(record)
        async for token in self.runtime.stream(self._messages(system_prompt=system_prompt, user_message=user_message, history=history), max_tokens=max_tokens or settings.local_model_max_tokens):
            yield token

    async def diagnostics(self) -> dict[str, Any]:
        try:
            mode = NetworkMode(settings.local_network_mode)
        except ValueError:
            mode = NetworkMode.LOCAL_ONLY
        health = await self.runtime.health()
        active = {capability.value: (record.model_id if (record := self.registry.active(capability)) else None) for capability in ModelCapability}
        return {"inference": "LOCAL", "runtime": health, "active_models": active, "model_root": str(ultron_home() / "models"), "memory_root": str(ultron_home() / "memory"), "network": mode.value, "cloud_llm": "DISABLED", "ollama": "NOT_REQUIRED"}


_orchestrator: LocalModelOrchestrator | None = None


def get_local_orchestrator() -> LocalModelOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LocalModelOrchestrator()
    return _orchestrator
