"""Bounded, persistent cognitive task loop with a validated model boundary."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class CognitiveTaskStatus(str, Enum):
    CREATED = "CREATED"
    ASSESSED = "ASSESSED"
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    VERIFYING = "VERIFYING"
    REFLECTING = "REFLECTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelOutputKind(str, Enum):
    ASSESSMENT = "ASSESSMENT"
    PLAN = "PLAN"
    TOOL_REQUEST = "TOOL_REQUEST"
    OBSERVATION_INTERPRETATION = "OBSERVATION_INTERPRETATION"
    VERIFICATION = "VERIFICATION"
    REFLECTION = "REFLECTION"
    FINAL_RESPONSE = "FINAL_RESPONSE"


class KnowledgeOrigin(str, Enum):
    MODEL_KNOWLEDGE = "MODEL_KNOWLEDGE"
    LOCAL_MEMORY = "LOCAL_MEMORY"
    LOCAL_KNOWLEDGE = "LOCAL_KNOWLEDGE"
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    USER_PROVIDED_INFORMATION = "USER_PROVIDED_INFORMATION"


class GroundedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content: str = Field(min_length=1, max_length=4_000)
    origin: KnowledgeOrigin
    source: str = Field(min_length=1, max_length=300)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evidence_derived: bool = False


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ModelOutputKind
    content: str = Field(min_length=1, max_length=4_000)
    tool_name: str | None = Field(default=None, max_length=100)
    arguments: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=12)


class CognitiveTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=1, max_length=1_000)
    constraints: tuple[str, ...] = ()
    task_type: str = "UNKNOWN"
    assessment: str | None = None
    state: CognitiveTaskStatus = CognitiveTaskStatus.CREATED
    plan: tuple[str, ...] = ()
    current_step: int = 0
    working_memory: tuple[GroundedItem, ...] = ()
    required_tools: tuple[str, ...] = ()
    evidence: tuple[GroundedItem, ...] = ()
    observations: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    verification: str | None = None
    reflection: str | None = None
    status: CognitiveTaskStatus = CognitiveTaskStatus.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_version: str = "UNKNOWN"
    knowledge_version: str = "UNKNOWN"
    iterations: int = 0
    replans: int = 0
    failures: int = 0


class CognitiveLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_iterations: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=4, ge=0, le=32)
    max_replans: int = Field(default=2, ge=0, le=16)
    max_failures: int = Field(default=2, ge=0, le=16)


class CognitiveTaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS phase_g_tasks (task_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, task: CognitiveTask) -> CognitiveTask:
        task = task.model_copy(update={"updated_at": datetime.now(UTC), "state": task.status})
        with self._connect() as connection:
            connection.execute("INSERT INTO phase_g_tasks VALUES (?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload", (str(task.task_id), task.status.value, task.updated_at.isoformat(), task.model_dump_json()))
        return task

    def get(self, task_id: UUID) -> CognitiveTask | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM phase_g_tasks WHERE task_id = ?", (str(task_id),)).fetchone()
        return CognitiveTask.model_validate_json(row["payload"]) if row else None


class CognitiveModel(Protocol):
    def produce(self, kind: ModelOutputKind, task: CognitiveTask) -> ModelOutput: ...


class GeneratedFoundationContractModel:
    """Adapter that preserves raw ULTRON generation as advisory contract text.

    The loop supplies the requested contract mode; it never synthesizes a plan,
    tool name, or answer when the foundation model fails to do so.
    """

    def __init__(self, generator, generation_config) -> None:
        self.generator, self.generation_config = generator, generation_config

    def produce(self, kind: ModelOutputKind, task: CognitiveTask) -> ModelOutput:
        prompt = f"MODE: {kind.value}\nGOAL: {task.goal}\nOUTPUT:"
        result = self.generator.generate(prompt, self.generation_config)
        if not result.text.strip():
            raise ValueError(f"MODEL_OUTPUT_SCHEMA_FAILURE:{result.termination_reason.value}: empty generated text")
        return ModelOutput(kind=kind, content=result.text)


class LocalToolExecutor(Protocol):
    def execute(self, name: str, arguments: dict[str, str | int | float | bool]) -> str: ...


class CognitiveLoop:
    """A small finite lifecycle; model output remains advisory and schema-checked."""

    def __init__(self, store: CognitiveTaskStore, model: CognitiveModel, limits: CognitiveLimits | None = None, tool_executor: LocalToolExecutor | None = None) -> None:
        self.store, self.model, self.limits, self.tool_executor = store, model, limits or CognitiveLimits(), tool_executor

    def run(self, task: CognitiveTask) -> CognitiveTask:
        task = self.store.save(task)
        for kind, status in ((ModelOutputKind.ASSESSMENT, CognitiveTaskStatus.ASSESSED), (ModelOutputKind.PLAN, CognitiveTaskStatus.PLANNED), (ModelOutputKind.TOOL_REQUEST, CognitiveTaskStatus.EXECUTING), (ModelOutputKind.OBSERVATION_INTERPRETATION, CognitiveTaskStatus.EXECUTING), (ModelOutputKind.VERIFICATION, CognitiveTaskStatus.VERIFYING), (ModelOutputKind.REFLECTION, CognitiveTaskStatus.REFLECTING), (ModelOutputKind.FINAL_RESPONSE, CognitiveTaskStatus.COMPLETED)):
            if task.iterations >= self.limits.max_iterations:
                return self.store.save(task.model_copy(update={"status": CognitiveTaskStatus.FAILED, "failures": task.failures + 1, "results": (*task.results, "LIMIT_EXCEEDED: max_iterations")}))
            try:
                output = self.model.produce(kind, task)
                if not isinstance(output, ModelOutput) or output.kind is not kind:
                    raise ValueError("MODEL_OUTPUT_SCHEMA_FAILURE")
            except (TypeError, ValueError) as exc:
                return self.store.save(task.model_copy(update={"status": CognitiveTaskStatus.FAILED, "failures": task.failures + 1, "results": (*task.results, f"MODEL_OUTPUT_SCHEMA_FAILURE:{exc}")}))
            updates: dict[str, object] = {"status": status, "iterations": task.iterations + 1}
            if kind is ModelOutputKind.ASSESSMENT:
                updates["assessment"] = output.content
            elif kind is ModelOutputKind.PLAN:
                updates["plan"] = tuple(part.strip() for part in output.content.split("\n") if part.strip())[:16]
            elif kind is ModelOutputKind.TOOL_REQUEST:
                if self.tool_executor is None or output.tool_name not in task.required_tools or len(task.actions) >= self.limits.max_tool_calls:
                    return self.store.save(task.model_copy(update={"status": CognitiveTaskStatus.FAILED, "failures": task.failures + 1, "results": (*task.results, "TOOL_REQUEST_REJECTED")}))
                try:
                    observation = self.tool_executor.execute(output.tool_name, output.arguments)
                except (OSError, RuntimeError, ValueError) as exc:
                    return self.store.save(task.model_copy(update={"status": CognitiveTaskStatus.FAILED, "failures": task.failures + 1, "results": (*task.results, f"TOOL_FAILURE:{exc}")}))
                updates["actions"] = (*task.actions, output.tool_name)
                updates["observations"] = (*task.observations, observation[:2_000])
            elif kind is ModelOutputKind.VERIFICATION:
                updates["verification"] = output.content
            elif kind is ModelOutputKind.REFLECTION:
                updates["reflection"] = output.content
            else:
                updates["results"] = (*task.results, output.content)
            task = self.store.save(task.model_copy(update=updates))
        return task
