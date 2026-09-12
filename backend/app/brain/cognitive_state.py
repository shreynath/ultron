"""Inspectable persistent task state, not a claim of consciousness."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class CognitiveStage(str, Enum):
    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    RESEARCHING = "RESEARCHING"
    REASONING = "REASONING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RESPONDING = "RESPONDING"
    LEARNING = "LEARNING"
    FAILED = "FAILED"


_LEGAL: dict[CognitiveStage, frozenset[CognitiveStage]] = {
    CognitiveStage.IDLE: frozenset({CognitiveStage.UNDERSTANDING}),
    CognitiveStage.UNDERSTANDING: frozenset({CognitiveStage.PLANNING, CognitiveStage.REASONING, CognitiveStage.FAILED}),
    CognitiveStage.PLANNING: frozenset({CognitiveStage.RESEARCHING, CognitiveStage.EXECUTING, CognitiveStage.REASONING, CognitiveStage.FAILED}),
    CognitiveStage.RESEARCHING: frozenset({CognitiveStage.REASONING, CognitiveStage.VERIFYING, CognitiveStage.FAILED}),
    CognitiveStage.REASONING: frozenset({CognitiveStage.EXECUTING, CognitiveStage.RESPONDING, CognitiveStage.FAILED}),
    CognitiveStage.EXECUTING: frozenset({CognitiveStage.VERIFYING, CognitiveStage.RESPONDING, CognitiveStage.FAILED}),
    CognitiveStage.VERIFYING: frozenset({CognitiveStage.REASONING, CognitiveStage.RESPONDING, CognitiveStage.LEARNING, CognitiveStage.FAILED}),
    CognitiveStage.RESPONDING: frozenset({CognitiveStage.VERIFYING, CognitiveStage.LEARNING, CognitiveStage.IDLE}),
    CognitiveStage.LEARNING: frozenset({CognitiveStage.IDLE, CognitiveStage.FAILED}),
    CognitiveStage.FAILED: frozenset({CognitiveStage.LEARNING, CognitiveStage.IDLE}),
}


class StateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sequence: int = Field(ge=1)
    stage: CognitiveStage
    component: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=500)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkingItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=2_000)
    source: str = Field(min_length=1, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    priority: int = Field(default=0, ge=0, le=10)
    expires_at: datetime | None = None


class CognitiveTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=1, max_length=400)
    stage: CognitiveStage = CognitiveStage.IDLE
    plan_id: UUID | None = None
    uncertainties: tuple[str, ...] = ()
    working_memory: tuple[WorkingItem, ...] = ()
    events: tuple[StateEvent, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_version: str
    available_tools: tuple[str, ...]
    known_capabilities: tuple[str, ...]
    known_limitations: tuple[str, ...]
    active_task_count: int
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CognitiveStateStore:
    """SQLite-backed task checkpoints containing concise factual metadata only."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS cognitive_tasks (task_id TEXT PRIMARY KEY, stage TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, goal: str, *, task_id: UUID | None = None) -> CognitiveTaskState:
        state = CognitiveTaskState(task_id=task_id or uuid4(), goal=goal)
        self._write(state)
        return state

    def get(self, task_id: UUID) -> CognitiveTaskState | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM cognitive_tasks WHERE task_id = ?", (str(task_id),)).fetchone()
        return CognitiveTaskState.model_validate_json(row["payload"]) if row else None

    def transition(self, task_id: UUID, stage: CognitiveStage, *, component: str, summary: str) -> CognitiveTaskState:
        state = self.get(task_id)
        if state is None:
            raise KeyError(task_id)
        if stage is not state.stage and stage not in _LEGAL[state.stage]:
            raise ValueError(f"Illegal cognitive transition: {state.stage.value} -> {stage.value}")
        event = StateEvent(sequence=len(state.events) + 1, stage=stage, component=component, summary=summary)
        updated = state.model_copy(update={"stage": stage, "events": (*state.events, event), "updated_at": datetime.now(UTC)})
        self._write(updated)
        return updated

    def put_working_item(self, task_id: UUID, item: WorkingItem) -> CognitiveTaskState:
        state = self.get(task_id)
        if state is None:
            raise KeyError(task_id)
        now = datetime.now(UTC)
        items = [existing for existing in state.working_memory if existing.key != item.key and (existing.expires_at is None or existing.expires_at > now)]
        items.append(item)
        items.sort(key=lambda candidate: (candidate.priority, candidate.key), reverse=True)
        updated = state.model_copy(update={"working_memory": tuple(items[:32]), "updated_at": now})
        self._write(updated)
        return updated

    def self_model(self, *, model_version: str, available_tools: tuple[str, ...], capabilities: tuple[str, ...], limitations: tuple[str, ...]) -> SelfModel:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM cognitive_tasks WHERE stage != ?", (CognitiveStage.IDLE.value,)).fetchone()
        return SelfModel(model_version=model_version, available_tools=available_tools, known_capabilities=capabilities, known_limitations=limitations, active_task_count=int(row["count"]))

    def _write(self, state: CognitiveTaskState) -> None:
        payload = state.model_dump_json()
        with self._connect() as connection:
            connection.execute("INSERT INTO cognitive_tasks VALUES (?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET stage=excluded.stage, updated_at=excluded.updated_at, payload=excluded.payload", (str(state.task_id), state.stage.value, state.updated_at.isoformat(), payload))
