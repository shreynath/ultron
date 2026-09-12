"""Persistent, local memory and explicit learning-event storage.

Memory records are not model weights and this module never describes them as
training.  Its small lexical retriever is intentionally labelled as such; a
semantic embedding index can be added only when a local embedding model has
been provisioned.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.local_intelligence.core import ensure_local_layout


class MemoryKind(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    USER = "user"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    FORGOTTEN = "forgotten"


class FeedbackKind(str, Enum):
    GOOD = "good"
    BAD = "bad"
    CORRECTION = "correction"
    IMPORTANT = "important"
    LEARN_THIS = "learn_this"
    DO_NOT_LEARN = "do_not_learn"


class LearningExampleStatus(str, Enum):
    RAW = "raw"
    REVIEW_REQUIRED = "review_required"
    CURATED = "curated"
    REJECTED = "rejected"
    TRAINING_READY = "training_ready"
    TRAINED = "trained"
    EVALUATED = "evaluated"


class LocalMemory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: UUID = Field(default_factory=uuid4)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=8_000)
    source: str = Field(min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: int = Field(default=0, ge=0, le=10)
    tags: tuple[str, ...] = ()


class FeedbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feedback_id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID | None = None
    kind: FeedbackKind
    original_output: str | None = Field(default=None, max_length=20_000)
    corrected_output: str | None = Field(default=None, max_length=20_000)
    reason: str | None = Field(default=None, max_length=2_000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    training_eligible: bool = False
    status: LearningExampleStatus = LearningExampleStatus.RAW


_SECRET = re.compile(r"(?:sk-[a-zA-Z0-9_-]{16,}|(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s]+|bearer\s+[a-zA-Z0-9._~+/=-]{8,})", re.IGNORECASE)
_WORDS = re.compile(r"[a-z0-9]{2,}", re.IGNORECASE)


def _safe_text(value: str) -> str:
    if _SECRET.search(value):
        raise ValueError("Sensitive credentials cannot be stored in memory or learning data.")
    return value.strip()


def _score(query: str, text: str) -> float:
    left, right = set(_WORDS.findall(query.lower())), set(_WORDS.findall(text.lower()))
    return len(left & right) / len(left | right) if left and right else 0.0


class LocalMemoryStore:
    """SQLite-backed four-namespace memory with recoverable deletion."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_local_layout(root)
        self.path = self.root / "memory" / "memory.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS memories_lookup ON memories(kind, status)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, memory: LocalMemory) -> LocalMemory:
        safe = memory.model_copy(update={"content": _safe_text(memory.content), "updated_at": datetime.now(UTC)})
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(memory_id) DO UPDATE SET status=excluded.status, content=excluded.content, updated_at=excluded.updated_at, payload=excluded.payload", (str(safe.memory_id), safe.kind.value, safe.status.value, safe.content, safe.created_at.isoformat(), safe.updated_at.isoformat(), safe.model_dump_json()))
        return safe

    def retrieve(self, query: str, *, kinds: tuple[MemoryKind, ...] | None = None, limit: int = 5) -> tuple[LocalMemory, ...]:
        selected = kinds or tuple(MemoryKind)
        marks = ",".join("?" for _ in selected)
        with self._lock, self._connect() as connection:
            rows = connection.execute(f"SELECT payload FROM memories WHERE status = ? AND kind IN ({marks})", (MemoryStatus.ACTIVE.value, *(kind.value for kind in selected))).fetchall()
        ranked = [(item, _score(query, item.content)) for row in rows if (item := LocalMemory.model_validate_json(row["payload"]))]
        return tuple(item for item, score in sorted(ranked, key=lambda row: row[1], reverse=True) if score > 0)[:max(1, min(limit, 20))]

    def forget(self, memory_id: UUID) -> LocalMemory | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM memories WHERE memory_id = ?", (str(memory_id),)).fetchone()
            if row is None:
                return None
            prior = LocalMemory.model_validate_json(row["payload"])
            forgotten = prior.model_copy(update={"status": MemoryStatus.FORGOTTEN, "updated_at": datetime.now(UTC)})
            connection.execute("UPDATE memories SET status = ?, updated_at = ?, payload = ? WHERE memory_id = ?", (forgotten.status.value, forgotten.updated_at.isoformat(), forgotten.model_dump_json(), str(memory_id)))
        return forgotten


class LearningEventStore:
    """Raw feedback is durable but cannot become training data without curation."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_local_layout(root)
        self.path = self.root / "learning" / "events.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS feedback (feedback_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def record(self, event: FeedbackEvent) -> FeedbackEvent:
        for value in (event.original_output, event.corrected_output, event.reason):
            if value:
                _safe_text(value)
        eligible = event.kind in {FeedbackKind.CORRECTION, FeedbackKind.LEARN_THIS} and bool(event.corrected_output or event.original_output)
        saved = event.model_copy(update={"training_eligible": eligible, "status": LearningExampleStatus.REVIEW_REQUIRED if eligible else LearningExampleStatus.RAW})
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", (str(saved.feedback_id), saved.status.value, saved.created_at.isoformat(), saved.model_dump_json()))
        return saved

    def get(self, feedback_id: UUID) -> FeedbackEvent | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM feedback WHERE feedback_id = ?", (str(feedback_id),)).fetchone()
        return FeedbackEvent.model_validate_json(row["payload"]) if row else None

    def curate(self, feedback_id: UUID, *, approve: bool) -> FeedbackEvent:
        event = self.get(feedback_id)
        if event is None:
            raise KeyError("Unknown learning event.")
        if not event.training_eligible:
            raise ValueError("This feedback is not eligible for training curation.")
        curated = event.model_copy(update={"status": LearningExampleStatus.CURATED if approve else LearningExampleStatus.REJECTED})
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE feedback SET status = ?, payload = ? WHERE feedback_id = ?", (curated.status.value, curated.model_dump_json(), str(feedback_id)))
        return curated

    def training_dataset(self, *, dataset_id: str) -> Path:
        """Write only curator-approved examples as a reproducible local JSONL artifact."""
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM feedback WHERE status = ? ORDER BY created_at", (LearningExampleStatus.CURATED.value,)).fetchall()
        items = [FeedbackEvent.model_validate_json(row["payload"]) for row in rows]
        if not items:
            raise ValueError("No curated learning examples are available.")
        target = self.root / "learning" / "curated" / f"{dataset_id}.jsonl"
        if target.exists():
            raise ValueError("Dataset IDs are immutable; choose a new version.")
        lines = []
        for event in items:
            prompt = event.original_output or ""
            completion = event.corrected_output or event.original_output or ""
            lines.append(json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "provenance": {"feedback_id": str(event.feedback_id), "kind": event.kind.value}}, separators=(",", ":")))
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target
