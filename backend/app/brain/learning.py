"""Durable, evidence-gated learning for the cognitive runtime.

This module stores *experience* separately from reusable knowledge.  It never
executes a capability, grants authority, or asks a model to decide truth.  The
only promotion path to ``VERIFIED`` is a verifier result backed by evidence.
SQLite is deliberately used directly: the application currently has no shared
database layer and this keeps the learning boundary small, durable, and easy
to audit.
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult
from app.brain.observation import ExecutionObservation
from app.brain.schemas import TaskAssessment
from app.brain.verification import EvidenceReliability, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class KnowledgeScope(str, Enum):
    """Namespaces that prevent preferences, experience, and world facts mixing."""

    SYSTEM_EXPERIENCE = "system_experience"
    CAPABILITY = "capability"
    USER_MEMORY = "user_memory"
    WORLD = "world"


class KnowledgeStatus(str, Enum):
    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    STALE = "stale"
    RETIRED = "retired"
    REJECTED = "rejected"


class LearningEventStatus(str, Enum):
    RECORDED = "recorded"
    PROMOTED = "promoted"
    NOT_ELIGIBLE = "not_eligible"
    FAILED = "failed"


class Provenance(BaseModel):
    """Origin metadata; source content is data, never privileged instruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=100)
    correlation_id: UUID
    source_experience_id: UUID | None = None
    verification_id: UUID | None = None
    capability: str | None = Field(default=None, max_length=64)
    capability_version: str | None = Field(default=None, max_length=100)
    source_ids: tuple[UUID, ...] = Field(default=(), max_length=8)


class ExperienceRecord(BaseModel):
    """Sanitized immutable account of a completed/interrupted cognitive run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experience_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID
    created_at: datetime
    normalized_goal: str = Field(min_length=1, max_length=400)
    assessment: TaskAssessment
    route: str = Field(min_length=1, max_length=80)
    execution_status: str = Field(min_length=1, max_length=80)
    lifecycle_outcome: str = Field(min_length=1, max_length=80)
    capability: str | None = Field(default=None, max_length=64)
    operation: str | None = Field(default=None, max_length=64)
    action_status: str | None = Field(default=None, max_length=64)
    action: dict[str, Any] | None = None
    observation: ExecutionObservation | None = None
    verification: VerificationResult | None = None
    failure_reason: str | None = Field(default=None, max_length=500)
    response_outcome: str | None = Field(default=None, max_length=80)
    model_provider: str | None = Field(default=None, max_length=100)
    runtime_metadata: dict[str, str] = Field(default_factory=dict, max_length=12)
    confidence: float = Field(ge=0, le=1)
    provenance: Provenance


class LearnedKnowledge(BaseModel):
    """Current, typed projection of a claim; revisions retain its prior state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_id: UUID
    scope: KnowledgeScope
    claim: str = Field(min_length=1, max_length=500)
    claim_key: str = Field(min_length=1, max_length=200)
    status: KnowledgeStatus
    confidence: float = Field(ge=0, le=1)
    provenance: Provenance
    created_at: datetime
    last_validated_at: datetime | None = None
    revision: int = Field(ge=1)
    contradiction_count: int = Field(ge=0)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    last_checked_at: datetime | None = None
    freshness_seconds: int | None = Field(default=None, ge=60, le=31_536_000)


class RetrievalResult(BaseModel):
    """A retrieved data item for planning, including the truth metadata it needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge: LearnedKnowledge
    relevance: float = Field(ge=0, le=1)
    retrieved_at: datetime
    instruction_safe: bool = True


class LearningEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    experience_id: UUID
    status: LearningEventStatus
    knowledge_ids: tuple[UUID, ...] = Field(default=(), max_length=8)
    reason: str = Field(min_length=1, max_length=500)
    created_at: datetime


_SENSITIVE_KEY = re.compile(r"(?:secret|token|password|credential|authorization|cookie|api[_-]?key|session)", re.I)
_BEARER = re.compile(r"\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}\b", re.I)
_WORDS = re.compile(r"[a-z0-9]{2,}", re.I)


def sanitize_for_persistence(value: Any) -> Any:
    """Recursively redact common secret fields and bound untrusted payloads."""

    if isinstance(value, dict):
        return {
            str(key)[:100]: "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else sanitize_for_persistence(item)
            for key, item in list(value.items())[:32]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_persistence(item) for item in value[:32]]
    if isinstance(value, str):
        return _BEARER.sub("[REDACTED]", value)[:2_000]
    if value is None or type(value) in {bool, int, float}:
        return value
    return f"<{type(value).__name__}>"


def confidence_for(
    verification: VerificationResult | None,
    *,
    repeated_successes: int = 0,
    contradictions: int = 0,
) -> float:
    """Deterministic confidence; model phrasing is intentionally irrelevant."""

    if verification is None:
        base = 0.15
    elif verification.status is VerificationStatus.VERIFIED:
        authoritative = sum(
            evidence.reliability is EvidenceReliability.AUTHORITATIVE
            for evidence in verification.supporting_evidence
        )
        base = min(0.98, 0.82 + 0.06 * authoritative)
    elif verification.status is VerificationStatus.PARTIALLY_VERIFIED:
        base = 0.55
    elif verification.status is VerificationStatus.UNVERIFIED:
        base = 0.20
    elif verification.status is VerificationStatus.CONTRADICTED:
        base = 0.0
    else:
        base = 0.05
    repeat_bonus = min(0.08, max(0, repeated_successes - 1) * 0.02)
    return round(max(0.0, min(1.0, base + repeat_bonus - contradictions * 0.20)), 3)


def _claim_key(claim: str) -> str:
    normalized = " ".join(_WORDS.findall(claim.lower()))
    return hashlib.sha256(normalized.encode()).hexdigest()[:40]


def _score(query: str, claim: str) -> float:
    query_words = set(_WORDS.findall(query.lower()))
    claim_words = set(_WORDS.findall(claim.lower()))
    if not query_words or not claim_words:
        return 0.0
    return len(query_words & claim_words) / len(query_words | claim_words)


class LearningRepository:
    """Transactional local persistence for immutable experience and knowledge revisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS brain_schema_migrations (version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO brain_schema_migrations(version) VALUES (1);
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY, correlation_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge (
                    knowledge_id TEXT PRIMARY KEY, scope TEXT NOT NULL, claim_key TEXT NOT NULL,
                    claim TEXT NOT NULL, status TEXT NOT NULL, confidence REAL NOT NULL,
                    revision INTEGER NOT NULL, contradiction_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL, last_validated_at TEXT, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS knowledge_lookup ON knowledge(scope, claim_key, status);
                CREATE TABLE IF NOT EXISTS knowledge_revisions (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, changed_at TEXT NOT NULL, reason TEXT NOT NULL,
                    payload TEXT NOT NULL, UNIQUE(knowledge_id, revision)
                );
                CREATE TABLE IF NOT EXISTS contradictions (
                    contradiction_id TEXT PRIMARY KEY, prior_knowledge_id TEXT NOT NULL,
                    newer_knowledge_id TEXT NOT NULL, experience_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capability_metrics (
                    capability TEXT PRIMARY KEY, attempts INTEGER NOT NULL, successes INTEGER NOT NULL,
                    verified INTEGER NOT NULL, failures INTEGER NOT NULL, average_latency_ms REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def commit_experience(self, experience: ExperienceRecord) -> ExperienceRecord:
        payload = experience.model_dump_json()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload FROM experiences WHERE correlation_id = ?", (str(experience.correlation_id),)
            ).fetchone()
            if existing:
                connection.execute("COMMIT")
                return ExperienceRecord.model_validate_json(existing["payload"])
            connection.execute(
                "INSERT INTO experiences VALUES (?, ?, ?, ?)",
                (str(experience.experience_id), str(experience.correlation_id), experience.created_at.isoformat(), payload),
            )
            self._update_capability_metrics(connection, experience)
            connection.execute("COMMIT")
        logger.info("brain.learning experience_recorded experience_id=%s correlation_id=%s outcome=%s", experience.experience_id, experience.correlation_id, experience.lifecycle_outcome)
        return experience

    def _update_capability_metrics(self, connection: sqlite3.Connection, experience: ExperienceRecord) -> None:
        if not experience.capability:
            return
        row = connection.execute("SELECT * FROM capability_metrics WHERE capability = ?", (experience.capability,)).fetchone()
        attempts = (row["attempts"] if row else 0) + 1
        success = int(experience.action_status == "completed")
        verified = int(experience.verification is not None and experience.verification.status is VerificationStatus.VERIFIED)
        failure = int(not success)
        old_average = row["average_latency_ms"] if row else 0.0
        latency = experience.action.get("duration_ms", 0.0) if experience.action else 0.0
        average = ((old_average * (attempts - 1)) + float(latency)) / attempts
        connection.execute(
            "INSERT INTO capability_metrics VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(capability) DO UPDATE SET attempts=excluded.attempts, successes=excluded.successes, verified=excluded.verified, failures=excluded.failures, average_latency_ms=excluded.average_latency_ms, updated_at=excluded.updated_at",
            (experience.capability, attempts, (row["successes"] if row else 0) + success,
             (row["verified"] if row else 0) + verified, (row["failures"] if row else 0) + failure,
             average, datetime.now(UTC).isoformat()),
        )

    def promote(self, experience: ExperienceRecord) -> LearningEvent:
        verification = experience.verification
        if verification is None or not verification.claims:
            return LearningEvent(experience_id=experience.experience_id, status=LearningEventStatus.NOT_ELIGIBLE, reason="No evidence-backed claims are available for promotion.", created_at=datetime.now(UTC))
        if verification.status not in {VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED}:
            candidate_status = (
                KnowledgeStatus.REJECTED
                if verification.status is VerificationStatus.CONTRADICTED
                else KnowledgeStatus.CANDIDATE
            )
            provenance = Provenance(
                source="verification_candidate",
                correlation_id=experience.correlation_id,
                source_experience_id=experience.experience_id,
                verification_id=verification.verification_id,
                capability=experience.capability,
            )
            ids = tuple(
                self.record_claim(
                    scope=KnowledgeScope.SYSTEM_EXPERIENCE,
                    claim=claim.text,
                    status=candidate_status,
                    confidence=confidence_for(verification),
                    provenance=provenance,
                ).knowledge_id
                for claim in verification.claims
            )
            return LearningEvent(experience_id=experience.experience_id, status=LearningEventStatus.RECORDED, knowledge_ids=ids, reason=f"Verification status {verification.status.value} retained claims without promoting them to truth.", created_at=datetime.now(UTC))
        status = KnowledgeStatus.VERIFIED if verification.status is VerificationStatus.VERIFIED else KnowledgeStatus.SUPPORTED
        ids = tuple(
            self._upsert_claim(experience, claim.text, status, claim.knowledge_key)
            for claim in verification.claims
        )
        return LearningEvent(experience_id=experience.experience_id, status=LearningEventStatus.PROMOTED, knowledge_ids=ids, reason="Evidence-backed claims were promoted with deterministic confidence.", created_at=datetime.now(UTC))

    def _upsert_claim(
        self,
        experience: ExperienceRecord,
        claim: str,
        status: KnowledgeStatus,
        stable_key: str | None,
    ) -> UUID:
        key = stable_key or _claim_key(claim)
        now = datetime.now(UTC)
        provenance = Provenance(source="verification", correlation_id=experience.correlation_id, source_experience_id=experience.experience_id, verification_id=experience.verification.verification_id if experience.verification else None, capability=experience.capability)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT * FROM knowledge WHERE scope = ? AND claim_key = ? AND status IN (?, ?) ORDER BY revision DESC",
                (KnowledgeScope.SYSTEM_EXPERIENCE.value, key, KnowledgeStatus.VERIFIED.value, KnowledgeStatus.SUPPORTED.value),
            ).fetchone()
            # Same claim receives a new revision, never a silent mutation.
            if prior and prior["claim"] == claim:
                previous = LearnedKnowledge.model_validate_json(prior["payload"])
                knowledge = previous.model_copy(update={"revision": previous.revision + 1, "last_validated_at": now, "confidence": confidence_for(experience.verification, repeated_successes=previous.revision + 1, contradictions=previous.contradiction_count), "status": status, "provenance": provenance})
                self._write_knowledge(connection, knowledge, "revalidated")
                connection.execute("COMMIT")
                return knowledge.knowledge_id
            knowledge = LearnedKnowledge(knowledge_id=uuid4(), scope=KnowledgeScope.SYSTEM_EXPERIENCE, claim=claim, claim_key=key, status=status, confidence=confidence_for(experience.verification), provenance=provenance, created_at=now, last_validated_at=now, revision=1, contradiction_count=0)
            self._write_knowledge(connection, knowledge, "promoted")
            # Explicit conflicting claims use a caller-defined stable claim key in the
            # lower-level API. Text claims here only conflict when their semantic key matches.
            if prior and prior["claim"] != claim:
                previous = LearnedKnowledge.model_validate_json(prior["payload"])
                contradicted = previous.model_copy(update={"status": KnowledgeStatus.CONTRADICTED, "confidence": 0.0, "contradiction_count": previous.contradiction_count + 1, "revision": previous.revision + 1})
                self._write_knowledge(connection, contradicted, "contradicted by newer verified evidence")
                connection.execute("INSERT INTO contradictions VALUES (?, ?, ?, ?, ?, ?)", (str(uuid4()), str(previous.knowledge_id), str(knowledge.knowledge_id), str(experience.experience_id), now.isoformat(), "Newer verified claim conflicts with active claim."))
            connection.execute("COMMIT")
        return knowledge.knowledge_id

    def record_claim(
        self, *, scope: KnowledgeScope, claim: str, status: KnowledgeStatus,
        provenance: Provenance, confidence: float, claim_key: str | None = None,
    ) -> LearnedKnowledge:
        """Record non-model knowledge; VERIFIED requires a stored verification event."""
        now = datetime.now(UTC)
        knowledge = LearnedKnowledge(knowledge_id=uuid4(), scope=scope, claim=claim, claim_key=claim_key or _claim_key(claim), status=status, confidence=confidence if status is not KnowledgeStatus.CONTRADICTED else 0.0, provenance=provenance, created_at=now, last_validated_at=now if status in {KnowledgeStatus.SUPPORTED, KnowledgeStatus.VERIFIED} else None, revision=1, contradiction_count=0)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if status is KnowledgeStatus.VERIFIED:
                if provenance.source_experience_id is None or provenance.verification_id is None:
                    connection.execute("ROLLBACK")
                    raise ValueError("Verified knowledge requires a stored verification event.")
                source = connection.execute(
                    "SELECT payload FROM experiences WHERE experience_id = ?",
                    (str(provenance.source_experience_id),),
                ).fetchone()
                if source is None:
                    connection.execute("ROLLBACK")
                    raise ValueError("Verified knowledge requires a stored source experience.")
                source_experience = ExperienceRecord.model_validate_json(source["payload"])
                if (
                    source_experience.verification is None
                    or source_experience.verification.verification_id != provenance.verification_id
                    or source_experience.verification.status is not VerificationStatus.VERIFIED
                ):
                    connection.execute("ROLLBACK")
                    raise ValueError("Verified knowledge requires a matching verified source experience.")
            prior = connection.execute("SELECT * FROM knowledge WHERE scope = ? AND claim_key = ? AND status IN (?, ?)", (scope.value, knowledge.claim_key, KnowledgeStatus.VERIFIED.value, KnowledgeStatus.SUPPORTED.value)).fetchone()
            self._write_knowledge(connection, knowledge, "recorded")
            if prior and prior["claim"] != claim:
                old = LearnedKnowledge.model_validate_json(prior["payload"])
                self._write_knowledge(connection, old.model_copy(update={"status": KnowledgeStatus.CONTRADICTED, "confidence": 0.0, "contradiction_count": old.contradiction_count + 1, "revision": old.revision + 1}), "contradicted by explicit evidence")
                connection.execute("INSERT INTO contradictions VALUES (?, ?, ?, ?, ?, ?)", (str(uuid4()), str(old.knowledge_id), str(knowledge.knowledge_id), str(provenance.source_experience_id or uuid4()), now.isoformat(), "Conflicting claim recorded."))
            connection.execute("COMMIT")
        return knowledge

    def knowledge_history(self, knowledge_id: UUID) -> tuple[LearnedKnowledge, ...]:
        """Return append-only revisions explaining why a knowledge item changed."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM knowledge_revisions WHERE knowledge_id = ? ORDER BY revision",
                (str(knowledge_id),),
            ).fetchall()
        return tuple(LearnedKnowledge.model_validate_json(row["payload"]) for row in rows)

    def stale_knowledge(self, *, limit: int = 10) -> tuple[LearnedKnowledge, ...]:
        """Mark time-expired world claims stale and return a bounded recheck set."""
        now = datetime.now(UTC)
        stale: list[LearnedKnowledge] = []
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT payload FROM knowledge WHERE scope = ? AND status IN (?, ?)",
                (KnowledgeScope.WORLD.value, KnowledgeStatus.VERIFIED.value, KnowledgeStatus.SUPPORTED.value),
            ).fetchall()
            for row in rows:
                knowledge = LearnedKnowledge.model_validate_json(row["payload"])
                checked = knowledge.last_checked_at or knowledge.last_validated_at or knowledge.created_at
                if knowledge.freshness_seconds and (now - checked).total_seconds() >= knowledge.freshness_seconds:
                    updated = knowledge.model_copy(update={"status": KnowledgeStatus.STALE, "revision": knowledge.revision + 1})
                    self._write_knowledge(connection, updated, "freshness interval elapsed")
                    stale.append(updated)
                    if len(stale) >= max(1, min(limit, 25)):
                        break
            connection.execute("COMMIT")
        return tuple(stale)

    @staticmethod
    def _write_knowledge(connection: sqlite3.Connection, knowledge: LearnedKnowledge, reason: str) -> None:
        payload = knowledge.model_dump_json()
        connection.execute("INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(knowledge_id) DO UPDATE SET status=excluded.status, confidence=excluded.confidence, revision=excluded.revision, contradiction_count=excluded.contradiction_count, last_validated_at=excluded.last_validated_at, payload=excluded.payload", (str(knowledge.knowledge_id), knowledge.scope.value, knowledge.claim_key, knowledge.claim, knowledge.status.value, knowledge.confidence, knowledge.revision, knowledge.contradiction_count, knowledge.created_at.isoformat(), knowledge.last_validated_at.isoformat() if knowledge.last_validated_at else None, payload))
        connection.execute("INSERT INTO knowledge_revisions(knowledge_id, revision, changed_at, reason, payload) VALUES (?, ?, ?, ?, ?)", (str(knowledge.knowledge_id), knowledge.revision, datetime.now(UTC).isoformat(), reason, payload))

    def retrieve(self, *, query: str, scope: KnowledgeScope = KnowledgeScope.SYSTEM_EXPERIENCE, capability: str | None = None, limit: int = 5, statuses: tuple[KnowledgeStatus, ...] = (KnowledgeStatus.VERIFIED,)) -> tuple[RetrievalResult, ...]:
        """Retrieve explicitly requested states; normal callers default to verified only."""
        bounded_limit = max(1, min(limit, 10))
        if not statuses:
            return ()
        with self._lock, self._connect() as connection:
            placeholders = ",".join("?" for _ in statuses)
            rows = connection.execute(
                f"SELECT payload FROM knowledge WHERE scope = ? AND status IN ({placeholders})",
                (scope.value, *(status.value for status in statuses)),
            ).fetchall()
        results = []
        for row in rows:
            knowledge = LearnedKnowledge.model_validate_json(row["payload"])
            if capability and knowledge.provenance.capability != capability:
                continue
            relevance = _score(query, knowledge.claim)
            if relevance:
                results.append(RetrievalResult(knowledge=knowledge, relevance=relevance, retrieved_at=datetime.now(UTC)))
        return tuple(sorted(results, key=lambda item: item.relevance, reverse=True)[:bounded_limit])

    def experience(self, correlation_id: UUID) -> ExperienceRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM experiences WHERE correlation_id = ?", (str(correlation_id),)).fetchone()
        return ExperienceRecord.model_validate_json(row["payload"]) if row else None

    def capability_reliability(self, capability: str) -> dict[str, float | int] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM capability_metrics WHERE capability = ?", (capability,)).fetchone()
        if not row:
            return None
        attempts = int(row["attempts"])
        return {"attempts": attempts, "successes": int(row["successes"]), "verified": int(row["verified"]), "failures": int(row["failures"]), "success_rate": round(int(row["successes"]) / attempts, 3), "verification_rate": round(int(row["verified"]) / attempts, 3), "average_latency_ms": round(float(row["average_latency_ms"]), 3)}

    def record_web_knowledge(
        self,
        *,
        claim: str,
        claim_key: str,
        status: KnowledgeStatus,
        confidence: float,
        source_ids: tuple[UUID, ...],
        research_id: UUID,
        freshness_seconds: int,
    ) -> LearnedKnowledge:
        """Promote only research-engine evidence already corroborated in code.

        This internal entry point deliberately has no model-controlled inputs:
        research status, source identities, and confidence are produced by the
        deterministic research verifier before this method is called.
        """
        if status not in {KnowledgeStatus.SUPPORTED, KnowledgeStatus.VERIFIED}:
            raise ValueError("Web promotion requires supported or verified evidence.")
        if not source_ids:
            raise ValueError("Web promotion requires source provenance.")
        now = datetime.now(UTC)
        provenance = Provenance(
            source="web_research_verifier",
            correlation_id=research_id,
            capability="web_research",
            source_ids=source_ids,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT * FROM knowledge WHERE scope = ? AND claim_key = ? AND status IN (?, ?) ORDER BY revision DESC",
                (KnowledgeScope.WORLD.value, claim_key, KnowledgeStatus.VERIFIED.value, KnowledgeStatus.SUPPORTED.value),
            ).fetchone()
            if prior and prior["claim"] == claim:
                existing = LearnedKnowledge.model_validate_json(prior["payload"])
                knowledge = existing.model_copy(update={
                    "status": status,
                    "confidence": max(existing.confidence, confidence),
                    "provenance": provenance,
                    "revision": existing.revision + 1,
                    "last_validated_at": now,
                    "last_checked_at": now,
                    "freshness_seconds": freshness_seconds,
                })
                self._write_knowledge(connection, knowledge, "revalidated by corroborated web evidence")
            else:
                knowledge = LearnedKnowledge(
                    knowledge_id=uuid4(), scope=KnowledgeScope.WORLD, claim=claim,
                    claim_key=claim_key, status=status, confidence=confidence,
                    provenance=provenance, created_at=now, last_validated_at=now,
                    revision=1, contradiction_count=0, valid_from=now,
                    last_checked_at=now, freshness_seconds=freshness_seconds,
                )
                self._write_knowledge(connection, knowledge, "promoted from corroborated web evidence")
                if prior:
                    old = LearnedKnowledge.model_validate_json(prior["payload"])
                    contradicted = old.model_copy(update={
                        "status": KnowledgeStatus.CONTRADICTED, "confidence": 0.0,
                        "contradiction_count": old.contradiction_count + 1,
                        "revision": old.revision + 1,
                    })
                    self._write_knowledge(connection, contradicted, "contradicted by newer web evidence")
                    connection.execute("INSERT INTO contradictions VALUES (?, ?, ?, ?, ?, ?)", (str(uuid4()), str(old.knowledge_id), str(knowledge.knowledge_id), str(research_id), now.isoformat(), "Newer corroborated web claim conflicts with prior knowledge."))
            connection.execute("COMMIT")
        return knowledge


def experience_from_execution(
    *, correlation_id: UUID, assessment: TaskAssessment, execution: Any,
    lifecycle_outcome: str, observation: ExecutionObservation | None = None,
    action_result: ActionResult | None = None, failure_reason: str | None = None,
    model_provider: str | None = None,
) -> ExperienceRecord:
    """Build the only persisted shape from typed runtime objects."""
    verification = execution.verification
    action = None
    if action_result is not None:
        action = sanitize_for_persistence({"action_id": str(action_result.action_id), "duration_ms": action_result.duration_ms, "output": action_result.output, "output_truncated": action_result.output_truncated, "error_code": action_result.error.code.value if action_result.error else None})
    return ExperienceRecord(correlation_id=correlation_id, created_at=datetime.now(UTC), normalized_goal=assessment.user_goal, assessment=assessment, route=execution.route.value, execution_status=execution.status.value, lifecycle_outcome=lifecycle_outcome, capability=execution.capability_name, operation=execution.capability_operation, action_status=execution.action_status, action=action, observation=observation, verification=verification, failure_reason=failure_reason, response_outcome=execution.response_generation_status, model_provider=model_provider, confidence=confidence_for(verification), provenance=Provenance(source="coordinator", correlation_id=correlation_id, capability=execution.capability_name))
