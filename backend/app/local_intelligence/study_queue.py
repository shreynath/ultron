"""Supervised, externally-grounded Level 1 memory-learning experiments.

This module deliberately does not use generated text as either a source or an
evaluator.  A queue item has an external source, a deterministic answer key,
and exact/paraphrase retests after the acquisition context is discarded.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.local_intelligence.core import ensure_local_layout
from app.local_intelligence.memory import LocalMemory, LocalMemoryStore, MemoryKind


class StudyQueueStatus(str, Enum):
    QUEUED = "QUEUED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class CollisionRisk(str, Enum):
    UNASSESSED = "UNASSESSED"
    LOW = "LOW_COLLISION_RISK"
    HIGH = "HIGH_COLLISION_RISK"
    SYNTHETIC = "SYNTHETIC_BY_CONSTRUCTION"


class AutonomyTier(str, Enum):
    SUPERVISED = "SUPERVISED"
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REVIEWED = "HUMAN_REVIEWED"


class LearningDecision(str, Enum):
    LEARNING_DEMONSTRATED = "LEARNING_DEMONSTRATED"
    LEARNING_FAILED = "LEARNING_FAILED"
    UNKNOWN = "UNKNOWN"


class EvaluatorCalibrationStatus(str, Enum):
    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"


class StudyTarget(BaseModel):
    """A small, independently reproducible retrieval-learning experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: UUID = Field(default_factory=uuid4)
    subject: str = Field(min_length=1, max_length=200)
    baseline_question: str = Field(min_length=1, max_length=1_000)
    paraphrase_question: str = Field(min_length=1, max_length=1_000)
    expected_answer: str = Field(min_length=1, max_length=500)
    candidate_information: str = Field(min_length=1, max_length=4_000)
    source_url: str = Field(min_length=1, max_length=2_048)
    source_type: str = Field(default="EXTERNAL_WEB_DOCUMENT", min_length=1, max_length=64)
    source_category: str = Field(default="OTHER_EXTERNAL", min_length=1, max_length=64)
    source_credibility: str = Field(min_length=1, max_length=500)
    constructed_source: str | None = Field(default=None, max_length=20_000)
    requires_supervision: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    answer: str
    correct: bool
    evaluator: str = "DETERMINISTIC_NORMALIZED_SUBSTRING_V1"


class StudyLearningEvent(BaseModel):
    """Versioned event with enough evidence to reproduce a Level 1 claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    schema_version: str = "ultron.study-learning-event.v1"
    target_id: UUID
    subject: str
    trigger: str = "STUDY_QUEUE"
    source_id: str
    source_type: str
    source_provenance: dict[str, str]
    candidate_information: str
    validation_status: str
    validation_method: str
    confidence: float = Field(ge=0, le=1)
    memory_ids: tuple[UUID, ...] = ()
    model_id: str | None = None
    model_revision: str | None = None
    embedding_model_id: str | None = None
    embedding_revision: str | None = None
    pre_test_result: TestResult
    post_test_result: TestResult | None = None
    generalization_test_result: TestResult | None = None
    context_reset_id: UUID | None = None
    decision: LearningDecision
    execution_tier: AutonomyTier = AutonomyTier.SUPERVISED
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    evidence_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CollisionProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: UUID
    variants: tuple[str, ...] = Field(min_length=3, max_length=8)
    answers: tuple[str, ...] = Field(min_length=3, max_length=8)
    risk: CollisionRisk
    prober_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CalibrationGrade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID
    human_decision: LearningDecision
    reviewer_id: str = Field(min_length=1, max_length=128)


class CalibrationRound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: UUID = Field(default_factory=uuid4)
    stage: AutonomyTier
    grades: tuple[CalibrationGrade, ...]
    agreement: float = Field(ge=0, le=1)
    wilson_interval: tuple[float, float]
    synthetic_or_low_count: int = Field(ge=0)
    passed: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cycles_per_hour: int = Field(default=6, ge=1, le=100)
    fetches_per_hour: int = Field(default=6, ge=1, le=100)
    fetches_per_day: int = Field(default=24, ge=1, le=1_000)
    consecutive_terminal_failures: int = Field(default=3, ge=1, le=20)
    fetch_failure_window: int = Field(default=10, ge=3, le=100)
    fetch_failure_rate: float = Field(default=0.5, gt=0, le=1)
    stage_1_min_hours: int = Field(default=48, ge=1, le=720)
    stage_1_min_cycles: int = Field(default=20, ge=1, le=1_000)
    stage_2_spot_check_every: int = Field(default=20, ge=1, le=1_000)


class CircuitBreaker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tripped: bool = False
    reason: str | None = None
    tripped_at: datetime | None = None


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return (round(max(0.0, centre - radius), 4), round(min(1.0, centre + radius), 4))


def _normal(value: str) -> str:
    return " ".join("".join(character if character.isalnum() else " " for character in value.lower()).split())


def _answer_is_present(answer: str, text: str) -> bool:
    return _normal(answer) in _normal(text)


class StudyQueueStore:
    """SQLite queue/event store; completed evidence is append-only by event id."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = ensure_local_layout(root)
        self.path = self.root / "learning" / "study_queue.sqlite3"
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS study_targets (target_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS study_events (event_id TEXT PRIMARY KEY, target_id TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluator_calibration (calibration_id INTEGER PRIMARY KEY CHECK(calibration_id = 1), status TEXT NOT NULL, human_sample_size INTEGER NOT NULL, agreement REAL NOT NULL, recorded_at TEXT NOT NULL, note TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS study_target_metadata (target_id TEXT PRIMARY KEY, collision_payload TEXT, source_category TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calibration_rounds (round_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runner_state (state_id INTEGER PRIMARY KEY CHECK(state_id = 1), tier TEXT NOT NULL, stage_1_unlocked_at TEXT, stage_1_cycles INTEGER NOT NULL, stage_2_cycles INTEGER NOT NULL, circuit_payload TEXT NOT NULL, config_payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runner_actions (action_id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, action TEXT NOT NULL, target_id TEXT, detail TEXT NOT NULL);
            """)
            connection.execute("INSERT OR IGNORE INTO runner_state VALUES (1, ?, NULL, 0, 0, ?, ?)", (AutonomyTier.SUPERVISED.value, CircuitBreaker().model_dump_json(), RunnerConfig().model_dump_json()))
            rows = connection.execute("SELECT target_id, payload FROM study_targets").fetchall()
            for row in rows:
                target = StudyTarget.model_validate_json(row["payload"])
                category = target.source_category
                if category == "OTHER_EXTERNAL" and urlparse(target.source_url).netloc == "www.rfc-editor.org":
                    category = "RFC_EDITOR"
                connection.execute("INSERT OR IGNORE INTO study_target_metadata VALUES (?, NULL, ?)", (row["target_id"], category))
                if category != target.source_category:
                    connection.execute("UPDATE study_target_metadata SET source_category = ? WHERE target_id = ?", (category, row["target_id"]))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def enqueue(self, target: StudyTarget) -> StudyTarget:
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO study_targets VALUES (?, ?, ?, ?)", (str(target.target_id), StudyQueueStatus.QUEUED.value, target.created_at.isoformat(), target.model_dump_json()))
            connection.execute("INSERT INTO study_target_metadata VALUES (?, NULL, ?)", (str(target.target_id), target.source_category))
        return target

    def next_target(self) -> StudyTarget | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM study_targets WHERE status = ? ORDER BY created_at, target_id LIMIT 1", (StudyQueueStatus.QUEUED.value,)).fetchone()
        return StudyTarget.model_validate_json(row["payload"]) if row else None

    def next_eligible_target(self) -> StudyTarget | None:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM study_targets WHERE status = ? ORDER BY created_at, target_id", (StudyQueueStatus.QUEUED.value,)).fetchall()
        for row in rows:
            target = StudyTarget.model_validate_json(row["payload"])
            if self.target_risk(target.target_id) in {CollisionRisk.LOW, CollisionRisk.SYNTHETIC}:
                return target
        return None

    def collision_probe(self, target_id: UUID) -> CollisionProbe | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT collision_payload FROM study_target_metadata WHERE target_id = ?", (str(target_id),)).fetchone()
        return CollisionProbe.model_validate_json(row["collision_payload"]) if row and row["collision_payload"] else None

    def record_collision_probe(self, probe: CollisionProbe) -> CollisionProbe:
        if len(probe.variants) != len(probe.answers):
            raise ValueError("Collision probe variants and answers must align.")
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT target_id FROM study_targets WHERE target_id = ?", (str(probe.target_id),)).fetchone()
            if row is None:
                raise KeyError("Unknown study target.")
            connection.execute("UPDATE study_target_metadata SET collision_payload = ? WHERE target_id = ?", (probe.model_dump_json(), str(probe.target_id)))
        return probe

    def target_risk(self, target_id: UUID) -> CollisionRisk:
        probe = self.collision_probe(target_id)
        return probe.risk if probe else CollisionRisk.UNASSESSED

    def target_source_category(self, target_id: UUID) -> str:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT source_category FROM study_target_metadata WHERE target_id = ?", (str(target_id),)).fetchone()
        return str(row["source_category"]) if row else "UNCLASSIFIED"

    def tier(self) -> AutonomyTier:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT tier FROM runner_state WHERE state_id = 1").fetchone()
        return AutonomyTier(row["tier"])

    def runner_config(self) -> RunnerConfig:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT config_payload FROM runner_state WHERE state_id = 1").fetchone()
        return RunnerConfig.model_validate_json(row["config_payload"])

    def set_runner_config(self, config: RunnerConfig) -> RunnerConfig:
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE runner_state SET config_payload = ? WHERE state_id = 1", (config.model_dump_json(),))
        return config

    def circuit_breaker(self) -> CircuitBreaker:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT circuit_payload FROM runner_state WHERE state_id = 1").fetchone()
        return CircuitBreaker.model_validate_json(row["circuit_payload"])

    def log_action(self, action: str, *, target_id: UUID | None = None, detail: str = "") -> None:
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO runner_actions(occurred_at, action, target_id, detail) VALUES (?, ?, ?, ?)", (datetime.now(UTC).isoformat(), action, str(target_id) if target_id else None, detail[:2_000]))

    def action_count_since(self, action: str, *, seconds: int) -> int:
        cutoff = datetime.fromtimestamp(datetime.now(UTC).timestamp() - seconds, UTC).isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM runner_actions WHERE action = ? AND occurred_at >= ?", (action, cutoff)).fetchone()
        return int(row["count"])

    def trip_circuit(self, reason: str) -> None:
        breaker = CircuitBreaker(tripped=True, reason=reason, tripped_at=datetime.now(UTC))
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE runner_state SET tier = ?, circuit_payload = ? WHERE state_id = 1", (AutonomyTier.CIRCUIT_BREAKER_TRIPPED.value, breaker.model_dump_json()))
        self.log_action("CIRCUIT_BREAKER_TRIPPED", detail=reason)

    def resume_after_human_action(self) -> None:
        if not self.circuit_breaker().tripped:
            raise ValueError("Circuit breaker is not tripped.")
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE runner_state SET tier = ?, circuit_payload = ? WHERE state_id = 1", (AutonomyTier.SUPERVISED.value, CircuitBreaker().model_dump_json()))
        self.log_action("CIRCUIT_BREAKER_HUMAN_RESET")

    def calibration(self) -> EvaluatorCalibrationStatus:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT status FROM evaluator_calibration WHERE calibration_id = 1").fetchone()
        return EvaluatorCalibrationStatus(row["status"]) if row else EvaluatorCalibrationStatus.UNVALIDATED

    def calibration_rounds(self) -> tuple[CalibrationRound, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM calibration_rounds ORDER BY created_at").fetchall()
        return tuple(CalibrationRound.model_validate_json(row["payload"]) for row in rows)

    def record_calibration_round(self, *, stage: AutonomyTier, grades: tuple[CalibrationGrade, ...]) -> CalibrationRound:
        if stage not in {AutonomyTier.STAGE_1, AutonomyTier.STAGE_2}:
            raise ValueError("Calibration rounds target Stage 1 or Stage 2 only.")
        minimum = 10 if stage is AutonomyTier.STAGE_1 else 30
        if len(grades) < minimum or len({grade.event_id for grade in grades}) != len(grades):
            raise ValueError(f"{stage.value} requires {minimum} uniquely human-graded events.")
        events = {event.event_id: event for event in self.events(limit=100)}
        eligible = 0
        agreements = 0
        for grade in grades:
            event = events.get(grade.event_id)
            if event is None or self.target_risk(event.target_id) not in {CollisionRisk.LOW, CollisionRisk.SYNTHETIC}:
                raise ValueError("Calibration cannot include high-collision or unassessed targets.")
            eligible += 1
            agreements += int(event.decision is grade.human_decision)
        agreement = agreements / len(grades)
        majority_ok = eligible * 2 > len(grades)
        passed = agreement >= 0.90 and majority_ok
        round_ = CalibrationRound(stage=stage, grades=grades, agreement=agreement, wilson_interval=_wilson(agreements, len(grades)), synthetic_or_low_count=eligible, passed=passed)
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO calibration_rounds VALUES (?, ?, ?)", (str(round_.round_id), round_.created_at.isoformat(), round_.model_dump_json()))
            if stage is AutonomyTier.STAGE_1 and passed:
                connection.execute("UPDATE runner_state SET tier = ?, stage_1_unlocked_at = ?, stage_1_cycles = 0 WHERE state_id = 1", (AutonomyTier.STAGE_1.value, round_.created_at.isoformat()))
            elif stage is AutonomyTier.STAGE_2 and passed:
                state = connection.execute("SELECT stage_1_unlocked_at, stage_1_cycles, config_payload FROM runner_state WHERE state_id = 1").fetchone()
                config = RunnerConfig.model_validate_json(state["config_payload"])
                prior_rows = connection.execute("SELECT payload FROM calibration_rounds").fetchall()
                prior_dates = {
                    CalibrationRound.model_validate_json(row["payload"]).created_at.date()
                    for row in prior_rows
                    if CalibrationRound.model_validate_json(row["payload"]).stage is AutonomyTier.STAGE_2
                    and CalibrationRound.model_validate_json(row["payload"]).round_id != round_.round_id
                }
                started = datetime.fromisoformat(state["stage_1_unlocked_at"]) if state["stage_1_unlocked_at"] else None
                elapsed_hours = (datetime.now(UTC) - started).total_seconds() / 3600 if started else 0
                if not any(day != round_.created_at.date() for day in prior_dates) or (elapsed_hours < config.stage_1_min_hours and state["stage_1_cycles"] < config.stage_1_min_cycles):
                    raise ValueError("Stage 2 requires a prior independent Stage 2 round plus the configured Stage 1 duration or cycle evidence.")
                connection.execute("UPDATE runner_state SET tier = ?, stage_2_cycles = 0 WHERE state_id = 1", (AutonomyTier.STAGE_2.value,))
        self.log_action("CALIBRATION_ROUND_RECORDED", detail=f"{stage.value}; n={len(grades)}; agreement={agreement:.3f}; passed={passed}")
        return round_

    def complete(self, event: StudyLearningEvent) -> StudyLearningEvent:
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO study_events VALUES (?, ?, ?, ?)", (str(event.event_id), str(event.target_id), event.created_at.isoformat(), event.model_dump_json()))
            connection.execute("UPDATE study_targets SET status = ? WHERE target_id = ?", (StudyQueueStatus.COMPLETED.value, str(event.target_id)))
        return event

    def events(self, *, limit: int = 20) -> tuple[StudyLearningEvent, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM study_events ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
        return tuple(StudyLearningEvent.model_validate_json(row["payload"]) for row in rows)

    def targets(self, *, limit: int = 100) -> tuple[StudyTarget, ...]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM study_targets ORDER BY created_at, target_id LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
        return tuple(StudyTarget.model_validate_json(row["payload"]) for row in rows)

    def observability(self) -> dict[str, object]:
        events = self.events(limit=100)
        counts = {decision.value: sum(event.decision is decision for event in events) for decision in LearningDecision}
        tier_counts = {tier.value: sum(event.execution_tier is tier for event in events) for tier in AutonomyTier}
        with self._lock, self._connect() as connection:
            state = connection.execute("SELECT stage_1_unlocked_at, stage_1_cycles, stage_2_cycles FROM runner_state WHERE state_id = 1").fetchone()
            trips = connection.execute("SELECT occurred_at, detail FROM runner_actions WHERE action = 'CIRCUIT_BREAKER_TRIPPED' ORDER BY action_id DESC").fetchall()
        return {
            "tier": self.tier(), "calibration_rounds": self.calibration_rounds(), "circuit_breaker": self.circuit_breaker(),
            "circuit_breaker_history": [{"occurred_at": row["occurred_at"], "reason": row["detail"]} for row in trips],
            "runner_config": self.runner_config(), "event_counts": counts, "event_counts_by_tier": tier_counts,
            "queue_targets": [{"target": target, "source_category": self.target_source_category(target.target_id), "collision_risk": self.target_risk(target.target_id)} for target in self.targets()],
            "stage_1_unlocked_at": state["stage_1_unlocked_at"], "stage_1_cycles": state["stage_1_cycles"], "stage_2_cycles": state["stage_2_cycles"],
            "rate_utilization": {"cycles_last_hour": self.action_count_since("CYCLE_STARTED", seconds=3600), "external_fetches_last_hour": self.action_count_since("EXTERNAL_FETCH", seconds=3600), "external_fetches_last_day": self.action_count_since("EXTERNAL_FETCH", seconds=86400)},
        }


Fetcher = Callable[[str], str]
BaselineProber = Callable[[str], str]


def fetch_external_document(url: str) -> str:
    """Fetch direct HTTPS document bytes; no model output participates."""
    request = Request(url, headers={"User-Agent": "ULTRON-study-queue/1.0"})
    with urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise ValueError(f"External source returned HTTP {response.status}.")
        return response.read(2_000_000).decode("utf-8", errors="replace")


class StudyQueueController:
    """Runs one supervised or calibrated queue item with deterministic scoring."""

    def __init__(self, root: Path | None = None, *, fetcher: Fetcher = fetch_external_document) -> None:
        self.root = ensure_local_layout(root)
        self.queue = StudyQueueStore(self.root)
        self.memory = LocalMemoryStore(self.root)
        self.fetcher = fetcher

    def assess_collision(self, target: StudyTarget, *, prober: BaselineProber, prober_id: str) -> CollisionProbe:
        """Probe three natural variants before allowing a public item into calibration."""
        variants = (target.baseline_question, target.paraphrase_question, f"State the answer briefly: {target.subject}.")
        answers = tuple(prober(question) for question in variants)
        risk = CollisionRisk.HIGH if any(_answer_is_present(target.expected_answer, answer) for answer in answers) else CollisionRisk.LOW
        probe = CollisionProbe(target_id=target.target_id, variants=variants, answers=answers, risk=risk, prober_id=prober_id)
        return self.queue.record_collision_probe(probe)

    def register_synthetic_collision_clearance(self, target: StudyTarget) -> CollisionProbe:
        if target.source_type != "SYNTHETIC_CONSTRUCTED_DOCUMENT" or not target.constructed_source:
            raise ValueError("Only constructed synthetic documents receive synthetic collision clearance.")
        probe = CollisionProbe(target_id=target.target_id, variants=(target.baseline_question, target.paraphrase_question, f"State the answer briefly: {target.subject}."), answers=("UNKNOWN", "UNKNOWN", "UNKNOWN"), risk=CollisionRisk.SYNTHETIC, prober_id="SYNTHETIC_BY_CONSTRUCTION_V1")
        return self.queue.record_collision_probe(probe)

    @staticmethod
    def _test(memory: LocalMemoryStore, question: str, expected_answer: str) -> TestResult:
        matches = memory.retrieve(question, kinds=(MemoryKind.SEMANTIC,), limit=1)
        answer = matches[0].content if matches else "UNKNOWN"
        return TestResult(question=question, answer=answer, correct=_answer_is_present(expected_answer, answer))

    def run_next(self, *, supervisor_approved: bool = False, prefer_calibration_target: bool = False) -> StudyLearningEvent | None:
        tier = self.queue.tier()
        if tier is AutonomyTier.CIRCUIT_BREAKER_TRIPPED:
            raise PermissionError("Study queue circuit breaker is tripped and requires explicit human reset.")
        target = self.queue.next_eligible_target() if prefer_calibration_target or not supervisor_approved else self.queue.next_target()
        if target is None:
            if not supervisor_approved and self.queue.next_target() is not None:
                self.queue.log_action("SKIPPED_COLLISION_RISK", detail="no low-collision or synthetic target is eligible")
                raise PermissionError("Study queue is supervised until evaluator agreement has been validated against a human-graded sample.")
            return None
        if not supervisor_approved and tier is AutonomyTier.SUPERVISED:
            raise PermissionError("Study queue is supervised until evaluator agreement has been validated against a human-graded sample.")
        if not supervisor_approved and self.queue.target_risk(target.target_id) not in {CollisionRisk.LOW, CollisionRisk.SYNTHETIC}:
            self.queue.log_action("SKIPPED_COLLISION_RISK", target_id=target.target_id, detail=self.queue.target_risk(target.target_id).value)
            raise PermissionError("Autonomous study requires a low-collision or synthetic target.")
        baseline = self._test(self.memory, target.baseline_question, target.expected_answer)
        try:
            content = target.constructed_source if target.source_type == "SYNTHETIC_CONSTRUCTED_DOCUMENT" else self.fetcher(target.source_url)
            if content is None:
                raise ValueError("Synthetic target has no constructed source.")
            parsed = urlparse(target.source_url)
            external_valid = parsed.scheme == "https" and bool(parsed.netloc)
            synthetic_valid = target.source_type == "SYNTHETIC_CONSTRUCTED_DOCUMENT" and parsed.scheme == "synthetic"
            if not (external_valid or synthetic_valid) or not _answer_is_present(target.expected_answer, content):
                raise ValueError("Source provenance or deterministic ground-truth support failed validation.")
        except (OSError, ValueError) as exc:
            event = StudyLearningEvent(target_id=target.target_id, subject=target.subject, source_id=target.source_url, source_type=target.source_type, source_provenance={"url": target.source_url}, candidate_information=target.candidate_information, validation_status="FAILED", validation_method="EXTERNAL_SOURCE_GROUND_TRUTH_V1", confidence=0.0, pre_test_result=baseline, decision=LearningDecision.UNKNOWN, execution_tier=tier, review_status=ReviewStatus.REQUIRED if tier is AutonomyTier.STAGE_1 else ReviewStatus.NOT_REQUIRED, evidence_ids=(f"source-error:{type(exc).__name__}",))
            return self._complete_and_check(event)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_cache = self.root / "learning" / "study_sources"
        source_cache.mkdir(parents=True, exist_ok=True)
        snapshot = source_cache / f"{digest}.txt"
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content:
            raise ValueError("Content-addressed study-source cache collision.")
        if not snapshot.exists():
            snapshot.write_text(content, encoding="utf-8")
        memory = self.memory.save(LocalMemory(kind=MemoryKind.SEMANTIC, content=target.candidate_information, source=f"external:{parsed.netloc}", importance=8, tags=("study_queue", str(target.target_id))))
        # A fresh store is a context reset: it can observe persistence but cannot
        # access the acquisition call's local variables or conversation state.
        reset_memory = LocalMemoryStore(self.root)
        exact = self._test(reset_memory, target.baseline_question, target.expected_answer)
        paraphrase = self._test(reset_memory, target.paraphrase_question, target.expected_answer)
        decision = LearningDecision.LEARNING_DEMONSTRATED if not baseline.correct and exact.correct and paraphrase.correct else LearningDecision.LEARNING_FAILED
        event = StudyLearningEvent(target_id=target.target_id, subject=target.subject, source_id=target.source_url, source_type=target.source_type, source_provenance={"url": target.source_url, "host": parsed.netloc or "constructed", "sha256": digest, "snapshot_path": str(snapshot), "credibility": target.source_credibility}, candidate_information=target.candidate_information, validation_status="VERIFIED", validation_method="EXTERNAL_SOURCE_GROUND_TRUTH_V1 + DETERMINISTIC_RETRIEVAL_RETEST_V1", confidence=0.95 if decision is LearningDecision.LEARNING_DEMONSTRATED else 0.20, memory_ids=(memory.memory_id,), pre_test_result=baseline, post_test_result=exact, generalization_test_result=paraphrase, context_reset_id=uuid4(), decision=decision, execution_tier=tier, review_status=ReviewStatus.REQUIRED if tier is AutonomyTier.STAGE_1 else ReviewStatus.NOT_REQUIRED, evidence_ids=(target.source_url, f"sha256:{digest}"))
        return self._complete_and_check(event)

    def _complete_and_check(self, event: StudyLearningEvent) -> StudyLearningEvent:
        completed = self.queue.complete(event)
        self.queue.log_action("CYCLE_COMPLETED", target_id=completed.target_id, detail=completed.decision.value)
        config = self.queue.runner_config()
        with self.queue._lock, self.queue._connect() as connection:
            if completed.execution_tier is AutonomyTier.STAGE_1:
                connection.execute("UPDATE runner_state SET stage_1_cycles = stage_1_cycles + 1 WHERE state_id = 1")
            elif completed.execution_tier is AutonomyTier.STAGE_2:
                connection.execute("UPDATE runner_state SET stage_2_cycles = stage_2_cycles + 1 WHERE state_id = 1")
                state = connection.execute("SELECT stage_2_cycles FROM runner_state WHERE state_id = 1").fetchone()
                if state["stage_2_cycles"] >= config.stage_2_spot_check_every:
                    connection.execute("UPDATE runner_state SET tier = ?, stage_2_cycles = 0 WHERE state_id = 1", (AutonomyTier.STAGE_1.value,))
                    self.queue.log_action("STAGE_2_REVERTED_RECALIBRATION_REQUIRED")
        recent = self.queue.events(limit=config.consecutive_terminal_failures)
        if len(recent) == config.consecutive_terminal_failures and all(item.decision in {LearningDecision.LEARNING_FAILED, LearningDecision.UNKNOWN} for item in recent):
            self.queue.trip_circuit(f"{config.consecutive_terminal_failures} consecutive LEARNING_FAILED/UNKNOWN events")
        window = self.queue.events(limit=config.fetch_failure_window)
        failures = sum(item.validation_status == "FAILED" for item in window)
        if len(window) >= config.fetch_failure_window and failures / len(window) > config.fetch_failure_rate:
            self.queue.trip_circuit(f"external fetch/validation failure rate {failures}/{len(window)} exceeded configured threshold")
        return completed

    def run_available(self, *, max_cycles: int = 1) -> tuple[StudyLearningEvent, ...]:
        """Rate-limited autonomous runner. It never promotes or auto-resumes."""
        config = self.queue.runner_config()
        if self.queue.tier() not in {AutonomyTier.STAGE_1, AutonomyTier.STAGE_2}:
            self.queue.log_action("RUNNER_HELD_TIER", detail=self.queue.tier().value)
            return ()
        results: list[StudyLearningEvent] = []
        for _ in range(min(max_cycles, config.cycles_per_hour)):
            if self.queue.action_count_since("CYCLE_STARTED", seconds=3600) >= config.cycles_per_hour:
                self.queue.log_action("RATE_LIMIT_HOLD", detail="cycles_per_hour")
                break
            target = self.queue.next_eligible_target()
            if target is None:
                self.queue.log_action("RUNNER_IDLE")
                break
            if self.queue.target_risk(target.target_id) not in {CollisionRisk.LOW, CollisionRisk.SYNTHETIC}:
                self.queue.log_action("SKIPPED_COLLISION_RISK", target_id=target.target_id, detail=self.queue.target_risk(target.target_id).value)
                break
            if target.source_type != "SYNTHETIC_CONSTRUCTED_DOCUMENT":
                if self.queue.action_count_since("EXTERNAL_FETCH", seconds=3600) >= config.fetches_per_hour or self.queue.action_count_since("EXTERNAL_FETCH", seconds=86400) >= config.fetches_per_day:
                    self.queue.log_action("RATE_LIMIT_HOLD", target_id=target.target_id, detail="external_fetch")
                    break
                self.queue.log_action("EXTERNAL_FETCH", target_id=target.target_id)
            self.queue.log_action("CYCLE_STARTED", target_id=target.target_id)
            event = self.run_next()
            if event:
                results.append(event)
            if self.queue.circuit_breaker().tripped:
                break
        return tuple(results)


class SyntheticTargetGenerator:
    """Makes collision-resistant, locally constructed calibration targets."""

    @staticmethod
    def generate(index: int) -> StudyTarget:
        identifier = f"ZK-{index:04d}-ORCHID"
        value = f"velora-{index:04d}"
        fact = f"In fictional protocol {identifier}, the routing color is {value}."
        return StudyTarget(subject=f"fictional protocol {identifier} routing color", baseline_question=f"What is the routing color in protocol {identifier}?", paraphrase_question=f"Name the color assigned to {identifier} routing.", expected_answer=value, candidate_information=fact, source_url=f"synthetic://study-queue/{identifier}", source_type="SYNTHETIC_CONSTRUCTED_DOCUMENT", source_category="SYNTHETIC_CALIBRATION", source_credibility="Locally constructed non-real calibration document.", constructed_source=f"Calibration source v1. {fact} This identifier and fact are invented solely for evaluator calibration.", requires_supervision=True)
