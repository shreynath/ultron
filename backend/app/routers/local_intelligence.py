"""Local-only diagnostics, memory, and explicit learning controls."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.foundation.phase_o import PhaseOEngine
from app.local_intelligence import get_local_orchestrator
from app.local_intelligence.memory import (
    FeedbackEvent,
    LearningEventStore,
    LocalMemory,
    LocalMemoryStore,
    MemoryKind,
)
from app.local_intelligence.study_queue import (
    AutonomyTier,
    CalibrationGrade,
    RunnerConfig,
    StudyLearningEvent,
    StudyQueueController,
    StudyQueueStore,
    StudyTarget,
)

router = APIRouter(prefix="/api/local-intelligence", tags=["local-intelligence"])
memory_store = LocalMemoryStore()
learning_store = LearningEventStore()
study_queue = StudyQueueController()


class CurateRequest(BaseModel):
    approve: bool


class DatasetRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class StudyRunRequest(BaseModel):
    supervisor_approved: bool = False


class EvaluatorCalibrationRequest(BaseModel):
    stage: AutonomyTier
    grades: tuple[CalibrationGrade, ...]


class RunnerRequest(BaseModel):
    max_cycles: int = Field(default=1, ge=1, le=100)


@router.get("/status")
async def status():
    return await get_local_orchestrator().diagnostics()


@router.get("/development-status")
async def development_status():
    """Read persistent proprietary-model development state without starting work."""
    return PhaseOEngine(Path(settings.ultron_home)).monitor()


@router.post("/memory", response_model=LocalMemory)
async def save_memory(memory: LocalMemory):
    try:
        return memory_store.save(memory)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/memory")
async def retrieve_memory(
    query: str = Query(min_length=1, max_length=1_000),
    kind: list[MemoryKind] | None = Query(default=None),  # noqa: B008 - FastAPI declares query metadata in the signature.
    limit: int = Query(default=5, ge=1, le=20),
):
    return {"items": memory_store.retrieve(query, kinds=tuple(kind or ()) or None, limit=limit), "retrieval": "local_lexical"}


@router.delete("/memory/{memory_id}", response_model=LocalMemory)
async def forget_memory(memory_id: UUID):
    forgotten = memory_store.forget(memory_id)
    if forgotten is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return forgotten


@router.post("/feedback", response_model=FeedbackEvent)
async def record_feedback(event: FeedbackEvent):
    try:
        return learning_store.record(event)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/feedback/{feedback_id}/curate", response_model=FeedbackEvent)
async def curate_feedback(feedback_id: UUID, body: CurateRequest):
    try:
        return learning_store.curate(feedback_id, approve=body.approve)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Learning event not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/datasets")
async def create_training_dataset(body: DatasetRequest):
    try:
        path = learning_store.training_dataset(dataset_id=body.dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"dataset_id": body.dataset_id, "path": str(path), "state": "curated", "training_started": False}


@router.post("/study-queue", response_model=StudyTarget)
async def enqueue_study_target(target: StudyTarget):
    return study_queue.queue.enqueue(target)


@router.get("/study-queue")
async def study_queue_status():
    store = StudyQueueStore()
    next_target = store.next_target()
    return {"next_target": next_target, "next_target_collision_risk": store.target_risk(next_target.target_id) if next_target else None, "events": store.events(), **store.observability()}


@router.post("/study-queue/run-next", response_model=StudyLearningEvent | None)
async def run_next_study_target(body: StudyRunRequest):
    try:
        return study_queue.run_next(supervisor_approved=body.supervisor_approved)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/study-queue/evaluator-calibration")
async def record_evaluator_calibration(body: EvaluatorCalibrationRequest):
    try:
        round_ = StudyQueueStore().record_calibration_round(stage=body.stage, grades=body.grades)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return round_


@router.post("/study-queue/run-available")
async def run_available_study_targets(body: RunnerRequest):
    return {"events": study_queue.run_available(max_cycles=body.max_cycles), "observability": StudyQueueStore().observability()}


@router.put("/study-queue/runner-config", response_model=RunnerConfig)
async def set_study_runner_config(config: RunnerConfig):
    return StudyQueueStore().set_runner_config(config)


@router.post("/study-queue/circuit-breaker/resume")
async def resume_study_queue_after_human_action():
    try:
        StudyQueueStore().resume_after_human_action()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StudyQueueStore().observability()
