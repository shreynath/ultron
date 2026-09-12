"""Local-only diagnostics, memory, and explicit learning controls."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.local_intelligence import get_local_orchestrator
from app.local_intelligence.memory import (
    FeedbackEvent,
    LearningEventStore,
    LocalMemory,
    LocalMemoryStore,
    MemoryKind,
)

router = APIRouter(prefix="/api/local-intelligence", tags=["local-intelligence"])
memory_store = LocalMemoryStore()
learning_store = LearningEventStore()


class CurateRequest(BaseModel):
    approve: bool


class DatasetRequest(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


@router.get("/status")
async def status():
    return await get_local_orchestrator().diagnostics()


@router.post("/memory", response_model=LocalMemory)
async def save_memory(memory: LocalMemory):
    try:
        return memory_store.save(memory)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/memory")
async def retrieve_memory(
    query: str = Query(min_length=1, max_length=1_000),
    kind: list[MemoryKind] = Query(default=[]),
    limit: int = Query(default=5, ge=1, le=20),
):
    return {"items": memory_store.retrieve(query, kinds=tuple(kind) or None, limit=limit), "retrieval": "local_lexical"}


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
