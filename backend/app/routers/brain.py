from __future__ import annotations

import hmac
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.brain import (
    BoundedExecutionCoordinator,
    CognitiveExecution,
    CognitiveRuntime,
    CoordinatorExecution,
    Executive,
    bound_history,
)
from app.config import settings
from app.brain.learning import KnowledgeScope, KnowledgeStatus
from app.brain.research import ResearchBudget, ResearchEngine, ResearchManager, ResearchTask

router = APIRouter(prefix="/api/brain", tags=["brain"])
executive = Executive()
runtime = CognitiveRuntime()
coordinator = BoundedExecutionCoordinator(runtime=runtime)
research_manager = ResearchManager(ResearchEngine(repository=coordinator.learning_repository))


class BrainRunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    budget: ResearchBudget | None = None


def require_research_token(
    x_ultron_research_token: str | None = Header(default=None),
) -> None:
    """Control and audit endpoints stay off until deployment config enables them."""
    if not settings.brain_research_api_token or not x_ultron_research_token:
        raise HTTPException(status_code=503, detail="Research API token is not configured.")
    if not hmac.compare_digest(x_ultron_research_token, settings.brain_research_api_token):
        raise HTTPException(status_code=403, detail="Invalid research API token.")


@router.post("/run", response_model=CognitiveExecution)
async def run_cognitive_task(body: BrainRunRequest):
    """Run the internal Executive-to-runtime path for diagnostics."""
    history = bound_history(body.history)
    assessment = await executive.assess(body.message, history=history)
    return await runtime.handle_conversation(
        user_input=body.message,
        history=history,
        assessment=assessment,
    )


@router.post("/coordinate", response_model=CoordinatorExecution)
async def coordinate_cognitive_task(body: BrainRunRequest):
    """Run the Step 8 lifecycle surface for engineering diagnostics."""
    history = bound_history(body.history)
    assessment = await executive.assess(body.message, history=history)
    return await coordinator.run(
        user_input=body.message,
        history=history,
        assessment=assessment,
    )


@router.get("/experiences/{correlation_id}", dependencies=[Depends(require_research_token)])
async def get_brain_experience(correlation_id: UUID):
    """Read one redacted audit record; it exposes no execution control."""
    experience = coordinator.learning_repository.experience(correlation_id)
    if experience is None:
        raise HTTPException(status_code=404, detail="Experience not found.")
    return experience


@router.get("/knowledge", dependencies=[Depends(require_research_token)])
async def retrieve_brain_knowledge(
    query: str = Query(min_length=1, max_length=400),
    capability: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=5, ge=1, le=10),
    status: list[KnowledgeStatus] = Query(default=[KnowledgeStatus.VERIFIED]),
):
    """Retrieve verified experience as data for inspection, never as instruction."""
    return {
        "items": coordinator.learning_repository.retrieve(
            query=query,
            scope=KnowledgeScope.SYSTEM_EXPERIENCE,
            capability=capability,
            limit=limit,
            statuses=tuple(status),
        )
    }


@router.get("/knowledge/{knowledge_id}/history", dependencies=[Depends(require_research_token)])
async def get_brain_knowledge_history(knowledge_id: UUID):
    """Expose revision provenance for an existing learned claim."""
    revisions = coordinator.learning_repository.knowledge_history(knowledge_id)
    if not revisions:
        raise HTTPException(status_code=404, detail="Knowledge item not found.")
    return {"knowledge_id": knowledge_id, "revisions": revisions}


@router.get("/capabilities/{capability}/reliability", dependencies=[Depends(require_research_token)])
async def get_capability_reliability(capability: str):
    metrics = coordinator.learning_repository.capability_reliability(capability)
    if metrics is None:
        raise HTTPException(status_code=404, detail="No capability experience found.")
    return {"capability": capability, "metrics": metrics}


@router.post("/research", response_model=ResearchTask, dependencies=[Depends(require_research_token)])
async def submit_research(body: ResearchRequest):
    """Queue one bounded public-web research job; no arbitrary HTTP endpoint exists."""
    return research_manager.submit(body.question, body.budget)


@router.post("/research/revalidate", dependencies=[Depends(require_research_token)])
async def queue_stale_revalidation(limit: int = Query(default=5, ge=1, le=10)):
    """Explicit scheduler trigger; it never starts an unbounded background loop."""
    return {"tasks": research_manager.queue_stale_revalidations(limit)}


@router.get("/research/{research_id}", dependencies=[Depends(require_research_token)])
async def get_research(research_id: UUID):
    task = research_manager.engine.store.task(research_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found.")
    return {"task": task, "result": research_manager.engine.store.result(research_id)}


@router.get("/research/sources/{domain}/reliability", dependencies=[Depends(require_research_token)])
async def get_source_reliability(domain: str):
    metrics = research_manager.engine.store.source_reliability(domain)
    if metrics is None:
        raise HTTPException(status_code=404, detail="No source history found.")
    return {"domain": domain, "metrics": metrics}


@router.post("/research/{research_id}/cancel", dependencies=[Depends(require_research_token)])
async def cancel_research(research_id: UUID):
    if not research_manager.cancel(research_id):
        raise HTTPException(status_code=409, detail="Research task is not running.")
    return {"research_id": research_id, "status": "cancelled"}


@router.get("/status")
async def get_brain_status():
    """Retrieve non-sensitive cognitive-layer configuration."""
    return {
        "status": "ready",
        "brain_model": settings.brain_model or "local:general",
        "brain_model_base_url": "local",
        "capabilities": {
            "web_research": "ready",
            "learning": "ready",
            "verification": "ready",
            "document_analysis": "unavailable",
        },
    }
