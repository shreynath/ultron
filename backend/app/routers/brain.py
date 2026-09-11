from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.brain import CognitiveExecution, CognitiveRuntime, Executive, bound_history
from app.config import settings

router = APIRouter(prefix="/api/brain", tags=["brain"])
executive = Executive()
runtime = CognitiveRuntime()


class BrainRunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list)


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


@router.get("/status")
async def get_brain_status():
    """Retrieve non-sensitive cognitive-layer configuration."""
    return {
        "status": "ready",
        "brain_model": settings.brain_model or settings.ollama_model,
        "brain_model_base_url": settings.brain_model_base_url or settings.ollama_host,
    }
