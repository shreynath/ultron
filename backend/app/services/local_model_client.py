"""Conversation adapter for the active local model orchestrator."""
from __future__ import annotations

from app.local_intelligence import ModelCapability, get_local_orchestrator


async def chat_response(
    *,
    user_message: str,
    system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    history: list[dict[str, str]] | None = None,
) -> str:
    """Generate locally or surface a local failure; never use a cloud fallback."""
    answer, _ = await get_local_orchestrator().generate(
        system_prompt=system_prompt,
        user_message=user_message,
        history=history,
        capability=ModelCapability.GENERAL,
    )
    return answer
