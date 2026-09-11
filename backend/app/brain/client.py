from __future__ import annotations

from typing import Any, Protocol

from app.config import settings
from app.services.ollama_client import chat_structured


class StructuredModelClient(Protocol):
    """Model-agnostic structured generation used by the Executive."""

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None,
        json_schema: dict[str, Any],
        timeout: float,
        model: str,
        host: str,
    ) -> dict[str, Any]:
        ...


class OllamaStructuredClient:
    """Adapter over the shared Ollama HTTP helper."""

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None,
        json_schema: dict[str, Any],
        timeout: float,
        model: str,
        host: str,
    ) -> dict[str, Any]:
        return await chat_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            history=history,
            json_schema=json_schema,
            timeout=timeout,
            model=model,
            host=host,
            api_key=settings.ollama_api_key or None,
        )
