from __future__ import annotations

import json
from typing import Any, Protocol

from app.local_intelligence import get_local_orchestrator
from app.services.ollama_client import parse_structured_content


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
    ) -> Any:
        ...


class LocalStructuredClient:
    """Structured generation through the active local model only.

    ``model`` and ``host`` remain in the boundary signature for the frozen
    Executive/Planner interfaces.  They are deliberately not treated as an
    endpoint or a provider selection mechanism.
    """

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
    ) -> Any:
        del model, host, timeout
        schema_instruction = (
            "\n\nReturn exactly one JSON object matching this schema. Do not use markdown.\n"
            + json.dumps(json_schema, separators=(",", ":"))
        )
        content, _ = await get_local_orchestrator().generate(
            system_prompt=system_prompt + schema_instruction,
            user_message=user_message,
            history=history,
            temperature=0.0,
        )
        return parse_structured_content(content)


# Compatibility name retained for callers and focused legacy tests.  It is an
# alias (not an Ollama adapter), so the default cognitive path has no network
# model dependency.
OllamaStructuredClient = LocalStructuredClient
