from __future__ import annotations

import json
import re
from typing import Any

from app.local_intelligence import LocalModelError, get_local_orchestrator

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class OllamaError(Exception):
    """Compatibility error name for retired callers; no Ollama request occurs."""


async def refine_transcript(text: str) -> str:
    return await chat_response(
        system_prompt=(
            "Clean up this dictated transcript. Remove filler words, false starts, "
            "and stutters. Preserve meaning and technical terms. Return only the cleaned text."
        ),
        user_message=text,
    )


async def chat_response(
    *,
    user_message: str,
    system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    history: list[dict[str, str]] | None = None,
) -> str:
    """Retired import path backed by the active local runtime only."""
    try:
        answer, _ = await get_local_orchestrator().generate(
            system_prompt=system_prompt, user_message=user_message, history=history,
        )
    except LocalModelError as exc:
        raise OllamaError(f"LOCAL MODEL UNAVAILABLE: {exc}") from exc
    return answer


def parse_structured_content(content: Any) -> dict[str, Any]:
    """Parse a provider's JSON-object representation without changing meaning.

    Some compatible endpoints return the object directly while others wrap it
    in a ``content``/``response`` field or place JSON after a short preamble.
    This function only unwraps or extracts a JSON object; typing, enum values,
    required fields, and unknown fields are still rejected by Pydantic at the
    Executive boundary.
    """

    if isinstance(content, dict):
        for key in ("content", "response"):
            wrapped = content.get(key)
            if isinstance(wrapped, (str, dict)):
                return parse_structured_content(wrapped)
        return content
    if not isinstance(content, str):
        raise OllamaError("Structured model output was not JSON")

    text = content.strip()
    if not text:
        raise OllamaError("Structured model output was empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        fenced = _JSON_FENCE.search(text)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
            except json.JSONDecodeError as exc:
                raise OllamaError("Structured model output was not valid JSON") from exc
        else:
            # JSONDecoder.raw_decode accepts a provider's harmless prose
            # preamble while refusing trailing prose or a non-object value.
            brace = text.find("{")
            if brace < 0:
                raise OllamaError("Structured model output was not valid JSON") from None
            try:
                parsed, end = json.JSONDecoder().raw_decode(text[brace:])
            except json.JSONDecodeError as exc:
                raise OllamaError("Structured model output was not valid JSON") from exc
            if text[brace + end :].strip():
                raise OllamaError("Structured model output included non-JSON trailing content")

    if not isinstance(parsed, dict):
        raise OllamaError("Structured model output was not a JSON object")
    return parsed


async def chat_structured(
    *,
    user_message: str,
    system_prompt: str,
    json_schema: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    host: str | None = None,
    timeout: float = 30.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Retired import path for structured generation, retained without HTTP IO."""
    del model, host, timeout, api_key
    try:
        content, _ = await get_local_orchestrator().generate(
            system_prompt=system_prompt + "\nReturn JSON only for this schema:\n" + json.dumps(json_schema),
            user_message=user_message,
            history=history,
            temperature=0.0,
        )
    except LocalModelError as exc:
        raise OllamaError(f"LOCAL MODEL UNAVAILABLE: {exc}") from exc
    return parse_structured_content(content)
