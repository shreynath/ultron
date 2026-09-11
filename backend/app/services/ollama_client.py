from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import settings

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class OllamaError(Exception):
    pass


async def refine_transcript(text: str) -> str:
    return await chat_response(
        system_prompt=(
            "Clean up this dictated transcript. Remove filler words, false starts, "
            "and stutters. Preserve meaning and technical terms. Return only the cleaned text."
        ),
        user_message=text,
    )


def _chat_url(host: str) -> str:
    return f"{host.rstrip('/')}/api/chat"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


async def chat_response(
    *,
    user_message: str,
    system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    history: list[dict[str, str]] | None = None,
) -> str:
    if not settings.ollama_api_key:
        raise OllamaError("OLLAMA_API_KEY is not configured")

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            _chat_url(settings.ollama_host),
            headers=_auth_headers(settings.ollama_api_key),
            json=payload,
        )
        if response.status_code >= 400:
            raise OllamaError(response.text)
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip() or user_message


def parse_structured_content(content: Any) -> dict[str, Any]:
    """Parse model content as a JSON object. Does not classify from prose."""

    if isinstance(content, dict):
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
        if not fenced:
            raise OllamaError("Structured model output was not valid JSON") from None
        parsed = json.loads(fenced.group(1))

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
    """Request JSON matching json_schema via Ollama's native format parameter."""

    selected_model = model or settings.ollama_model
    selected_host = host or settings.ollama_host
    key = settings.ollama_api_key if api_key is None else api_key

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "format": json_schema,
        "options": {"temperature": 0},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _chat_url(selected_host),
                headers=_auth_headers(key or None),
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise OllamaError("Structured model call timed out") from exc
    except httpx.HTTPError as exc:
        raise OllamaError("Structured model call failed") from exc

    if response.status_code >= 400:
        raise OllamaError("Structured model call returned an error")

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OllamaError("Structured model HTTP body was not JSON") from exc

    message = data.get("message", {})
    return parse_structured_content(message.get("content", ""))
