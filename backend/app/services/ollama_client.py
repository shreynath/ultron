from __future__ import annotations

import httpx

from app.config import settings


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
            f"{settings.ollama_host.rstrip('/')}/api/chat",
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
            json=payload,
        )
        if response.status_code >= 400:
            raise OllamaError(response.text)
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip() or user_message
