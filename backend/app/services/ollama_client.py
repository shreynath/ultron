from __future__ import annotations

import httpx

from app.config import settings


class OllamaError(Exception):
    pass


async def refine_transcript(text: str) -> str:
    if not settings.ollama_api_key:
        raise OllamaError("OLLAMA_API_KEY is not configured")

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Clean up this dictated transcript. Remove filler words, false starts, "
                    "and stutters. Preserve meaning and technical terms. Return only the cleaned text."
                ),
            },
            {"role": "user", "content": text},
        ],
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
        return message.get("content", "").strip() or text
