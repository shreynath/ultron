from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.local_intelligence import LocalModelUnavailable
from app.services.ollama_client import OllamaError, chat_structured, parse_structured_content


def test_parse_rejects_non_object_json():
    with pytest.raises(OllamaError):
        parse_structured_content("[1, 2]")


def test_parse_rejects_empty():
    with pytest.raises(OllamaError):
        parse_structured_content("   ")


def test_retired_structured_entrypoint_reports_local_unavailable():
    async def run() -> None:
        with patch("app.services.ollama_client.get_local_orchestrator") as orchestrator:
            orchestrator.return_value.generate = AsyncMock(side_effect=LocalModelUnavailable("weights absent"))
            await chat_structured(
                user_message="hi", system_prompt="sys", json_schema={"type": "object"},
                timeout=0.1, model="ignored", host="http://127.0.0.1:9", api_key=None,
            )
    with pytest.raises(OllamaError, match="LOCAL MODEL UNAVAILABLE"):
        asyncio.run(run())
