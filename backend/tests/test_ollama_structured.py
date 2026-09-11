from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.services.ollama_client import OllamaError, chat_structured, parse_structured_content


def test_parse_rejects_non_object_json():
    with pytest.raises(OllamaError):
        parse_structured_content("[1, 2]")


def test_parse_rejects_empty():
    with pytest.raises(OllamaError):
        parse_structured_content("   ")


def test_chat_structured_timeout():
    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("timeout")

    async def _run():
        with patch("app.services.ollama_client.httpx.AsyncClient", _Client):
            await chat_structured(
                user_message="hi",
                system_prompt="sys",
                json_schema={"type": "object"},
                timeout=0.1,
                model="test-model",
                host="http://127.0.0.1:9",
                api_key=None,
            )

    with pytest.raises(OllamaError, match="timed out"):
        asyncio.run(_run())


def test_chat_structured_http_error_is_generic():
    class _Response:
        status_code = 500
        text = "secret-internal"

        def json(self):
            return {}

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _Response()

    async def _run():
        with patch("app.services.ollama_client.httpx.AsyncClient", _Client):
            await chat_structured(
                user_message="hi",
                system_prompt="sys",
                json_schema={"type": "object"},
                model="test-model",
                host="http://127.0.0.1:9",
                api_key=None,
            )

    with pytest.raises(OllamaError, match="returned an error") as exc_info:
        asyncio.run(_run())
    assert "secret-internal" not in str(exc_info.value)
