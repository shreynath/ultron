from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.brain.actions import CapabilityResult
from app.brain.executive import Executive
from app.brain.runtime import (
    CognitiveExecution,
    CognitiveRuntime,
    ExecutionRoute,
    ExecutionStatus,
)
from app.brain.schemas import TaskType, fallback_assessment
from app.main import app
from app.services.ollama_client import OllamaError

client = TestClient(app)


class _WeatherExecutiveClient:
    """A structured Executive adapter that intentionally misclassifies weather.

    Executive normalization must still prevent the request from reaching the
    direct-response path, proving the endpoint exercises production triage.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "task_type": "question",
            "complexity": "simple",
            "requires_tools": False,
            "requires_memory": False,
            "requires_planning": False,
            "requires_verification": False,
            "direct_response_allowed": True,
            "reasoning_budget": "low",
            "urgency": "normal",
            "user_goal": "Answer the weather question.",
            "rationale": "A direct answer is enough.",
        }


class _DirectExecutiveClient:
    async def generate_structured(self, **_: Any) -> dict[str, Any]:
        return {
            "task_type": "conversation",
            "complexity": "trivial",
            "requires_tools": False,
            "requires_memory": False,
            "requires_planning": False,
            "requires_verification": False,
            "direct_response_allowed": True,
            "reasoning_budget": "minimal",
            "urgency": "normal",
            "user_goal": "Reply to a greeting.",
            "rationale": "No external information is needed.",
        }


def test_health_endpoint():
    with patch("app.main.voicebox.health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = {"status": "ok"}
        response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "voicebox" in data
    assert "tools" in data
    assert "ollama_configured" in data


def test_list_voices_unavailable():
    with patch("app.routers.voices.client.list_profiles", new_callable=AsyncMock) as mock_list:
        mock_list.side_effect = ConnectionError("Voicebox down")
        response = client.get("/api/voices")
    assert response.status_code == 502


def test_refine_without_ollama_key():
    with patch("app.config.settings.ollama_api_key", ""):
        response = client.post("/api/transcribe/refine", json={"text": "hello um world"})
    assert response.status_code == 400


def test_chat_assesses_routes_then_uses_existing_reply_path():
    assessment = fallback_assessment()
    assessment.task_type = TaskType.CONVERSATION
    with (
        patch("app.routers.chat.executive.assess", new_callable=AsyncMock) as mock_assess,
        patch("app.routers.chat.runtime.handle_conversation", new_callable=AsyncMock) as mock_handle,
        patch("app.routers.chat.deliver_speech", new_callable=AsyncMock) as mock_tts,
    ):
        mock_assess.return_value = assessment
        mock_handle.return_value = CognitiveExecution(
            answer="Hello there.",
            route=ExecutionRoute.DIRECT_RESPONSE,
            status=ExecutionStatus.COMPLETED,
            assessment=assessment,
        )
        mock_tts.return_value = b"RIFF"
        response = client.post(
            "/api/chat",
            json={"message": "Hey Ultron, how are you?", "profile_id": "test-voice"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["user_message"] == "Hey Ultron, how are you?"
    assert data["assistant_message"] == "Hello there."
    assert "audio_base64" in data
    assert data["metadata"] == {
        "task_type": "conversation",
        "route": "direct_response",
        "status": "completed",
    }
    mock_assess.assert_awaited_once_with("Hey Ultron, how are you?", history=[])
    mock_handle.assert_awaited_once_with(
        user_input="Hey Ultron, how are you?",
        history=[],
        assessment=assessment,
        system_prompt="You are a helpful assistant. Keep replies concise and conversational, 1-3 sentences.",
    )
    mock_tts.assert_awaited_once()


def test_chat_keeps_text_answer_when_voice_delivery_fails():
    assessment = fallback_assessment()
    with (
        patch("app.routers.chat.executive.assess", new_callable=AsyncMock, return_value=assessment),
        patch("app.routers.chat.runtime.handle_conversation", new_callable=AsyncMock) as mock_handle,
        patch("app.routers.chat.deliver_speech", new_callable=AsyncMock) as mock_tts,
    ):
        mock_handle.return_value = CognitiveExecution(
            answer="The text answer is still available.",
            route=ExecutionRoute.DIRECT_RESPONSE,
            status=ExecutionStatus.COMPLETED,
            assessment=assessment,
        )
        mock_tts.side_effect = RuntimeError("voicebox detail must not leak")
        response = client.post("/api/chat", json={"message": "Hello", "profile_id": "voice"})
    assert response.status_code == 200
    assert response.json()["assistant_message"] == "The text answer is still available."
    assert response.json()["audio_base64"] is None
    assert response.json()["voice_error"] == "Audio is unavailable for this response."


def test_chat_weather_uses_the_real_executive_runtime_bridge_and_executor_path():
    executive_client = _WeatherExecutiveClient()
    runtime = CognitiveRuntime(response_generator=AsyncMock(side_effect=OllamaError("unavailable")))
    provider_result = CapabilityResult(output={
        "location": "College Station, Texas, United States",
        "temperature_f": 89.0,
        "apparent_temperature_f": 91.0,
        "precipitation_inches": 0.0,
        "condition": "Partly cloudy",
        "wind_mph": 8.0,
        "observed_at": "2026-09-10T12:00",
        "timezone": "America/Chicago",
        "provider": "Open-Meteo",
        "retrieved_at": "2026-09-10T12:01:00+00:00",
    })

    with (
        patch("app.routers.chat.executive", Executive(executive_client)),
        patch("app.routers.chat.runtime", runtime),
        patch(
            "app.brain.weather.WeatherCapability.execute",
            new_callable=AsyncMock,
            return_value=provider_result,
        ) as weather_execute,
    ):
        response = client.post(
            "/api/chat",
            json={"message": "what's the weather in college station, texas", "voice_enabled": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert executive_client.calls == 1
    assert weather_execute.call_count == 1
    assert data["metadata"] == {
        "task_type": "retrieval",
        "route": "cognitive_execution_required",
        "status": "completed",
    }
    assert "89°F" in data["assistant_message"]
    assert "College Station, Texas, United States" in data["assistant_message"]
    assert "live weather data" not in data["assistant_message"]


def test_chat_direct_response_does_not_construct_or_execute_a_capability_action():
    generator = AsyncMock(return_value="Hello from the direct path.")
    local_runtime = CognitiveRuntime(response_generator=generator)
    with (
        patch("app.routers.chat.executive", Executive(_DirectExecutiveClient())),
        patch("app.routers.chat.runtime", local_runtime),
        patch.object(local_runtime._conversation_bridge._executor, "execute", new_callable=AsyncMock) as execute,
    ):
        response = client.post("/api/chat", json={"message": "Hello Ultron.", "voice_enabled": False})

    assert response.status_code == 200
    assert response.json()["assistant_message"] == "Hello from the direct path."
    generator.assert_awaited_once()
    execute.assert_not_awaited()


def test_unsupported_retrieval_request_keeps_the_runtime_limitation_truthful():
    assessment = fallback_assessment()
    assessment.task_type = TaskType.RETRIEVAL
    assessment.requires_tools = True
    assessment.requires_verification = True
    assessment.direct_response_allowed = False
    limitation = "I can identify that this request needs live retrieval, but Ultron's current runtime does not have a matching retrieval capability connected yet."
    with (
        patch("app.routers.chat.executive.assess", new_callable=AsyncMock, return_value=assessment) as assess,
        patch("app.routers.chat.runtime.handle_conversation", new_callable=AsyncMock) as handle,
    ):
        handle.return_value = CognitiveExecution(
            answer=limitation,
            route=ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
            status=ExecutionStatus.REQUIRES_FUTURE_CAPABILITY,
            assessment=assessment,
            required_capabilities=("tools", "verification"),
        )
        response = client.post(
            "/api/chat",
            json={"message": "What is the current stock price of ULTRON?", "voice_enabled": False},
        )
    assert response.status_code == 200
    assert response.json()["assistant_message"] == limitation
    assert "stock price" not in response.json()["assistant_message"]
    assert response.json()["metadata"]["status"] == "requires_future_capability"
    assess.assert_awaited_once()
    handle.assert_awaited_once()
