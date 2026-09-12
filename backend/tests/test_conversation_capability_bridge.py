from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.brain.actions import ActionResult, ActionStatus, CapabilityResult
from app.brain.conversation_bridge import ConversationCapabilityBridge
from app.brain.conversation_capabilities import (
    CapabilityDescriptor,
    ConversationAction,
    ConversationCapabilityRegistry,
)
from app.brain.executor import ActionExecutor
from app.brain.runtime import CognitiveRuntime, ExecutionStatus
from app.brain.schemas import (
    Complexity,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
)
from app.brain.weather import (
    WeatherCapability,
    WeatherCapabilityError,
    WeatherParameters,
)
from app.brain.working_memory import create_cognitive_context


def _assessment(*, tools: bool = True, task_type: TaskType = TaskType.RETRIEVAL) -> TaskAssessment:
    return TaskAssessment(
        task_type=task_type,
        complexity=Complexity.SIMPLE,
        requires_tools=tools,
        requires_memory=False,
        requires_planning=False,
        requires_verification=False,
        direct_response_allowed=not tools,
        reasoning_budget=ReasoningBudget.LOW,
        urgency=Urgency.NORMAL,
        user_goal="Get a current result.",
        rationale="A test requires one safe tool.",
    )


class EchoParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)


class EchoCapability:
    name = "echo"
    operations = ("lookup",)
    descriptor = CapabilityDescriptor(
        name=name,
        operations=operations,
        supported_task_types=(TaskType.RETRIEVAL,),
    )

    def __init__(self) -> None:
        self.calls = 0

    def supports(self, operation: str) -> bool:
        return operation == "lookup"

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return EchoParameters if self.supports(operation) else None

    def build_conversation_action(self, *, user_input: str, assessment: TaskAssessment) -> ConversationAction | None:
        if user_input == "echo please":
            return ConversationAction(operation="lookup", parameters={"value": "actual result"})
        return None

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        self.calls += 1
        assert isinstance(parameters, EchoParameters)
        return CapabilityResult(output={"value": parameters.value})

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is ActionStatus.COMPLETED:
            return f"Echo capability returned: {result.output['value']}."
        return None


class InventedOperationCapability(EchoCapability):
    name = "invented"
    descriptor = CapabilityDescriptor(
        name=name,
        operations=("lookup",),
        supported_task_types=(TaskType.RETRIEVAL,),
    )

    def build_conversation_action(self, *, user_input: str, assessment: TaskAssessment) -> ConversationAction | None:
        return ConversationAction(operation="not_registered", parameters={"value": "nope"})


class _IsolatedParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(pattern=r"^RESULT_[AB]$")


class _IsolatedCapability:
    """Test-only capability used to prove independent runtime turns."""

    name = "isolated"
    operations = ("read",)
    descriptor = CapabilityDescriptor(
        name=name,
        operations=operations,
        supported_task_types=(TaskType.RETRIEVAL,),
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def supports(self, operation: str) -> bool:
        return operation == "read"

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return _IsolatedParameters if self.supports(operation) else None

    def build_conversation_action(
        self, *, user_input: str, assessment: TaskAssessment
    ) -> ConversationAction | None:
        values = {
            "synthetic result a": "RESULT_A",
            "synthetic result b": "RESULT_B",
        }
        value = values.get(user_input)
        if assessment.requires_tools and value:
            return ConversationAction(operation="read", parameters={"value": value})
        return None

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        assert operation == "read"
        assert isinstance(parameters, _IsolatedParameters)
        self.calls.append(parameters.value)
        return CapabilityResult(output={"value": parameters.value})

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is ActionStatus.COMPLETED and isinstance(result.output, dict):
            return str(result.output["value"])
        return None


def _bridge(*capabilities: EchoCapability) -> ConversationCapabilityBridge:
    registry = ConversationCapabilityRegistry(tuple(capabilities))
    return ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(
            registry=registry.executor_registry,
            allowed_capabilities=registry.names(),
            timeout=1,
        ),
    )


def test_selection_requires_a_real_registered_capability() -> None:
    bridge = _bridge(EchoCapability())
    assert bridge.select(user_input="echo please", assessment=_assessment(tools=False)) is None
    assert bridge.select(user_input="unknown request", assessment=_assessment()) is None


def test_default_registry_connects_external_retrieval_to_web_research() -> None:
    bridge = ConversationCapabilityBridge()
    selection = bridge.select(
        user_input="Who is the current CEO of NVIDIA?",
        assessment=_assessment(task_type=TaskType.RETRIEVAL),
    )

    assert selection is not None
    assert selection.descriptor.name == "web_research"
    assert selection.action.operation == "research"


def test_default_registry_returns_corpus_clarification_through_a_real_capability() -> None:
    bridge = ConversationCapabilityBridge()
    selection = bridge.select(
        user_input="what is the 500th word in the bible",
        assessment=_assessment(task_type=TaskType.RETRIEVAL),
    )

    assert selection is not None
    assert selection.descriptor.name == "web_research"
    assert selection.action.operation == "clarify_corpus"


@pytest.mark.anyio
async def test_corpus_ambiguity_executes_authorized_capability_not_missing_capability_fallback() -> None:
    runtime = CognitiveRuntime(response_generator=AsyncMock(side_effect=RuntimeError("model unavailable")))

    execution = await runtime.handle_conversation(
        user_input="what is the 500th word in the bible",
        history=[],
        assessment=_assessment(task_type=TaskType.RETRIEVAL),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_name == "web_research"
    assert execution.capability_operation == "clarify_corpus"
    assert "translation" in execution.answer.lower()
    assert "matching retrieval capability" not in execution.answer


def test_weather_selector_preserves_an_explicit_spoken_location_for_geocoding() -> None:
    bridge = _weather_bridge()

    selection = bridge.select(
        user_input="what's the weather like in college station texas right now",
        assessment=_assessment(),
    )

    assert selection is not None
    assert selection.action.operation == "current_weather"
    assert selection.action.parameters == {"location": "college station, texas"}


def test_selector_rejects_an_invented_operation() -> None:
    bridge = _bridge(InventedOperationCapability())
    assert bridge.select(user_input="anything", assessment=_assessment()) is None


@pytest.mark.anyio
async def test_bridge_executes_exactly_one_generic_action_and_grounds_the_result() -> None:
    capability = EchoCapability()
    bridge = _bridge(capability)
    context = create_cognitive_context(
        user_input="echo please",
        history=[],
        assessment=_assessment(),
    )

    outcome = await bridge.execute(context=context)

    assert outcome is not None
    assert capability.calls == 1
    assert outcome.action_result.status is ActionStatus.COMPLETED
    assert len(context.working_memory.action_executions) == 1
    assert len(context.working_memory.observations) == 1
    assert bridge.response_for(outcome) == "Echo capability returned: actual result."


def _weather_transport(*, fail: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ReadTimeout("provider timeout", request=request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.path == "/v1/search"
            return httpx.Response(200, json={"results": [{
                "name": "College Station", "admin1": "Texas", "country": "United States",
                "latitude": 30.62798, "longitude": -96.33441,
            }]})
        assert request.url.host == "api.open-meteo.com"
        assert request.url.path == "/v1/forecast"
        return httpx.Response(200, json={"timezone": "America/Chicago", "current": {
            "time": "2026-09-10T12:00", "temperature_2m": 84.0,
            "apparent_temperature": 87.0, "precipitation": 0.0,
            "weather_code": 2, "wind_speed_10m": 9.0,
        }})

    return httpx.MockTransport(handler)


def _weather_bridge(*, fail: bool = False) -> ConversationCapabilityBridge:
    weather = WeatherCapability(transport=_weather_transport(fail=fail), timeout=0.2)
    registry = ConversationCapabilityRegistry((weather,))
    return ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(registry=registry.executor_registry, allowed_capabilities=registry.names(), timeout=0.5),
    )


@pytest.mark.anyio
async def test_weather_conversation_uses_provider_output_even_when_llm_is_unavailable() -> None:
    runtime = CognitiveRuntime(
        response_generator=AsyncMock(side_effect=RuntimeError("Ollama unavailable")),
        conversation_bridge=_weather_bridge(),
    )

    execution = await runtime.handle_conversation(
        user_input="What is the current weather in College Station, Texas?",
        history=[],
        assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_name == "weather"
    assert execution.action_status == "completed"
    assert execution.required_capabilities == ()
    assert "84°F" in execution.answer
    assert "College Station, Texas, United States" in execution.answer
    assert "Ollama" not in execution.answer
    assert execution.response_generation_status == "grounded_capability"


@pytest.mark.anyio
async def test_weather_result_is_rendered_only_from_the_authoritative_action_result() -> None:
    generator = AsyncMock(return_value="It is 84°F and partly cloudy in College Station, Texas.")
    runtime = CognitiveRuntime(response_generator=generator, conversation_bridge=_weather_bridge())

    execution = await runtime.handle_conversation(
        user_input="What's the weather like in College Station, Texas right now?",
        history=[],
        assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert "84°F" in execution.answer
    assert execution.response_generation_status == "grounded_capability"
    generator.assert_not_awaited()


@pytest.mark.anyio
async def test_weather_follow_up_executes_one_fresh_lookup_from_explicit_history() -> None:
    capability = WeatherCapability(transport=_weather_transport(), timeout=0.2)
    registry = ConversationCapabilityRegistry((capability,))
    runtime = CognitiveRuntime(
        response_generator=AsyncMock(return_value="The fresh temperature is 84°F."),
        conversation_bridge=ConversationCapabilityBridge(
            registry=registry,
            executor=ActionExecutor(
                registry=registry.executor_registry,
                allowed_capabilities=registry.names(),
                timeout=0.5,
            ),
        ),
    )

    execution = await runtime.handle_conversation(
        user_input="what about the temperature?",
        history=[{"role": "user", "content": "What's the weather like in College Station, Texas?"}],
        assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_name == "weather"
    assert execution.capability_operation == "current_weather"


@pytest.mark.anyio
async def test_weather_history_cannot_answer_an_unrelated_bible_request() -> None:
    """Regression: an unmatched request must not replay an earlier action."""

    weather = WeatherCapability(transport=_weather_transport(), timeout=0.2)
    weather.execute = AsyncMock(return_value=CapabilityResult(output={  # type: ignore[method-assign]
        "location": "College Station, Texas, United States",
        "temperature_f": 83.0,
        "apparent_temperature_f": 90.9,
        "precipitation_inches": 0.0,
        "condition": "Clear sky",
        "wind_mph": 5.5,
        "observed_at": "2026-09-10T12:00",
        "timezone": "America/Chicago",
        "provider": "Open-Meteo",
        "retrieved_at": "2026-09-10T12:01:00+00:00",
    }))
    registry = ConversationCapabilityRegistry((weather,))
    runtime = CognitiveRuntime(
        response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")),
        conversation_bridge=ConversationCapabilityBridge(
            registry=registry,
            executor=ActionExecutor(
                registry=registry.executor_registry,
                allowed_capabilities=registry.names(),
                timeout=0.5,
            ),
        ),
    )

    weather_request = "what's the weather in college station, texas"
    weather_turn = await runtime.handle_conversation(
        user_input=weather_request, history=[], assessment=_assessment(),
    )
    bible_turn = await runtime.handle_conversation(
        user_input="what's the 500th word of the bible",
        history=[
            {"role": "user", "content": weather_request},
            {"role": "assistant", "content": weather_turn.answer},
        ],
        assessment=_assessment(),
    )

    assert weather_turn.status is ExecutionStatus.COMPLETED
    assert "83°F" in weather_turn.answer
    assert bible_turn.status is ExecutionStatus.REQUIRES_FUTURE_CAPABILITY
    assert bible_turn.action_status is None
    assert bible_turn.verification is None
    assert "83°F" not in bible_turn.answer
    assert "College Station" not in bible_turn.answer
    assert "weather" not in bible_turn.answer.lower()
    weather.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_runtime_turns_never_inherit_a_prior_synthetic_result() -> None:
    capability = _IsolatedCapability()
    registry = ConversationCapabilityRegistry((capability,))
    runtime = CognitiveRuntime(
        response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")),
        conversation_bridge=ConversationCapabilityBridge(
            registry=registry,
            executor=ActionExecutor(
                registry=registry.executor_registry,
                allowed_capabilities=registry.names(),
                timeout=0.5,
            ),
        ),
    )

    first = await runtime.handle_conversation(
        user_input="synthetic result a", history=[], assessment=_assessment(),
    )
    second = await runtime.handle_conversation(
        user_input="synthetic result b",
        history=[{"role": "user", "content": "synthetic result a"}],
        assessment=_assessment(),
    )
    missing = await runtime.handle_conversation(
        user_input="a request with no capability",
        history=[{"role": "user", "content": "synthetic result a"}],
        assessment=_assessment(),
    )

    assert first.status is ExecutionStatus.COMPLETED
    assert first.answer == "RESULT_A"
    assert second.status is ExecutionStatus.COMPLETED
    assert second.answer == "RESULT_B"
    assert missing.status is ExecutionStatus.REQUIRES_FUTURE_CAPABILITY
    assert missing.action_status is None
    assert "RESULT_A" not in missing.answer
    assert capability.calls == ["RESULT_A", "RESULT_B"]


@pytest.mark.anyio
async def test_weather_provider_failure_never_fabricates_weather() -> None:
    runtime = CognitiveRuntime(conversation_bridge=_weather_bridge(fail=True))

    execution = await runtime.handle_conversation(
        user_input="What is the current weather in College Station, Texas?",
        history=[],
        assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.FAILED
    assert execution.action_status == "timed_out"
    assert "couldn't complete" in execution.answer
    assert "84°F" not in execution.answer


@pytest.mark.anyio
async def test_weather_capability_validates_a_real_provider_payload() -> None:
    capability = WeatherCapability(transport=_weather_transport(), timeout=0.2)

    result = await capability.execute(
        "current_weather",
        WeatherParameters(location="College Station, Texas"),
    )

    assert result.output is not None
    assert result.output["location"] == "College Station, Texas, United States"
    assert result.output["temperature_f"] == 84.0
    assert result.output["condition"] == "Partly cloudy"


@pytest.mark.anyio
async def test_weather_timeout_has_a_distinct_action_status() -> None:
    runtime = CognitiveRuntime(conversation_bridge=_weather_bridge(fail=True))

    execution = await runtime.handle_conversation(
        user_input="What is the current weather in College Station, Texas?",
        history=[],
        assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.FAILED
    assert execution.action_status == "timed_out"
    assert "timed out" in execution.answer


@pytest.mark.anyio
async def test_weather_invalid_provider_response_and_missing_location_are_rejected() -> None:
    invalid_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": []})
        if request.url.host == "geocoding-api.open-meteo.com"
        else httpx.Response(200, json={"timezone": "America/Chicago", "current": {"temperature_2m": "bad"}})
    )
    capability = WeatherCapability(transport=invalid_transport, timeout=0.2)

    with pytest.raises(WeatherCapabilityError):
        await capability.execute("current_weather", WeatherParameters(location="No Such Place"))

    invalid_forecast_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": [{
            "name": "College Station", "latitude": 30.6, "longitude": -96.3,
        }]})
        if request.url.host == "geocoding-api.open-meteo.com"
        else httpx.Response(200, json={"timezone": "America/Chicago", "current": {"temperature_2m": "bad"}})
    )
    with pytest.raises(ValidationError):
        await WeatherCapability(transport=invalid_forecast_transport, timeout=0.2).execute(
            "current_weather", WeatherParameters(location="College Station, Texas")
        )
