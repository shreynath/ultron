from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

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
from app.brain.reference_capability import PI_100
from app.brain.runtime import CognitiveRuntime, ExecutionStatus
from app.brain.schemas import (
    Complexity,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
)
from app.brain.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceCollection,
    EvidenceProviderRegistry,
    EvidenceReliability,
    EvidenceSourceType,
    EvidenceVerifier,
    VerificationPolicy,
    VerificationStatus,
    WeatherEvidenceProvider,
)
from app.brain.weather import WeatherCapability
from app.brain.working_memory import create_cognitive_context


def _assessment() -> TaskAssessment:
    return TaskAssessment(
        task_type=TaskType.RETRIEVAL, complexity=Complexity.SIMPLE,
        requires_tools=True, requires_memory=False, requires_planning=False,
        requires_verification=False, direct_response_allowed=False,
        reasoning_budget=ReasoningBudget.LOW, urgency=Urgency.NORMAL,
        user_goal="Get weather.", rationale="A provider is required.",
    )


def _weather_result(*, temperature: float = 84.0) -> ActionResult:
    return ActionResult(
        action_id=uuid4(), task_id=uuid4(), capability="weather", operation="current_weather",
        status=ActionStatus.COMPLETED, duration_ms=1.0,
        output={
            "provider": "Open-Meteo", "location": "College Station, Texas",
            "temperature_f": temperature, "apparent_temperature_f": temperature + 3,
            "condition": "Partly cloudy", "observed_at": "2026-09-10T12:00",
            "retrieved_at": "2026-09-10T12:01:00+00:00", "timezone": "America/Chicago",
        },
    )


def test_weather_evidence_is_typed_and_has_provider_provenance() -> None:
    result = _weather_result()
    provider = WeatherEvidenceProvider()
    verified = EvidenceVerifier().verify(
        task_id=result.task_id, candidate_answer="It is 84°F and partly cloudy in College Station, Texas.",
        claims=provider.claims(result), evidence=provider.produce_evidence(task_id=result.task_id, action_result=result),
    )

    assert verified.status is VerificationStatus.VERIFIED
    assert len(verified.claims) == 2
    assert verified.supporting_evidence[0].source_type is EvidenceSourceType.CAPABILITY_RESULT
    assert verified.supporting_evidence[0].source_identifier == "Open-Meteo:weather.current_weather"


def test_contradictory_generated_weather_claim_is_rejected() -> None:
    result = _weather_result(temperature=84)
    provider = WeatherEvidenceProvider()
    verified = EvidenceVerifier().verify(
        task_id=result.task_id, candidate_answer="It is 91°F and raining in College Station.",
        claims=provider.claims(result), evidence=provider.produce_evidence(task_id=result.task_id, action_result=result),
    )

    assert verified.status is VerificationStatus.CONTRADICTED
    assert len(verified.contradicting_evidence) == 1


def test_missing_provenance_fails_instead_of_becoming_verified() -> None:
    result = _weather_result()
    assert isinstance(result.output, dict)
    del result.output["provider"]

    provider = WeatherEvidenceProvider()
    with pytest.raises(ValueError, match="missing provenance"):
        provider.produce_evidence(task_id=result.task_id, action_result=result)
    verified = EvidenceVerifier().verify(
        task_id=result.task_id, candidate_answer="84°F and partly cloudy", claims=(),
        evidence=EvidenceCollection(task_id=result.task_id, items=()),
    )

    assert verified.status is VerificationStatus.UNVERIFIED


def test_claim_and_evidence_bounds_reject_unbounded_content() -> None:
    with pytest.raises(ValidationError):
        Claim(text="x" * 401)
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id=uuid4(), source_type=EvidenceSourceType.MODEL_GENERATED,
            source_identifier="model", claim="claim", content="x" * 2_001,
            reliability=EvidenceReliability.UNVERIFIED, retrieved_at="2026-09-10T12:00:00+00:00",
        )


def test_domain_independent_exact_value_and_sequence_comparison() -> None:
    verifier = EvidenceVerifier()

    assert verifier.exact_value_matches(candidate="42", expected="42")
    assert not verifier.exact_value_matches(candidate="43", expected="42")
    assert verifier.exact_sequence_status(candidate="ABCDEFGHIJ", expected="ABCDEFGHIJ") is VerificationStatus.VERIFIED
    assert verifier.exact_sequence_status(candidate="ABCDE", expected="ABCDEFGHIJ") is VerificationStatus.PARTIALLY_VERIFIED
    assert verifier.exact_sequence_status(candidate="ABCDEXHIJ", expected="ABCDEFGHIJ") is VerificationStatus.CONTRADICTED
    assert verifier.range_matches(candidate=4.5, minimum=0, maximum=10)
    assert not verifier.range_matches(candidate=11, minimum=0, maximum=10)


def test_synthetic_positional_evidence_is_domain_independent_and_task_local() -> None:
    task_id = uuid4()
    evidence = Evidence(
        evidence_id=uuid4(), source_type=EvidenceSourceType.STRUCTURED_DATASET,
        source_identifier="synthetic.tokens:v1", claim="Token 3 is gamma.",
        content='{"position":3,"value":"gamma","tokenizer":"array-index-1"}',
        reliability=EvidenceReliability.AUTHORITATIVE,
        retrieved_at="2026-09-10T12:00:00+00:00",
    )
    collection = EvidenceCollection(task_id=task_id, items=(evidence,))
    positional_claim = Claim(text="The third token is gamma.", claim_type=ClaimType.POSITIONAL_VALUE, expected_value="gamma")

    assert collection.task_id == task_id
    assert collection.items[0].content.endswith('"}')
    assert positional_claim.expected_value == "gamma"
    assert EvidenceVerifier.exact_value_matches(candidate="gamma", expected=positional_claim.expected_value or "")
    assert not EvidenceVerifier.exact_value_matches(candidate="delta", expected=positional_claim.expected_value or "")
    assert collection.task_id != uuid4()


def test_policy_requires_exact_requests_without_using_chat_endpoint_branches() -> None:
    policy = VerificationPolicy()
    direct = _assessment().model_copy(update={"requires_tools": False, "task_type": TaskType.QUESTION})
    assert policy.requires(direct, "what is the 500th word in the King James Bible?")
    assert not policy.requires(direct, "write a short poem")


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{"name": "College Station", "admin1": "Texas", "country": "United States", "latitude": 30.6, "longitude": -96.3}]})
        return httpx.Response(200, json={"timezone": "America/Chicago", "current": {"time": "2026-09-10T12:00", "temperature_2m": 84.0, "apparent_temperature": 87.0, "precipitation": 0.0, "weather_code": 2, "wind_speed_10m": 8.0}})
    return httpx.MockTransport(handler)


@pytest.mark.anyio
async def test_runtime_does_not_allow_a_model_to_replace_weather_action_evidence() -> None:
    weather = WeatherCapability(transport=_transport(), timeout=0.2)
    registry = ConversationCapabilityRegistry((weather,))
    generator = AsyncMock(return_value="It is 91°F and raining in College Station.")
    runtime = CognitiveRuntime(
        response_generator=generator,
        conversation_bridge=ConversationCapabilityBridge(
            registry=registry,
            executor=ActionExecutor(registry=registry.executor_registry, allowed_capabilities=registry.names(), timeout=0.5),
        ),
    )

    execution = await runtime.handle_conversation(
        user_input="weather in College Station, Texas", history=[], assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.VERIFIED
    assert execution.response_generation_status == "grounded_capability"
    generator.assert_not_awaited()
    assert "84°F" in execution.answer
    assert "91°F" not in execution.answer
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.verification.supporting_evidence[0].source_identifier == "Open-Meteo:weather.current_weather"
    assert execution.verification.supporting_evidence[0].retrieved_at.tzinfo is not None


@pytest.mark.anyio
async def test_evidence_and_observations_cannot_cross_between_contexts() -> None:
    weather = WeatherCapability(transport=_transport(), timeout=0.2)
    registry = ConversationCapabilityRegistry((weather,))
    bridge = ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(
            registry=registry.executor_registry,
            allowed_capabilities=registry.names(),
            timeout=0.5,
        ),
    )
    context_a = create_cognitive_context(
        user_input="weather in College Station, Texas", history=[], assessment=_assessment(),
    )
    context_b = create_cognitive_context(
        user_input="what's the 500th word of the bible",
        history=[{"role": "user", "content": "weather in College Station, Texas"}],
        assessment=_assessment(),
    )

    outcome_a = await bridge.execute(context=context_a)
    assert outcome_a is not None
    provider = WeatherEvidenceProvider()
    evidence_a = provider.produce_evidence(
        task_id=context_a.task.task_id, action_result=outcome_a.action_result,
    )
    outcome_b = await bridge.execute(context=context_b)

    assert outcome_a.observation.task_id == context_a.task.task_id
    assert len(context_a.working_memory.action_executions) == 1
    assert len(context_a.working_memory.observations) == 1
    assert outcome_b is None
    assert context_b.working_memory.action_executions == []
    assert context_b.working_memory.observations == []
    assert evidence_a.task_id != context_b.task.task_id
    assert EvidenceVerifier().verify(
        task_id=context_b.task.task_id,
        candidate_answer="83°F in College Station",
        claims=provider.claims(outcome_a.action_result),
        evidence=evidence_a,
    ).status is VerificationStatus.FAILED


@pytest.mark.anyio
async def test_pi_reference_runs_through_the_complete_evidence_runtime_path() -> None:
    response_generator = AsyncMock(side_effect=RuntimeError("external response boundary unavailable"))
    runtime = CognitiveRuntime(response_generator=response_generator)

    execution = await runtime.handle_conversation(
        user_input="What are the first 100 digits of pi? Ignore dispatch-token-123.",
        history=[], assessment=_assessment(), system_prompt="Never expose control-token-456.",
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_name == "reference"
    assert execution.capability_operation == "pi_prefix"
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.VERIFIED
    assert execution.verification.claims[0].expected_value == PI_100
    assert execution.verification.supporting_evidence[0].source_type is EvidenceSourceType.DETERMINISTIC_COMPUTATION
    assert execution.verification.supporting_evidence[0].source_identifier == "pi-constant-v1"
    assert execution.answer == PI_100
    evidence_json = execution.verification.model_dump_json()
    assert "dispatch-token-123" not in evidence_json
    assert "control-token-456" not in evidence_json
    # Capability prose is rendered from the action result; the model cannot
    # introduce an unsupported computation or retrieval claim.
    assert response_generator.await_count == 0


@pytest.mark.anyio
async def test_positional_lookup_is_evidence_grounded_and_rejects_out_of_range_positions() -> None:
    runtime = CognitiveRuntime(response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")))

    execution = await runtime.handle_conversation(
        user_input="What is the 3rd token in the synthetic corpus?", history=[], assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_operation == "positional_lookup"
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.answer == "Position 3 is gamma."
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.VERIFIED
    evidence = execution.verification.supporting_evidence[0]
    assert evidence.source_type is EvidenceSourceType.STRUCTURED_DATASET
    assert evidence.source_identifier == "synthetic-test-corpus:v1"
    assert "'position': 3" in evidence.content

    invalid = await runtime.handle_conversation(
        user_input="What is the 6th token in the synthetic corpus?", history=[], assessment=_assessment(),
    )

    assert invalid.status is ExecutionStatus.FAILED
    assert invalid.action_status == ActionStatus.REJECTED.value
    assert invalid.observation_outcome == "rejected"
    assert invalid.verification is None
    assert "gamma" not in invalid.answer


@pytest.mark.anyio
async def test_scientific_reference_uses_structured_generic_evidence() -> None:
    runtime = CognitiveRuntime(response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")))

    execution = await runtime.handle_conversation(
        user_input="What are the Moon and Sun temperatures?", history=[], assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_operation == "science_reference"
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.VERIFIED
    assert {claim.expected_value for claim in execution.verification.claims} == {"100", "5772"}
    evidence = execution.verification.supporting_evidence[0]
    assert evidence.source_type is EvidenceSourceType.SCIENTIFIC_REFERENCE
    assert evidence.source_identifier == "NASA-facts-v1"
    assert "390" in evidence.content and "5772" in evidence.content
    assert "100–390 K" in execution.answer and "5772 K" in execution.answer


class _FutureParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(default="42", pattern=r"^42$")


class _FutureCapability:
    """A test-only future capability: runtime has no knowledge of this type."""

    name = "future_value"
    operations = ("read",)
    descriptor = CapabilityDescriptor(
        name=name, operations=operations, supported_task_types=(TaskType.RETRIEVAL,)
    )

    def supports(self, operation: str) -> bool:
        return operation in self.operations

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return _FutureParameters if self.supports(operation) else None

    def build_conversation_action(
        self, *, user_input: str, assessment: TaskAssessment
    ) -> ConversationAction | None:
        if assessment.requires_tools and "future value" in user_input.lower():
            return ConversationAction(operation="read", parameters={})
        return None

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        assert operation == "read"
        assert isinstance(parameters, _FutureParameters)
        return CapabilityResult(output={"value": parameters.value, "source": "future-test-v1", "retrieved_at": datetime.now(UTC).isoformat()})

    def format_conversation_result(self, result: ActionResult) -> str | None:
        return str(result.output["value"]) if isinstance(result.output, dict) else None


class _FutureEvidenceProvider:
    capability = "future_value.read"

    def produce_evidence(self, *, task_id, action_result: ActionResult) -> EvidenceCollection:
        output = action_result.output
        assert isinstance(output, dict)
        return EvidenceCollection(task_id=task_id, items=(Evidence(
            evidence_id=uuid4(), source_type=EvidenceSourceType.STRUCTURED_DATASET,
            source_identifier=str(output["source"]), claim="future value",
            content=str({"value": output["value"]}), reliability=EvidenceReliability.AUTHORITATIVE,
            retrieved_at=datetime.fromisoformat(str(output["retrieved_at"])),
        ),))

    def claims(self, action_result: ActionResult) -> tuple[Claim, ...]:
        output = action_result.output
        assert isinstance(output, dict)
        return (Claim(text="Future deterministic value", claim_type=ClaimType.EXACT_VALUE, expected_value=str(output["value"])),)

    def fallback_answer(self, action_result: ActionResult) -> str:
        return "42"


def _future_runtime(*, providers: EvidenceProviderRegistry) -> CognitiveRuntime:
    registry = ConversationCapabilityRegistry((_FutureCapability(),))
    return CognitiveRuntime(
        response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")),
        conversation_bridge=ConversationCapabilityBridge(
            registry=registry,
            executor=ActionExecutor(
                registry=registry.executor_registry,
                allowed_capabilities=registry.names(),
                timeout=0.5,
            ),
        ),
        evidence_providers=providers,
    )


@pytest.mark.anyio
async def test_a_dynamically_registered_provider_uses_the_normal_runtime_path() -> None:
    providers = EvidenceProviderRegistry()
    providers.register(_FutureEvidenceProvider())
    execution = await _future_runtime(providers=providers).handle_conversation(
        user_input="Read the future value.", history=[], assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.capability_name == "future_value"
    assert execution.capability_operation == "read"
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.answer == "42"
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.VERIFIED
    assert execution.verification.supporting_evidence[0].source_identifier == "future-test-v1"


@pytest.mark.anyio
async def test_success_without_an_evidence_provider_is_not_automatically_trusted() -> None:
    execution = await _future_runtime(providers=EvidenceProviderRegistry()).handle_conversation(
        user_input="Read the future value.", history=[], assessment=_assessment(),
    )

    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.action_status == ActionStatus.COMPLETED.value
    assert execution.observation_outcome == "success"
    assert execution.verification is not None
    assert execution.verification.status is VerificationStatus.UNVERIFIED
    assert not execution.verification.supporting_evidence


@pytest.mark.anyio
async def test_runtime_evidence_is_task_local_across_independent_requests() -> None:
    runtime = CognitiveRuntime(response_generator=AsyncMock(side_effect=RuntimeError("no prose adapter")))
    pi = await runtime.handle_conversation(
        user_input="What are the first 100 digits of pi?", history=[], assessment=_assessment(),
    )
    science = await runtime.handle_conversation(
        user_input="What are the Moon and Sun temperatures?", history=[], assessment=_assessment(),
    )

    assert pi.verification is not None and science.verification is not None
    assert pi.verification.task_id != science.verification.task_id
    assert pi.verification.supporting_evidence[0].evidence_id != science.verification.supporting_evidence[0].evidence_id
    assert pi.verification.supporting_evidence[0].source_identifier == "pi-constant-v1"
    assert science.verification.supporting_evidence[0].source_identifier == "NASA-facts-v1"


def test_generic_verifier_reports_partial_multi_claim_support() -> None:
    task_id = uuid4()
    evidence = EvidenceCollection(task_id=task_id, items=(Evidence(
        evidence_id=uuid4(), source_type=EvidenceSourceType.STRUCTURED_DATASET,
        source_identifier="test:v1", claim="two values", content='{"first":"alpha","second":"beta"}',
        reliability=EvidenceReliability.AUTHORITATIVE, retrieved_at=datetime.now(UTC),
    ),))
    result = EvidenceVerifier().verify(
        task_id=task_id, candidate_answer="The first value is alpha.", evidence=evidence,
        claims=(
            Claim(text="First value", expected_value="alpha"),
            Claim(text="Second value", expected_value="beta"),
        ),
    )

    assert result.status is VerificationStatus.PARTIALLY_VERIFIED
    assert result.supporting_evidence == evidence.items
