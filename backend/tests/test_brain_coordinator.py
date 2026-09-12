from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import anyio
from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult, ActionStatus, CapabilityResult
from app.brain.conversation_bridge import ConversationCapabilityBridge
from app.brain.conversation_capabilities import (
    CapabilityDescriptor,
    ConversationAction,
    ConversationCapabilityRegistry,
)
from app.brain.coordinator import (
    BoundedExecutionCoordinator,
    ExecutionLifecycle,
    LifecycleTransitionError,
    OrchestrationStage,
)
from app.brain.executor import ActionExecutor
from app.brain.runtime import CognitiveRuntime, ExecutionStatus
from app.brain.schemas import Complexity, ReasoningBudget, TaskAssessment, TaskType, Urgency
from app.brain.verification import (
    Claim,
    Evidence,
    EvidenceCollection,
    EvidenceProviderRegistry,
    EvidenceReliability,
    EvidenceSourceType,
)


def _assessment(*, tools: bool = False) -> TaskAssessment:
    return TaskAssessment(
        task_type=TaskType.RETRIEVAL if tools else TaskType.QUESTION,
        complexity=Complexity.SIMPLE,
        requires_tools=tools,
        requires_memory=False,
        requires_planning=False,
        requires_verification=False,
        direct_response_allowed=not tools,
        reasoning_budget=ReasoningBudget.LOW,
        urgency=Urgency.NORMAL,
        user_goal="Return one bounded result.",
        rationale="The test exercises the established execution boundary.",
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

    def build_conversation_action(
        self, *, user_input: str, assessment: TaskAssessment
    ) -> ConversationAction | None:
        if user_input == "echo please":
            return ConversationAction(operation="lookup", parameters={"value": "actual result"})
        return None

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        self.calls += 1
        assert isinstance(parameters, EchoParameters)
        return CapabilityResult(output={"value": parameters.value})

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is ActionStatus.COMPLETED:
            return f"Echo result: {result.output['value']}."
        return None


class FailingEchoCapability(EchoCapability):
    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        self.calls += 1
        raise RuntimeError("provider unavailable")


class SlowEchoCapability(EchoCapability):
    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        self.calls += 1
        await anyio.sleep(1)
        return CapabilityResult(output={"value": "late"})


class CancelledEchoCapability(EchoCapability):
    def __init__(self) -> None:
        super().__init__()
        self.started = anyio.Event()
        self.cancelled = False

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        self.calls += 1
        self.started.set()
        try:
            await anyio.Event().wait()
        except BaseException:
            self.cancelled = True
            raise


class EchoEvidenceProvider:
    capability = "echo.lookup"

    def produce_evidence(
        self, *, task_id, action_result: ActionResult
    ) -> EvidenceCollection:
        return EvidenceCollection(task_id=task_id, items=(Evidence(
            evidence_id=uuid4(),
            source_type=EvidenceSourceType.CAPABILITY_RESULT,
            source_identifier="test:echo.lookup",
            claim="Echo result is actual result",
            content="actual result",
            reliability=EvidenceReliability.AUTHORITATIVE,
            retrieved_at=datetime.now(UTC),
        ),))

    def claims(self, action_result: ActionResult) -> tuple[Claim, ...]:
        return (Claim(text="Echo result is actual result", expected_value="actual result"),)

    def fallback_answer(self, action_result: ActionResult) -> str:
        return "Verified echo result: actual result."


def _bridge(capability: EchoCapability, *, timeout: float = 0.2) -> ConversationCapabilityBridge:
    registry = ConversationCapabilityRegistry((capability,))
    return ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(
            registry=registry.executor_registry,
            allowed_capabilities=registry.names(),
            timeout=timeout,
        ),
    )


def _coordinator(
    capability: EchoCapability | None = None,
    *,
    generator: AsyncMock | None = None,
    evidence: EvidenceProviderRegistry | None = None,
    timeout: float = 0.2,
) -> BoundedExecutionCoordinator:
    runtime = CognitiveRuntime(
        response_generator=generator or AsyncMock(return_value="The actual result is available."),
        conversation_bridge=_bridge(capability, timeout=timeout) if capability else ConversationCapabilityBridge(
            registry=ConversationCapabilityRegistry(),
        ),
        evidence_providers=evidence,
    )
    return BoundedExecutionCoordinator(runtime=runtime)


@pytest.mark.anyio
async def test_direct_response_has_a_complete_lifecycle() -> None:
    generator = AsyncMock(return_value="Direct answer.")
    result = await _coordinator(generator=generator).run(
        user_input="Hello", history=[], assessment=_assessment(),
    )

    assert result.lifecycle_stage is OrchestrationStage.COMPLETED
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert [event.to_stage for event in result.transitions] == [
        OrchestrationStage.RECEIVED,
        OrchestrationStage.ASSESSED,
        OrchestrationStage.DISPATCHED,
        OrchestrationStage.RESPONDING,
        OrchestrationStage.COMPLETED,
    ]
    generator.assert_awaited_once()


@pytest.mark.anyio
async def test_completed_action_is_executed_once_and_unverified_is_degraded() -> None:
    capability = EchoCapability()
    result = await _coordinator(capability).run(
        user_input="echo please", history=[], assessment=_assessment(tools=True),
    )

    assert capability.calls == 1
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.execution.action_status == "completed"
    assert result.lifecycle_stage is OrchestrationStage.DEGRADED
    assert result.degradation_reasons == ("verification_unverified",)
    assert [event.to_stage for event in result.transitions][-4:] == [
        OrchestrationStage.EXECUTING,
        OrchestrationStage.RESPONDING,
        OrchestrationStage.VERIFYING,
        OrchestrationStage.DEGRADED,
    ]


@pytest.mark.anyio
async def test_capability_answer_cannot_be_replaced_by_response_model_prose() -> None:
    capability = EchoCapability()
    generator = AsyncMock(return_value="The result is something else.")
    result = await _coordinator(
        capability,
        generator=generator,
        evidence=EvidenceProviderRegistry((EchoEvidenceProvider(),)),
    ).run(user_input="echo please", history=[], assessment=_assessment(tools=True))

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.execution.verification is not None
    assert result.execution.verification.status.value == "verified"
    assert result.execution.answer == "Echo result: actual result."
    assert result.execution.response_generation_status == "grounded_capability"
    generator.assert_not_awaited()
    assert result.lifecycle_stage is OrchestrationStage.COMPLETED


@pytest.mark.anyio
async def test_unavailable_response_generator_cannot_degrade_completed_capability() -> None:
    capability = EchoCapability()
    generator = AsyncMock(side_effect=RuntimeError("model unavailable"))
    result = await _coordinator(
        capability,
        generator=generator,
    ).run(user_input="echo please", history=[], assessment=_assessment(tools=True))

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.execution.response_generation_status == "grounded_capability"
    generator.assert_not_awaited()
    assert result.lifecycle_stage is OrchestrationStage.DEGRADED
    assert "verification_unverified" in result.degradation_reasons
    assert result.execution.answer == "Echo result: actual result."


@pytest.mark.anyio
async def test_missing_capability_is_blocked_without_an_action() -> None:
    result = await _coordinator().run(
        user_input="find a current answer", history=[], assessment=_assessment(tools=True),
    )

    assert result.lifecycle_stage is OrchestrationStage.BLOCKED
    assert result.execution.status is ExecutionStatus.REQUIRES_FUTURE_CAPABILITY
    assert result.execution.action_status is None


@pytest.mark.anyio
async def test_provider_failure_is_failed_not_degraded_or_completed() -> None:
    capability = FailingEchoCapability()
    result = await _coordinator(capability).run(
        user_input="echo please", history=[], assessment=_assessment(tools=True),
    )

    assert capability.calls == 1
    assert result.lifecycle_stage is OrchestrationStage.FAILED
    assert result.execution.status is ExecutionStatus.FAILED
    assert result.execution.action_status == "failed"


@pytest.mark.anyio
async def test_timeout_is_failed_not_completed() -> None:
    capability = SlowEchoCapability()
    result = await _coordinator(capability, timeout=0.01).run(
        user_input="echo please", history=[], assessment=_assessment(tools=True),
    )

    assert capability.calls == 1
    assert result.lifecycle_stage is OrchestrationStage.FAILED
    assert result.execution.action_status == "timed_out"
    assert result.execution.status is ExecutionStatus.FAILED


@pytest.mark.anyio
async def test_cancellation_propagates_without_a_successful_result() -> None:
    capability = CancelledEchoCapability()
    failures: list[BaseException] = []

    async def run_and_capture() -> None:
        try:
            await _coordinator(capability).run(
                user_input="echo please", history=[], assessment=_assessment(tools=True),
            )
        except BaseException as exc:
            failures.append(exc)

    async with anyio.create_task_group() as group:
        group.start_soon(run_and_capture)
        await capability.started.wait()
        group.cancel_scope.cancel()

    assert failures
    assert capability.calls == 1
    assert capability.cancelled is True


def test_lifecycle_rejects_terminal_and_backward_transitions() -> None:
    lifecycle = ExecutionLifecycle(uuid4())
    lifecycle.transition(OrchestrationStage.RECEIVED, component="test", status="received")
    lifecycle.transition(OrchestrationStage.ASSESSED, component="test", status="assessed")
    lifecycle.transition(OrchestrationStage.DISPATCHED, component="test", status="dispatched")
    lifecycle.transition(OrchestrationStage.COMPLETED, component="test", status="completed")

    with pytest.raises(LifecycleTransitionError, match="Illegal"):
        lifecycle.transition(OrchestrationStage.EXECUTING, component="test", status="started")
