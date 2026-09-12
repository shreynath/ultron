from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.brain.errors import CognitiveRuntimeError
from app.brain.planner import ExecutionPlan, PlanFailureCategory, PlanStatus
from app.brain.runtime import (
    CognitiveRuntime,
    ExecutionRoute,
    ExecutionStatus,
    required_capabilities,
    route_assessment,
    require_live_research,
)
from app.brain.schemas import (
    Complexity,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
    fallback_assessment,
)
from app.services.ollama_client import OllamaError


def _assessment(
    *,
    task_type: TaskType = TaskType.CASUAL,
    direct: bool = True,
    tools: bool = False,
    memory: bool = False,
    planning: bool = False,
    verification: bool = False,
) -> TaskAssessment:
    return TaskAssessment(
        task_type=task_type,
        complexity=Complexity.TRIVIAL if direct else Complexity.MODERATE,
        requires_tools=tools,
        requires_memory=memory,
        requires_planning=planning,
        requires_verification=verification,
        direct_response_allowed=direct,
        reasoning_budget=ReasoningBudget.MINIMAL if direct else ReasoningBudget.MEDIUM,
        urgency=Urgency.NORMAL,
        user_goal="Test the runtime route.",
        rationale="A deterministic test assessment.",
    )


@pytest.mark.parametrize(
    ("assessment", "expected"),
    [
        (_assessment(task_type=TaskType.CASUAL), ExecutionRoute.DIRECT_RESPONSE),
        (_assessment(task_type=TaskType.QUESTION), ExecutionRoute.DIRECT_RESPONSE),
        (_assessment(task_type=TaskType.CREATIVE), ExecutionRoute.DIRECT_RESPONSE),
        (
            _assessment(
                task_type=TaskType.COMPUTATION,
                direct=False,
                tools=True,
                verification=True,
            ),
            ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
        ),
        (
            _assessment(
                task_type=TaskType.RESEARCH,
                direct=False,
                tools=True,
                planning=True,
                verification=True,
            ),
            ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
        ),
        (
            _assessment(
                task_type=TaskType.CODING,
                direct=False,
                tools=True,
                planning=True,
                verification=True,
            ),
            ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
        ),
        (
            _assessment(task_type=TaskType.PLANNING, direct=False, planning=True),
            ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
        ),
        (
            _assessment(task_type=TaskType.UNKNOWN, direct=False, memory=True),
            ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED,
        ),
    ],
)
def test_route_assessment_is_deterministic(
    assessment: TaskAssessment,
    expected: ExecutionRoute,
) -> None:
    assert route_assessment(assessment) is expected


def test_direct_route_precedence_without_capability_requirements() -> None:
    assessment = _assessment(task_type=TaskType.QUESTION, direct=True)
    assert required_capabilities(assessment) == ()
    assert route_assessment(assessment) is ExecutionRoute.DIRECT_RESPONSE


def test_explicit_capability_requirement_precedes_direct_response() -> None:
    assessment = _assessment(direct=True, verification=True)
    assert route_assessment(assessment) is ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED


def test_explicit_current_information_overrides_an_underclassified_direct_route() -> None:
    adjusted = require_live_research(_assessment(task_type=TaskType.UNKNOWN), "Who is the current CEO of NVIDIA?")

    assert adjusted.task_type is TaskType.RETRIEVAL
    assert adjusted.requires_tools is True
    assert adjusted.requires_verification is True
    assert route_assessment(adjusted) is ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED


@pytest.mark.anyio
async def test_direct_response_calls_existing_generator_once() -> None:
    generator = AsyncMock(return_value="Direct answer.")
    planner = MagicMock()
    runtime = CognitiveRuntime(response_generator=generator, planner=planner)
    assessment = _assessment()

    execution = await runtime.handle(
        user_input="Hello Ultron",
        history=[{"role": "user", "content": "Earlier message"}],
        assessment=assessment,
        system_prompt="Test prompt",
    )

    assert execution.route is ExecutionRoute.DIRECT_RESPONSE
    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.answer == "Direct answer."
    assert execution.required_capabilities == ()
    generator.assert_awaited_once_with(
        user_message="Hello Ultron",
        system_prompt="Test prompt",
        history=[{"role": "user", "content": "Earlier message"}],
    )
    planner.create_plan.assert_not_called()


@pytest.mark.anyio
async def test_runtime_accepts_prebuilt_cognitive_context() -> None:
    generator = AsyncMock(return_value="Context answer.")
    runtime = CognitiveRuntime(response_generator=generator)
    assessment = _assessment()
    context = runtime.create_context(
        user_input="Use context",
        history=[{"role": "assistant", "content": "Prior response"}],
        assessment=assessment,
    )

    execution = await runtime.handle(context=context)

    assert execution.status is ExecutionStatus.COMPLETED
    generator.assert_awaited_once_with(
        user_message="Use context",
        system_prompt="You are a helpful assistant. Keep replies concise and conversational.",
        history=[{"role": "assistant", "content": "Prior response"}],
    )


@pytest.mark.anyio
async def test_future_capability_does_not_call_response_generator() -> None:
    generator = AsyncMock()
    planner = MagicMock()
    planner.create_plan = AsyncMock(
        return_value=ExecutionPlan(
            plan_id="b1dccb73-31eb-4b12-a917-f6f33597e3cc",
            task_id="cc1d48bf-5498-4c25-b377-a8108f9c0605",
            status=PlanStatus.FAILED,
            objective="Test the runtime route.",
            failure_category=PlanFailureCategory.SCHEMA_ERROR,
        )
    )
    runtime = CognitiveRuntime(response_generator=generator, planner=planner)
    assessment = _assessment(
        task_type=TaskType.RESEARCH,
        direct=False,
        tools=True,
        planning=True,
        verification=True,
    )

    execution = await runtime.handle(
        user_input="Research this claim.",
        history=[],
        assessment=assessment,
    )

    assert execution.route is ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED
    assert execution.status is ExecutionStatus.REQUIRES_FUTURE_CAPABILITY
    assert execution.required_capabilities == ("planning", "tools", "verification")
    assert "can't complete" in execution.answer
    assert execution.plan is not None
    assert execution.plan.status is PlanStatus.FAILED
    planner.create_plan.assert_awaited_once()
    generator.assert_not_awaited()


@pytest.mark.anyio
async def test_fallback_assessment_remains_direct_and_observable() -> None:
    generator = AsyncMock(return_value="Fallback reply.")
    execution = await CognitiveRuntime(response_generator=generator).handle(
        user_input="Anything",
        history=[],
        assessment=fallback_assessment(),
    )

    assert execution.route is ExecutionRoute.DIRECT_RESPONSE
    assert execution.status is ExecutionStatus.COMPLETED
    assert execution.assessment.used_fallback is True
    generator.assert_awaited_once()


@pytest.mark.anyio
async def test_response_failure_is_distinct_from_future_capability() -> None:
    generator = AsyncMock(side_effect=OllamaError("unavailable"))
    execution = await CognitiveRuntime(response_generator=generator).handle(
        user_input="Hello",
        history=[],
        assessment=_assessment(),
    )

    assert execution.route is ExecutionRoute.DIRECT_RESPONSE
    assert execution.status is ExecutionStatus.FAILED
    assert execution.required_capabilities == ()


@pytest.mark.anyio
async def test_runtime_rejects_unvalidated_assessment() -> None:
    runtime = CognitiveRuntime(response_generator=AsyncMock())
    with pytest.raises(CognitiveRuntimeError):
        await runtime.handle(
            user_input="Hello",
            history=[],
            assessment={"direct_response_allowed": True},  # type: ignore[arg-type]
        )
