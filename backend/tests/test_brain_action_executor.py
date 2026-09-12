from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.brain.action_builder import ActionBuildError, ActionRequestBuilder
from app.brain.actions import (
    MAX_ACTION_OUTPUT_BYTES,
    ActionError,
    ActionErrorCode,
    ActionRequest,
    ActionResult,
    ActionStatus,
    CapabilityResult,
)
from app.brain.capabilities import CapabilityRegistry
from app.brain.executor import ActionExecutor
from app.brain.planner import ExecutionPlan, PlanCapability, PlanStatus, PlanStep
from app.brain.plan_execution import PlanExecutionCoordinator, PlanExecutionStatus, PlanStepState
from app.brain.runtime import CognitiveRuntime
from app.brain.schemas import (
    Complexity,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
)
from app.brain.working_memory import create_cognitive_context


@pytest.fixture
def anyio_backend() -> str:
    """The application runs its executor on asyncio (FastAPI/Uvicorn)."""

    return "asyncio"


class NumberParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int = Field(ge=0, le=100)


class MockCapability:
    name = "computation"
    operations = ("double",)

    def __init__(self, *, error: Exception | None = None, delay: float = 0, malformed: bool = False) -> None:
        self.calls: list[tuple[str, NumberParameters]] = []
        self.error = error
        self.delay = delay
        self.malformed = malformed

    def supports(self, operation: str) -> bool:
        return operation in self.operations

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return NumberParameters if self.supports(operation) else None

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        assert isinstance(parameters, NumberParameters)
        self.calls.append((operation, parameters))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        if self.malformed:
            return {"not": "a capability result"}  # type: ignore[return-value]
        return CapabilityResult(output={"value": parameters.value * 2})


def _context():
    assessment = TaskAssessment(
        task_type=TaskType.COMPUTATION,
        complexity=Complexity.SIMPLE,
        requires_tools=True,
        requires_memory=False,
        requires_planning=True,
        requires_verification=False,
        direct_response_allowed=False,
        reasoning_budget=ReasoningBudget.LOW,
        urgency=Urgency.NORMAL,
        user_goal="Perform bounded computation.",
        rationale="Tests need a task-local context.",
    )
    return create_cognitive_context(
        user_input="Sensitive input should never be logged.", history=[], assessment=assessment
    )


def _action(context, **overrides: Any) -> ActionRequest:
    values = {
        "task_id": context.task.task_id,
        "capability": "computation",
        "operation": "double",
        "parameters": {"value": 4},
    }
    values.update(overrides)
    return ActionRequest(**values)


def _executor(capability: MockCapability | None = None, **kwargs: Any) -> ActionExecutor:
    registry = CapabilityRegistry((capability or MockCapability(),))
    return ActionExecutor(registry=registry, allowed_capabilities={"computation"}, **kwargs)


def test_action_request_is_strict_and_generates_an_identity() -> None:
    context = _context()
    action = _action(context)
    assert action.action_id
    assert action.parameters == {"value": 4}

    with pytest.raises(ValidationError):
        ActionRequest(capability="computation", operation="double")
    with pytest.raises(ValidationError):
        _action(context, task_id="not-a-uuid")
    with pytest.raises(ValidationError):
        _action(context, capability="")
    with pytest.raises(ValidationError):
        _action(context, operation="")
    with pytest.raises(ValidationError):
        _action(context, extra_field=True)
    with pytest.raises(ValidationError):
        _action(context, parameters={str(index): index for index in range(33)})


@pytest.mark.parametrize("status", list(ActionStatus))
def test_action_result_has_typed_statuses(status: ActionStatus) -> None:
    kwargs: dict[str, Any] = {
        "action_id": uuid4(), "task_id": uuid4(), "capability": "test", "operation": "run",
        "status": status, "duration_ms": 0,
    }
    if status is ActionStatus.COMPLETED:
        kwargs["output"] = {"ok": True}
    else:
        kwargs["error"] = ActionError(code=ActionErrorCode.EXECUTION_FAILED, message="Failed safely.")
    assert ActionResult(**kwargs).status is status

    invalid = {**kwargs, "error": None}
    if status is ActionStatus.COMPLETED:
        invalid["error"] = ActionError(code=ActionErrorCode.EXECUTION_FAILED, message="Invalid")
    with pytest.raises(ValidationError):
        ActionResult(**invalid)


def test_registry_is_explicit_deterministic_and_isolated() -> None:
    capability = MockCapability()
    registry = CapabilityRegistry()
    registry.register(capability)
    assert registry.get("computation") is capability
    assert registry.has("computation")
    assert registry.names() == ("computation",)
    assert registry.supported_operations("computation") == ("double",)
    assert registry.get("unknown") is None
    with pytest.raises(ValueError, match="already registered"):
        registry.register(capability)
    assert CapabilityRegistry().names() == ()


@pytest.mark.anyio
async def test_executor_dispatches_one_valid_action_and_records_facts(caplog: pytest.LogCaptureFixture) -> None:
    context = _context()
    capability = MockCapability()
    caplog.set_level("INFO", logger="app.brain.executor")
    result = await _executor(capability).execute(context=context, action=_action(context))

    assert result.status is ActionStatus.COMPLETED
    assert result.output == {"value": 8}
    assert len(capability.calls) == 1
    assert len(context.working_memory.action_executions) == 1
    assert context.working_memory.observations == []
    assert "Sensitive input" not in caplog.text
    assert "parameters" not in caplog.text
    assert str(result.action_id) in caplog.text
    assert "duration_ms" in caplog.text


@pytest.mark.anyio
async def test_same_action_identity_is_executed_only_once_under_concurrency() -> None:
    context = _context()
    capability = MockCapability(delay=0.01)
    executor = _executor(capability)
    action = _action(context)

    first, second = await asyncio.gather(
        executor.execute(context=context, action=action),
        executor.execute(context=context, action=action),
    )

    assert first.status is ActionStatus.COMPLETED
    assert second.status is ActionStatus.COMPLETED
    assert first.action_id == second.action_id == action.action_id
    assert len(capability.calls) == 1


@pytest.mark.anyio
async def test_registered_but_disallowed_capability_is_rejected_without_dispatch() -> None:
    context = _context()
    capability = MockCapability()
    executor = ActionExecutor(registry=CapabilityRegistry((capability,)), allowed_capabilities=())
    result = await executor.execute(context=context, action=_action(context))
    assert result.status is ActionStatus.REJECTED
    assert result.error and result.error.code is ActionErrorCode.CAPABILITY_NOT_ALLOWED
    assert capability.calls == []


@pytest.mark.anyio
async def test_unknown_capability_and_operation_are_unsupported_without_dispatch() -> None:
    context = _context()
    capability = MockCapability()
    executor = _executor(capability)
    unknown_capability = await executor.execute(
        context=context, action=_action(context, capability="unknown")
    )
    unknown_operation = await executor.execute(
        context=context, action=_action(context, operation="dangerous_internal_method")
    )
    assert unknown_capability.status is ActionStatus.UNSUPPORTED
    assert unknown_capability.error and unknown_capability.error.code is ActionErrorCode.CAPABILITY_NOT_REGISTERED
    assert unknown_operation.status is ActionStatus.UNSUPPORTED
    assert unknown_operation.error and unknown_operation.error.code is ActionErrorCode.OPERATION_NOT_REGISTERED
    assert capability.calls == []


@pytest.mark.anyio
async def test_task_mismatch_and_invalid_parameters_are_rejected_before_dispatch() -> None:
    context = _context()
    capability = MockCapability()
    executor = _executor(capability)
    wrong_task = await executor.execute(context=context, action=_action(context, task_id=uuid4()))
    invalid_parameters = await executor.execute(
        context=context, action=_action(context, parameters={"value": "4", "unexpected": True})
    )
    assert wrong_task.error and wrong_task.error.code is ActionErrorCode.TASK_MISMATCH
    assert invalid_parameters.error and invalid_parameters.error.code is ActionErrorCode.INVALID_PARAMETERS
    assert capability.calls == []


@pytest.mark.anyio
async def test_capability_failure_timeout_and_invalid_return_are_truthful() -> None:
    context = _context()
    failure = MockCapability(error=RuntimeError("do not leak"))
    failed = await _executor(failure).execute(context=context, action=_action(context))
    assert failed.status is ActionStatus.FAILED
    assert failed.error and failed.error.code is ActionErrorCode.EXECUTION_FAILED
    assert "do not leak" not in failed.error.message
    assert len(failure.calls) == 1

    slow = MockCapability(delay=0.05)
    timeout_context = _context()
    timed_out = await _executor(slow, timeout=0.001).execute(
        context=timeout_context, action=_action(timeout_context)
    )
    assert timed_out.status is ActionStatus.TIMED_OUT
    assert timed_out.error and timed_out.error.code is ActionErrorCode.EXECUTION_TIMEOUT
    assert len(slow.calls) == 1

    malformed_context = _context()
    malformed = await _executor(MockCapability(malformed=True)).execute(
        context=malformed_context, action=_action(malformed_context)
    )
    assert malformed.status is ActionStatus.FAILED
    assert malformed.error and malformed.error.code is ActionErrorCode.INTERNAL_ERROR


@pytest.mark.anyio
async def test_output_is_bounded_and_raw_output_is_not_added_to_working_memory() -> None:
    class LargeCapability(MockCapability):
        async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
            self.calls.append((operation, parameters))
            return CapabilityResult(output="x" * (MAX_ACTION_OUTPUT_BYTES + 1))

    context = _context()
    result = await _executor(LargeCapability()).execute(context=context, action=_action(context))
    assert result.status is ActionStatus.COMPLETED
    assert result.output_truncated is True
    assert result.output and result.output["truncated"] is True
    assert context.working_memory.observations == []


def _plan(context) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=uuid4(), task_id=context.task.task_id, status=PlanStatus.PLANNED,
        objective=context.task.objective,
        steps=[
            PlanStep(id="step_1", description="First", purpose="First", required_capabilities=[PlanCapability.COMPUTATION]),
            PlanStep(id="step_2", description="Second", purpose="Second", required_capabilities=[PlanCapability.COMPUTATION]),
            PlanStep(id="step_3", description="Third", purpose="Third", required_capabilities=[PlanCapability.COMPUTATION]),
        ],
    )


@pytest.mark.anyio
async def test_plan_translation_executes_only_selected_step_and_never_plans_again() -> None:
    context = _context()
    plan = _plan(context)
    action = ActionRequestBuilder.from_plan_step(
        context=context, plan=plan, step_id="step_2", capability="computation", operation="double", parameters={"value": 2}
    )
    capability = MockCapability()
    runtime = CognitiveRuntime(action_executor=_executor(capability))
    result = await runtime.execute_action(context=context, action=action)

    assert result.status is ActionStatus.COMPLETED
    assert len(capability.calls) == 1  # step_2
    assert [record.plan_step_id for record in context.working_memory.action_executions] == ["step_2"]
    assert "step_1" not in [record.plan_step_id for record in context.working_memory.action_executions]
    assert "step_3" not in [record.plan_step_id for record in context.working_memory.action_executions]


@pytest.mark.anyio
async def test_dependencies_require_actual_prior_executor_result() -> None:
    context = _context()
    plan = _plan(context)
    plan.steps[1].depends_on = ["step_1"]
    dependent = ActionRequestBuilder.from_plan_step(
        context=context, plan=plan, step_id="step_2", capability="computation", operation="double", parameters={"value": 2}
    )
    capability = MockCapability()
    executor = _executor(capability)
    rejected = await executor.execute(context=context, action=dependent)
    assert rejected.error and rejected.error.code is ActionErrorCode.PREREQUISITE_NOT_SATISFIED
    assert capability.calls == []

    prerequisite = ActionRequestBuilder.from_plan_step(
        context=context, plan=plan, step_id="step_1", capability="computation", operation="double", parameters={"value": 1}
    )
    await executor.execute(context=context, action=prerequisite)
    completed = await executor.execute(context=context, action=dependent)
    assert completed.status is ActionStatus.COMPLETED
    assert len(capability.calls) == 2


@pytest.mark.anyio
async def test_application_built_plan_graph_orders_steps_and_blocks_dependents() -> None:
    context = _context()
    plan = _plan(context)
    plan.steps[1].depends_on = ["step_1"]
    actions = {
        step.id: ActionRequestBuilder.from_plan_step(
            context=context,
            plan=plan,
            step_id=step.id,
            capability="computation",
            operation="double",
            parameters={"value": index + 1},
        )
        for index, step in enumerate(plan.steps)
    }
    capability = MockCapability()
    execution = await PlanExecutionCoordinator(executor=_executor(capability)).execute(
        context=context,
        plan=plan,
        actions=actions,
    )

    assert execution.status is PlanExecutionStatus.COMPLETED
    assert [item.step_id for item in execution.steps] == ["step_1", "step_2", "step_3"]
    assert all(item.state is PlanStepState.COMPLETED for item in execution.steps)
    assert len(capability.calls) == 3

    failed_context = _context()
    failed_plan = _plan(failed_context)
    failed_plan.steps[1].depends_on = ["step_1"]
    failed_actions = {
        step.id: ActionRequestBuilder.from_plan_step(
            context=failed_context,
            plan=failed_plan,
            step_id=step.id,
            capability="computation",
            operation="double",
            parameters={"value": 1},
        )
        for step in failed_plan.steps
    }
    failed = await PlanExecutionCoordinator(executor=_executor(MockCapability(error=RuntimeError()))).execute(
        context=failed_context,
        plan=failed_plan,
        actions=failed_actions,
    )

    assert failed.status is PlanExecutionStatus.FAILED
    assert failed.steps[0].state is PlanStepState.FAILED
    assert failed.steps[1].state is PlanStepState.BLOCKED


@pytest.mark.anyio
async def test_executor_prerequisites_are_scoped_to_the_current_context() -> None:
    context_a = _context()
    context_b = _context()
    plan_a = _plan(context_a)
    executor = _executor(MockCapability())
    prerequisite_a = ActionRequestBuilder.from_plan_step(
        context=context_a,
        plan=plan_a,
        step_id="step_1",
        capability="computation",
        operation="double",
        parameters={"value": 1},
    )
    await executor.execute(context=context_a, action=prerequisite_a)
    dependent_b = ActionRequest(
        task_id=context_b.task.task_id,
        capability="computation",
        operation="double",
        parameters={"value": 2},
        # Deliberately reuse A's plan identity: B still has no local step_1.
        plan_id=plan_a.plan_id,
        plan_step_id="step_2",
        depends_on=("step_1",),
    )

    result_b = await executor.execute(context=context_b, action=dependent_b)

    assert result_b.status is ActionStatus.REJECTED
    assert result_b.error and result_b.error.code is ActionErrorCode.PREREQUISITE_NOT_SATISFIED
    assert len(context_a.working_memory.action_executions) == 1
    assert context_b.working_memory.action_executions[-1].status is ActionStatus.REJECTED


def test_builder_rejects_cross_task_and_undeclared_capability() -> None:
    context = _context()
    other_context = _context()
    plan = _plan(context)
    with pytest.raises(ActionBuildError):
        ActionRequestBuilder.from_plan_step(
            context=other_context, plan=plan, step_id="step_1", capability="computation", operation="double", parameters={}
        )
    with pytest.raises(ActionBuildError):
        ActionRequestBuilder.from_plan_step(
            context=context, plan=plan, step_id="step_1", capability="retrieval", operation="get", parameters={}
        )


@pytest.mark.anyio
async def test_default_production_executor_never_fakes_success_or_leaks_between_contexts() -> None:
    context_a = _context()
    context_b = _context()
    result = await ActionExecutor().execute(context=context_a, action=_action(context_a))
    assert result.status is ActionStatus.UNSUPPORTED
    assert result.error and result.error.code is ActionErrorCode.CAPABILITY_NOT_REGISTERED
    assert len(context_a.working_memory.action_executions) == 1
    assert context_b.working_memory.action_executions == []
    assert context_b.working_memory.observations == []


@pytest.mark.anyio
async def test_executor_never_calls_the_llm() -> None:
    context = _context()
    with patch(
        "app.brain.client.OllamaStructuredClient.generate_structured",
        new_callable=AsyncMock,
    ) as model_call:
        result = await _executor().execute(context=context, action=_action(context))
    assert result.status is ActionStatus.COMPLETED
    model_call.assert_not_awaited()


def test_malicious_capability_name_is_schema_rejected_without_dynamic_execution() -> None:
    context = _context()
    with pytest.raises(ValidationError):
        _action(context, capability="__import__('os')")
