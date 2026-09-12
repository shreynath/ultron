"""Bounded graph execution for application-built multi-step plans.

Planner prose is never executable. Callers must provide an ``ActionRequest``
for each intended step through an application-owned mapping, after which this
module enforces dependency order, per-step authorization/observation, and
failure/cancellation propagation.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionRequest, ActionResult, ActionStatus
from app.brain.executor import ActionExecutor
from app.brain.observation import ExecutionObservation, ObservationInterpreter
from app.brain.planner import ExecutionPlan, PlanStatus
from app.brain.verification import VerificationResult
from app.brain.working_memory import CognitiveContext


class PlanStepState(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class PlanExecutionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class PlanStepExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    state: PlanStepState
    action: ActionResult | None = None
    observation: ExecutionObservation | None = None
    verification: VerificationResult | None = None
    reason: str | None = Field(default=None, max_length=500)


class PlanExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    task_id: str
    status: PlanExecutionStatus
    steps: tuple[PlanStepExecution, ...]


Verifier = Callable[[ActionResult, ExecutionObservation], VerificationResult | None]


class PlanExecutionError(ValueError):
    """The supplied action graph does not match the validated plan contract."""


class PlanExecutionCoordinator:
    """Execute a finite, application-authorized action graph once.

    It has no planner, model, registry mutation, retry, or re-planning access.
    Cancellation is deliberately allowed to propagate to the executor/capability
    stack instead of being transformed into a successful aggregate result.
    """

    def __init__(
        self,
        *,
        executor: ActionExecutor,
        observer: ObservationInterpreter | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self._executor = executor
        self._observer = observer or ObservationInterpreter()
        self._verifier = verifier

    async def execute(
        self,
        *,
        context: CognitiveContext,
        plan: ExecutionPlan,
        actions: dict[str, ActionRequest],
    ) -> PlanExecution:
        self._validate_graph(context=context, plan=plan, actions=actions)
        results: dict[str, PlanStepExecution] = {}

        # Every pass completes at least one ready step or marks a remaining
        # step blocked. The validated plan is acyclic and max 12 steps, so this
        # cannot become an uncontrolled execution loop.
        while len(results) < len(plan.steps):
            progressed = False
            for step in plan.steps:
                if step.id in results:
                    continue
                dependency_results = [results.get(dependency) for dependency in step.depends_on]
                if any(
                    result is not None
                    and result.state in {PlanStepState.FAILED, PlanStepState.BLOCKED, PlanStepState.CANCELLED}
                    for result in dependency_results
                ):
                    results[step.id] = PlanStepExecution(
                        step_id=step.id,
                        state=PlanStepState.BLOCKED,
                        reason="A required plan step did not complete.",
                    )
                    progressed = True
                    continue
                if not all(result is not None and result.state is PlanStepState.COMPLETED for result in dependency_results):
                    continue
                action = actions.get(step.id)
                if action is None:
                    results[step.id] = PlanStepExecution(
                        step_id=step.id,
                        state=PlanStepState.BLOCKED,
                        reason="No application-authorized action was supplied for this plan step.",
                    )
                    progressed = True
                    continue
                action_result = await self._executor.execute(context=context, action=action)
                observation = self._observer.observe(context=context, action_result=action_result)
                if action_result.status is not ActionStatus.COMPLETED:
                    results[step.id] = PlanStepExecution(
                        step_id=step.id,
                        state=PlanStepState.FAILED,
                        action=action_result,
                        observation=observation,
                        reason=action_result.error.code.value if action_result.error else "action_failed",
                    )
                else:
                    verification = self._verifier(action_result, observation) if self._verifier else None
                    results[step.id] = PlanStepExecution(
                        step_id=step.id,
                        state=PlanStepState.COMPLETED,
                        action=action_result,
                        observation=observation,
                        verification=verification,
                    )
                progressed = True
            if not progressed:
                # Defensive guard for an invalid graph loaded from persistence.
                for step in plan.steps:
                    if step.id not in results:
                        results[step.id] = PlanStepExecution(
                            step_id=step.id,
                            state=PlanStepState.BLOCKED,
                            reason="No executable dependency path remains.",
                        )

        ordered = tuple(results[step.id] for step in plan.steps)
        states = {item.state for item in ordered}
        if states == {PlanStepState.COMPLETED}:
            status = PlanExecutionStatus.COMPLETED
        elif PlanStepState.FAILED in states:
            status = PlanExecutionStatus.PARTIALLY_COMPLETED if PlanStepState.COMPLETED in states else PlanExecutionStatus.FAILED
        else:
            status = PlanExecutionStatus.BLOCKED
        return PlanExecution(
            plan_id=str(plan.plan_id),
            task_id=str(plan.task_id),
            status=status,
            steps=ordered,
        )

    @staticmethod
    def _validate_graph(
        *, context: CognitiveContext, plan: ExecutionPlan, actions: dict[str, ActionRequest]
    ) -> None:
        if plan.status is not PlanStatus.PLANNED:
            raise PlanExecutionError("Only a validated planned execution graph can run.")
        if plan.task_id != context.task.task_id:
            raise PlanExecutionError("Plan does not belong to this cognitive context.")
        valid_steps = {step.id for step in plan.steps}
        if not set(actions).issubset(valid_steps):
            raise PlanExecutionError("Action graph includes an unknown plan step.")
        for step_id, action in actions.items():
            step = next(step for step in plan.steps if step.id == step_id)
            if (
                action.task_id != context.task.task_id
                or action.plan_id != plan.plan_id
                or action.plan_step_id != step_id
                or tuple(action.depends_on) != tuple(step.depends_on)
            ):
                raise PlanExecutionError("Action metadata does not match its plan step.")
