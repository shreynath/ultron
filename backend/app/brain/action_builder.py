"""Controlled translation from a descriptive plan step to an ActionRequest."""
from __future__ import annotations

from typing import Any

from app.brain.actions import ActionRequest
from app.brain.planner import ExecutionPlan, PlanStatus
from app.brain.working_memory import CognitiveContext


class ActionBuildError(ValueError):
    """The application cannot safely construct the requested action."""


class ActionRequestBuilder:
    """Create requests with application-generated identity and plan metadata.

    The caller must supply a capability, operation, and parameters from its
    own fixed application mapping.  Planner prose is never converted into an
    operation name, callable, path, command, or implementation object.
    """

    @staticmethod
    def from_plan_step(
        *,
        context: CognitiveContext,
        plan: ExecutionPlan,
        step_id: str,
        capability: str,
        operation: str,
        parameters: dict[str, Any],
    ) -> ActionRequest:
        if plan.status is not PlanStatus.PLANNED:
            raise ActionBuildError("Actions can only be created from a planned execution plan")
        if plan.task_id != context.task.task_id:
            raise ActionBuildError("Plan does not belong to the supplied cognitive context")
        step = next((item for item in plan.steps if item.id == step_id), None)
        if step is None:
            raise ActionBuildError("Selected plan step does not exist")
        if capability not in {item.value for item in step.required_capabilities}:
            raise ActionBuildError("Selected capability was not declared for this plan step")
        return ActionRequest(
            task_id=context.task.task_id,
            capability=capability,
            operation=operation,
            parameters=parameters,
            plan_id=plan.plan_id,
            plan_step_id=step.id,
            depends_on=tuple(step.depends_on),
        )
