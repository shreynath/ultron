"""One-action executor and the narrow capability authorization boundary."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Collection
from typing import Any

from anyio import fail_after
from pydantic import ValidationError

from app.brain.actions import (
    MAX_ACTION_OUTPUT_BYTES,
    ActionError,
    ActionErrorCode,
    ActionRequest,
    ActionResult,
    ActionStatus,
    CapabilityResult,
)
from app.brain.capabilities import Capability, CapabilityRegistry
from app.brain.working_memory import ActionExecutionRecord, CognitiveContext
from app.config import settings

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Validate and dispatch at most one concrete action.

    This class deliberately has no planner or LLM dependency.  It does not
    iterate plans, retry calls, select alternatives, or invoke itself.
    """

    def __init__(
        self,
        *,
        registry: CapabilityRegistry | None = None,
        allowed_capabilities: Collection[str] = (),
        timeout: float | None = None,
    ) -> None:
        self._registry = registry or CapabilityRegistry()
        self._allowed_capabilities = frozenset(allowed_capabilities)
        configured_timeout = settings.brain_action_timeout if timeout is None else timeout
        if configured_timeout <= 0 or configured_timeout > 60:
            raise ValueError("Action timeout must be greater than zero and no more than 60 seconds")
        self._timeout = configured_timeout

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def allowed_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._allowed_capabilities))

    async def execute(
        self,
        *,
        context: CognitiveContext,
        action: ActionRequest,
    ) -> ActionResult:
        """Execute only ``action`` once after all application checks succeed."""

        started = time.perf_counter()
        if not isinstance(context, CognitiveContext) or not isinstance(action, ActionRequest):
            # This is an internal programming-contract violation. There is no
            # safe identity available from an arbitrary object to form a result.
            raise TypeError("ActionExecutor requires validated CognitiveContext and ActionRequest")

        result = self._validate_request(context=context, action=action, started=started)
        if result is None:
            capability = self._registry.get(action.capability)
            assert capability is not None  # validated immediately above
            result = await self._dispatch_once(
                context=context,
                action=action,
                capability=capability,
                started=started,
            )

        self._record(context, action, result)
        self._log(action, result)
        return result

    def _validate_request(
        self,
        *,
        context: CognitiveContext,
        action: ActionRequest,
        started: float,
    ) -> ActionResult | None:
        if action.task_id != context.task.task_id:
            return self._result(
                action, ActionStatus.REJECTED, started,
                ActionErrorCode.TASK_MISMATCH, "Action does not belong to this cognitive task."
            )
        capability = self._registry.get(action.capability)
        if capability is None:
            return self._result(
                action, ActionStatus.UNSUPPORTED, started,
                ActionErrorCode.CAPABILITY_NOT_REGISTERED, "Capability is not registered."
            )
        if action.capability not in self._allowed_capabilities:
            return self._result(
                action, ActionStatus.REJECTED, started,
                ActionErrorCode.CAPABILITY_NOT_ALLOWED, "Capability is not permitted by application policy."
            )
        if not capability.supports(action.operation):
            return self._result(
                action, ActionStatus.UNSUPPORTED, started,
                ActionErrorCode.OPERATION_NOT_REGISTERED, "Operation is not registered for this capability."
            )
        if not self._dependencies_satisfied(context, action):
            return self._result(
                action, ActionStatus.REJECTED, started,
                ActionErrorCode.PREREQUISITE_NOT_SATISFIED,
                "Required plan-step execution has not completed.",
            )
        return None

    async def _dispatch_once(
        self,
        *,
        context: CognitiveContext,
        action: ActionRequest,
        capability: Capability,
        started: float,
    ) -> ActionResult:
        parameter_model = capability.parameter_model(action.operation)
        if parameter_model is None:
            return self._result(
                action, ActionStatus.UNSUPPORTED, started,
                ActionErrorCode.OPERATION_NOT_REGISTERED, "Operation is not registered for this capability."
            )
        try:
            parameters = parameter_model.model_validate(action.parameters)
        except ValidationError:
            return self._result(
                action, ActionStatus.REJECTED, started,
                ActionErrorCode.INVALID_PARAMETERS, "Action parameters failed validation."
            )

        # This is the single and only capability invocation in this method.
        try:
            # AnyIO keeps the execution deadline correct under FastAPI and
            # under both asyncio and Trio test backends. This still invokes the
            # selected capability exactly once.
            with fail_after(self._timeout):
                raw_result = await capability.execute(action.operation, parameters)
        except TimeoutError:
            return self._result(
                action, ActionStatus.TIMED_OUT, started,
                ActionErrorCode.EXECUTION_TIMEOUT, "Action execution exceeded its deadline."
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # capability exceptions are never exposed through the cognitive API
            return self._result(
                action, ActionStatus.FAILED, started,
                ActionErrorCode.EXECUTION_FAILED, "Capability execution failed."
            )

        if not isinstance(raw_result, CapabilityResult):
            return self._result(
                action, ActionStatus.FAILED, started,
                ActionErrorCode.INTERNAL_ERROR, "Capability returned an invalid result."
            )
        output, output_truncated = _bounded_output(raw_result.output)
        result = ActionResult(
            action_id=action.action_id,
            task_id=action.task_id,
            capability=action.capability,
            operation=action.operation,
            status=ActionStatus.COMPLETED,
            output=output,
            output_truncated=output_truncated,
            duration_ms=_duration_ms(started),
        )
        return result

    @staticmethod
    def _dependencies_satisfied(context: CognitiveContext, action: ActionRequest) -> bool:
        if not action.depends_on:
            return True
        if action.plan_id is None:
            return False
        completed_steps = {
            record.plan_step_id
            for record in context.working_memory.action_executions
            if record.plan_id == action.plan_id
            and record.status is ActionStatus.COMPLETED
            and record.plan_step_id is not None
        }
        return set(action.depends_on).issubset(completed_steps)

    def _result(
        self,
        action: ActionRequest,
        status: ActionStatus,
        started: float,
        code: ActionErrorCode,
        message: str,
    ) -> ActionResult:
        return ActionResult(
            action_id=action.action_id,
            task_id=action.task_id,
            capability=action.capability,
            operation=action.operation,
            status=status,
            error=ActionError(code=code, message=message),
            duration_ms=_duration_ms(started),
        )

    def _record(
        self,
        context: CognitiveContext,
        action: ActionRequest,
        result: ActionResult,
    ) -> None:
        context.working_memory.add_action_execution(
            ActionExecutionRecord(
                action_id=result.action_id,
                plan_id=action.plan_id,
                plan_step_id=action.plan_step_id,
                status=result.status,
            )
        )

    @staticmethod
    def _log(action: ActionRequest, result: ActionResult) -> None:
        logger.info(
            "brain.action action_id=%s task_id=%s capability=%s operation=%s status=%s "
            "duration_ms=%.1f failure_code=%s",
            action.action_id,
            action.task_id,
            action.capability,
            action.operation,
            result.status.value,
            result.duration_ms,
            result.error.code.value if result.error else None,
        )


def _duration_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _bounded_output(value: Any) -> tuple[Any | None, bool]:
    """Bound output before it can reach an action caller or working memory."""

    if value is None:
        return None, False
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_type)
    except (TypeError, ValueError):
        return {"summary": "Capability returned a non-serializable output."}, True
    if len(encoded.encode("utf-8")) <= MAX_ACTION_OUTPUT_BYTES:
        return value, False
    return {
        "truncated": True,
        "summary": "Capability output exceeded the action output limit.",
        "limit_bytes": MAX_ACTION_OUTPUT_BYTES,
    }, True


def _json_type(value: Any) -> str:
    return f"<{type(value).__name__}>"
