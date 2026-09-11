"""Deterministic conversion of factual action results into observations."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from app.brain.actions import ActionResult, ActionStatus

if TYPE_CHECKING:
    from app.brain.working_memory import CognitiveContext

logger = logging.getLogger(__name__)

MAX_OBSERVATION_FACTS = 16
MAX_OBSERVATION_FACT_VALUE_CHARS = 512
MAX_OUTPUT_FACTS = 8


class ObservationOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    TIMED_OUT = "timed_out"


class ObservationSource(str, Enum):
    ACTION_RESULT = "action_result"
    CAPABILITY_OUTPUT = "capability_output"
    EXECUTOR_METADATA = "executor_metadata"


class ObservationFact(BaseModel):
    """One small, serializable fact; it is not arbitrary capability output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    value: StrictStr | StrictInt | StrictFloat | StrictBool | None

    @field_validator("value")
    @classmethod
    def bound_string_values(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_FACT_VALUE_CHARS:
            raise ValueError("Observation fact string value exceeds its bound")
        return value


class ExecutionObservation(BaseModel):
    """Immutable, bounded evidence derived from one exact ActionResult."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: UUID
    action_id: UUID
    task_id: UUID
    capability: str = Field(min_length=1, max_length=64)
    operation: str = Field(min_length=1, max_length=64)
    action_status: ActionStatus
    outcome: ObservationOutcome
    facts: tuple[ObservationFact, ...] = Field(max_length=MAX_OBSERVATION_FACTS)
    source: ObservationSource
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Observation timestamp must be timezone-aware")
        return value


class ObservationProcessingError(Exception):
    """A safe failure to process an action result into an observation."""


_OUTCOME_BY_STATUS = {
    ActionStatus.COMPLETED: ObservationOutcome.SUCCESS,
    ActionStatus.FAILED: ObservationOutcome.FAILURE,
    ActionStatus.REJECTED: ObservationOutcome.REJECTED,
    ActionStatus.UNSUPPORTED: ObservationOutcome.UNSUPPORTED,
    ActionStatus.TIMED_OUT: ObservationOutcome.TIMED_OUT,
}


class ObservationInterpreter:
    """Record only the bounded facts present in one actual action result.

    It has no planner, executor, capability, or LLM dependency.  Re-observing
    the same result in one context returns its existing immutable observation.
    """

    def observe(
        self,
        *,
        context: CognitiveContext,
        action_result: ActionResult,
    ) -> ExecutionObservation:
        # Importing here avoids a model/WorkingMemory import cycle while
        # retaining the runtime boundary check.
        from app.brain.working_memory import CognitiveContext as RuntimeCognitiveContext

        if not isinstance(context, RuntimeCognitiveContext) or not isinstance(action_result, ActionResult):
            raise ObservationProcessingError(
                "Observation requires a validated CognitiveContext and ActionResult"
            )
        if action_result.task_id != context.task.task_id:
            raise ObservationProcessingError("Action result does not belong to this cognitive task")

        existing = context.working_memory.get_execution_observation(action_result.action_id)
        if existing is not None:
            if (
                existing.task_id != action_result.task_id
                or existing.capability != action_result.capability
                or existing.operation != action_result.operation
                or existing.action_status is not action_result.status
            ):
                raise ObservationProcessingError("Action identity conflicts with an existing observation")
            return existing

        try:
            outcome = _OUTCOME_BY_STATUS[action_result.status]
            facts = _facts_from_result(action_result)
            observation = ExecutionObservation(
                observation_id=uuid4(),
                action_id=action_result.action_id,
                task_id=action_result.task_id,
                capability=action_result.capability,
                operation=action_result.operation,
                action_status=action_result.status,
                outcome=outcome,
                facts=facts,
                source=ObservationSource.ACTION_RESULT,
                created_at=datetime.now(UTC),
            )
        except ObservationProcessingError:
            raise
        except Exception as exc:  # output is untrusted data and can be malformed
            raise ObservationProcessingError("Action result could not be normalized safely") from exc

        context.working_memory.add_observation(observation)
        logger.info(
            "brain.observation observation_id=%s action_id=%s task_id=%s capability=%s "
            "operation=%s action_status=%s observation_outcome=%s source=%s",
            observation.observation_id,
            observation.action_id,
            observation.task_id,
            observation.capability,
            observation.operation,
            observation.action_status.value,
            observation.outcome.value,
            observation.source.value,
        )
        return observation


def _facts_from_result(action_result: ActionResult) -> tuple[ObservationFact, ...]:
    """Extract finite metadata plus safe scalar output facts deterministically."""

    facts = [
        ObservationFact(key="action_status", value=action_result.status.value),
        ObservationFact(key="duration_ms", value=float(action_result.duration_ms)),
        ObservationFact(key="output_present", value=action_result.output is not None),
        ObservationFact(key="output_truncated", value=action_result.output_truncated),
    ]
    if action_result.error is not None:
        facts.append(ObservationFact(key="error_code", value=action_result.error.code.value))

    if action_result.output is None:
        return tuple(facts)
    if action_result.output_truncated:
        facts.append(ObservationFact(key="output_details_omitted", value=True))
        return tuple(facts)

    output = action_result.output
    if _safe_scalar(output):
        facts.append(ObservationFact(key="output_value", value=output))
    elif isinstance(output, str):
        # Raw text can carry credentials, prompts, or unlimited instruction
        # content. Keep only a fact that text was intentionally omitted.
        facts.append(ObservationFact(key="output_text_omitted", value=True))
    elif isinstance(output, dict):
        facts.extend(_safe_mapping_facts(output, remaining=MAX_OBSERVATION_FACTS - len(facts)))
    else:
        facts.append(ObservationFact(key="output_details_omitted", value=True))
    return tuple(facts[:MAX_OBSERVATION_FACTS])


def _safe_mapping_facts(output: dict[Any, Any], *, remaining: int) -> list[ObservationFact]:
    facts: list[ObservationFact] = []
    for key in sorted(output):
        if len(facts) >= min(MAX_OUTPUT_FACTS, remaining):
            break
        if not isinstance(key, str) or not _safe_output_key(key):
            continue
        value = output[key]
        if _safe_scalar(value):
            facts.append(ObservationFact(key=f"output_{key}", value=value))
    return facts


def _safe_scalar(value: Any) -> bool:
    return value is None or type(value) in {bool, int, float}


def _safe_output_key(key: str) -> bool:
    if not key or len(key) > 56 or not key.replace("_", "").isalnum():
        return False
    lowered = key.lower()
    sensitive_parts = ("secret", "token", "password", "credential", "authorization", "cookie", "api_key")
    return not any(part in lowered for part in sensitive_parts)
