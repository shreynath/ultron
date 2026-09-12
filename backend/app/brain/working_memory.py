"""Bounded, task-local cognitive context.

This module is intentionally deterministic. It stores only explicit task
state for one request and never retrieves, persists, or summarizes memory.
"""
from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.brain.errors import CognitiveRuntimeError
from app.brain.actions import ActionStatus
from app.brain.observation import ExecutionObservation
from app.brain.schemas import TaskAssessment
from app.brain.task_contract import TaskContract, build_task_contract
from app.brain.learning import RetrievalResult

logger = logging.getLogger(__name__)

MAX_RECENT_HISTORY = 8
MAX_OBSERVATIONS = 32
MAX_INTERMEDIATE_RESULTS = 32
MAX_ACTION_EXECUTIONS = 32
MAX_USER_INPUT_CHARS = 10_000
MAX_HISTORY_CONTENT_CHARS = 10_000
MAX_ARTIFACT_CONTENT_CHARS = 8_000
MAX_ARTIFACT_LABEL_CHARS = 100
MAX_RETRIEVED_KNOWLEDGE = 8


class HistoryMessage(BaseModel):
    """One user or assistant message in bounded conversational context."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_HISTORY_CONTENT_CHARS)


class Observation(BaseModel):
    """An explicit fact or output produced by a future execution subsystem."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=MAX_ARTIFACT_LABEL_CHARS)
    content: str = Field(min_length=1, max_length=MAX_ARTIFACT_CONTENT_CHARS)


class IntermediateResult(BaseModel):
    """A bounded, explicit task artifact; never a hidden reasoning trace."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=MAX_ARTIFACT_LABEL_CHARS)
    content: str = Field(min_length=1, max_length=MAX_ARTIFACT_CONTENT_CHARS)


class ActionExecutionRecord(BaseModel):
    """Minimal factual state used only to validate plan prerequisites."""

    model_config = ConfigDict(extra="forbid")

    action_id: UUID
    plan_id: UUID | None = None
    plan_step_id: str | None = Field(default=None, pattern=r"^step_[1-9][0-9]*$")
    status: ActionStatus


class CognitiveTask(BaseModel):
    """Executive-owned objective and assessment for the current task."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID = Field(default_factory=uuid4)
    objective: str = Field(min_length=1, max_length=400)
    assessment: TaskAssessment
    contract: TaskContract


class WorkingMemory(BaseModel):
    """Task-local mutable state with deterministic newest-item retention."""

    model_config = ConfigDict(extra="forbid")

    user_input: str = Field(min_length=1, max_length=MAX_USER_INPUT_CHARS)
    recent_history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_RECENT_HISTORY,
    )
    observations: list[ExecutionObservation | Observation] = Field(
        default_factory=list,
        max_length=MAX_OBSERVATIONS,
    )
    intermediate_results: list[IntermediateResult] = Field(
        default_factory=list,
        max_length=MAX_INTERMEDIATE_RESULTS,
    )
    action_executions: list[ActionExecutionRecord] = Field(
        default_factory=list,
        max_length=MAX_ACTION_EXECUTIONS,
    )
    # Retrieval is a bounded data channel for the current task. It is never
    # authority, instruction, or a substitute for a verifier.
    retrieved_knowledge: list[RetrievalResult] = Field(
        default_factory=list,
        max_length=MAX_RETRIEVED_KNOWLEDGE,
    )

    def add_retrieved_knowledge(self, items: list[RetrievalResult] | tuple[RetrievalResult, ...]) -> None:
        if not all(isinstance(item, RetrievalResult) for item in items):
            raise CognitiveRuntimeError("WorkingMemory requires typed retrieval results")
        self.retrieved_knowledge = list(items)[-MAX_RETRIEVED_KNOWLEDGE:]

    def add_observation(self, observation: ExecutionObservation | Observation) -> None:
        """Append an explicit observation, retaining only the newest items."""

        if not isinstance(observation, (ExecutionObservation, Observation)):
            raise CognitiveRuntimeError("WorkingMemory requires a validated Observation")
        self.observations.append(observation)
        del self.observations[:-MAX_OBSERVATIONS]

    def get_execution_observation(self, action_id: UUID) -> ExecutionObservation | None:
        """Return an existing typed observation for exactly one action, if any."""

        for observation in self.observations:
            if isinstance(observation, ExecutionObservation) and observation.action_id == action_id:
                return observation
        return None

    def add_intermediate_result(self, result: IntermediateResult) -> None:
        """Append an explicit artifact, retaining only the newest items."""

        if not isinstance(result, IntermediateResult):
            raise CognitiveRuntimeError(
                "WorkingMemory requires a validated IntermediateResult"
            )
        self.intermediate_results.append(result)
        del self.intermediate_results[:-MAX_INTERMEDIATE_RESULTS]

    def add_action_execution(self, record: ActionExecutionRecord) -> None:
        """Retain executor-produced result metadata for this task only."""

        if not isinstance(record, ActionExecutionRecord):
            raise CognitiveRuntimeError("WorkingMemory requires an action execution record")
        self.action_executions.append(record)
        del self.action_executions[:-MAX_ACTION_EXECUTIONS]


class CognitiveContext(BaseModel):
    """Complete bounded state supplied to the runtime and future planners."""

    model_config = ConfigDict(extra="forbid")

    task: CognitiveTask
    working_memory: WorkingMemory


def normalize_history(
    history: list[dict] | None,
    *,
    limit: int = MAX_RECENT_HISTORY,
) -> list[HistoryMessage]:
    """Filter raw history to valid messages and preserve the newest entries."""

    if not history or limit <= 0:
        return []

    messages: list[HistoryMessage] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        try:
            messages.append(HistoryMessage(role=role, content=content))
        except ValidationError:
            continue
    return messages[-limit:]


def create_cognitive_context(
    *,
    user_input: str,
    history: list[dict] | None,
    assessment: TaskAssessment,
) -> CognitiveContext:
    """Construct independent, bounded task state without model or tool calls."""

    if not isinstance(assessment, TaskAssessment):
        raise CognitiveRuntimeError("Cognitive context requires a validated TaskAssessment")

    context = CognitiveContext(
        task=CognitiveTask(
            objective=assessment.user_goal,
            assessment=assessment,
            contract=build_task_contract(user_input=user_input, assessment=assessment),
        ),
        working_memory=WorkingMemory(
            user_input=user_input,
            recent_history=normalize_history(history),
        ),
    )
    logger.info(
        "brain.context created task_type=%s complexity=%s history_messages=%d "
        "observations=%d intermediate_results=%d bounds=%d/%d/%d",
        assessment.task_type.value,
        assessment.complexity.value,
        len(context.working_memory.recent_history),
        len(context.working_memory.observations),
        len(context.working_memory.intermediate_results),
        MAX_RECENT_HISTORY,
        MAX_OBSERVATIONS,
        MAX_INTERMEDIATE_RESULTS,
    )
    return context
