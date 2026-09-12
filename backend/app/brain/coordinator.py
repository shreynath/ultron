"""Step 8 bounded execution coordinator.

The coordinator owns only the lifecycle around an already assessed task.  It
composes the existing runtime, single-action bridge, and evidence verifier; it
does not classify, plan, authorize, execute arbitrary operations, retrieve
memory, or implement verification itself.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID

import anyio
from pydantic import BaseModel, ConfigDict, Field

from app.brain.cognitive_state import CognitiveStage, CognitiveStateStore
from app.brain.conversation_bridge import ConversationCapabilityOutcome
from app.brain.errors import CognitiveRuntimeError
from app.brain.learning import (
    LearningEvent,
    LearningRepository,
    experience_from_execution,
)
from app.brain.runtime import (
    CognitiveExecution,
    CognitiveRuntime,
    ExecutionRoute,
    ExecutionStatus,
    RuntimeTraceEvent,
    route_assessment,
)
from app.brain.schemas import TaskAssessment
from app.brain.verification import VerificationStatus
from app.brain.working_memory import CognitiveContext
from app.config import settings

logger = logging.getLogger(__name__)


class OrchestrationStage(str, Enum):
    """Observable lifecycle states for one bounded Step 8 run."""

    RECEIVED = "received"
    ASSESSED = "assessed"
    DISPATCHED = "dispatched"
    EXECUTING = "executing"
    RESPONDING = "responding"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


_TERMINAL_STAGES = frozenset({
    OrchestrationStage.COMPLETED,
    OrchestrationStage.DEGRADED,
    OrchestrationStage.FAILED,
    OrchestrationStage.CANCELLED,
    OrchestrationStage.BLOCKED,
})
_LEGAL_TRANSITIONS: dict[OrchestrationStage, frozenset[OrchestrationStage]] = {
    OrchestrationStage.RECEIVED: frozenset({
        OrchestrationStage.ASSESSED, OrchestrationStage.FAILED, OrchestrationStage.CANCELLED,
    }),
    OrchestrationStage.ASSESSED: frozenset({
        OrchestrationStage.DISPATCHED, OrchestrationStage.BLOCKED,
        OrchestrationStage.FAILED, OrchestrationStage.CANCELLED,
    }),
    OrchestrationStage.DISPATCHED: frozenset({
        OrchestrationStage.EXECUTING, OrchestrationStage.RESPONDING, OrchestrationStage.COMPLETED,
        OrchestrationStage.FAILED, OrchestrationStage.BLOCKED, OrchestrationStage.CANCELLED,
    }),
    OrchestrationStage.EXECUTING: frozenset({
        OrchestrationStage.RESPONDING, OrchestrationStage.FAILED, OrchestrationStage.CANCELLED,
    }),
    OrchestrationStage.RESPONDING: frozenset({
        OrchestrationStage.VERIFYING, OrchestrationStage.COMPLETED,
        OrchestrationStage.DEGRADED, OrchestrationStage.FAILED, OrchestrationStage.CANCELLED,
    }),
    OrchestrationStage.VERIFYING: frozenset({
        OrchestrationStage.COMPLETED, OrchestrationStage.DEGRADED,
        OrchestrationStage.FAILED, OrchestrationStage.CANCELLED,
    }),
}


class LifecycleTransitionError(CognitiveRuntimeError):
    """Raised when a caller attempts to move a completed task backwards."""


class OrchestrationTransition(BaseModel):
    """One factual lifecycle event; no prompts, secrets, or raw output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    from_stage: OrchestrationStage | None = None
    to_stage: OrchestrationStage
    component: str = Field(min_length=1, max_length=100)
    status: str = Field(min_length=1, max_length=100)
    reason: str | None = Field(default=None, max_length=500)
    occurred_at: datetime


class CoordinatorExecution(BaseModel):
    """Result of coordinating one task through the established brain layers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: UUID
    lifecycle_stage: OrchestrationStage
    execution: CognitiveExecution
    transitions: tuple[OrchestrationTransition, ...] = Field(min_length=1, max_length=12)
    degradation_reasons: tuple[str, ...] = Field(max_length=4)
    learning: LearningEvent | None = None


class ExecutionLifecycle:
    """Per-run state machine; intentionally not global or persistent."""

    def __init__(self, task_id: UUID, *, persist_transition=None) -> None:
        self.task_id = task_id
        self._persist_transition = persist_transition
        self._stage: OrchestrationStage | None = None
        self._transitions: list[OrchestrationTransition] = []

    @property
    def stage(self) -> OrchestrationStage | None:
        return self._stage

    @property
    def transitions(self) -> tuple[OrchestrationTransition, ...]:
        return tuple(self._transitions)

    def transition(
        self,
        to_stage: OrchestrationStage,
        *,
        component: str,
        status: str,
        reason: str | None = None,
    ) -> None:
        if self._stage is None:
            if to_stage is not OrchestrationStage.RECEIVED:
                raise LifecycleTransitionError("The first orchestration stage must be received")
        elif self._stage in _TERMINAL_STAGES or to_stage not in _LEGAL_TRANSITIONS.get(self._stage, frozenset()):
            raise LifecycleTransitionError(
                f"Illegal orchestration transition: {self._stage.value} -> {to_stage.value}"
            )
        previous = self._stage
        self._stage = to_stage
        event = OrchestrationTransition(
            sequence=len(self._transitions) + 1,
            from_stage=previous,
            to_stage=to_stage,
            component=component,
            status=status,
            reason=reason,
            occurred_at=datetime.now(UTC),
        )
        self._transitions.append(event)
        if self._persist_transition:
            self._persist_transition(event)
        logger.info(
            "brain.coordinator task_id=%s stage=%s transition=%s component=%s status=%s reason=%s",
            self.task_id,
            to_stage.value,
            f"{previous.value if previous else 'none'}->{to_stage.value}",
            component,
            status,
            reason,
        )


class BoundedExecutionCoordinator:
    """Coordinate at most one registered action after deterministic routing.

    A run creates no durable state and never retries.  Its task identity is
    the existing ``CognitiveContext.task.task_id``; no parallel request or
    trace identifier is introduced.
    """

    def __init__(
        self,
        *,
        runtime: CognitiveRuntime | None = None,
        learning_repository: LearningRepository | None = None,
        cognitive_state_store: CognitiveStateStore | None = None,
    ) -> None:
        self._runtime = runtime or CognitiveRuntime()
        self._learning_repository = learning_repository or LearningRepository(settings.brain_learning_db)
        self._cognitive_state_store = cognitive_state_store or CognitiveStateStore(
            self._learning_repository.path.with_name("cognitive_state.sqlite3")
        )

    @property
    def learning_repository(self) -> LearningRepository:
        """Read-only diagnostic access; it never exposes execution authority."""
        return self._learning_repository

    def create_context(
        self,
        *,
        user_input: str,
        history: list[dict] | None,
        assessment: TaskAssessment,
    ) -> CognitiveContext:
        return self._runtime.create_context(
            user_input=user_input,
            history=history,
            assessment=assessment,
        )

    async def run(
        self,
        *,
        user_input: str | None = None,
        history: list[dict] | None = None,
        assessment: TaskAssessment | None = None,
        context: CognitiveContext | None = None,
        system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    ) -> CoordinatorExecution:
        """Run one assessed task, preserving every real terminal condition."""

        if context is None:
            if user_input is None or not isinstance(assessment, TaskAssessment):
                raise CognitiveRuntimeError(
                    "Coordinator requires a validated TaskAssessment and user input"
                )
            context = self.create_context(
                user_input=user_input,
                history=history,
                assessment=assessment,
            )
        elif not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Coordinator requires a validated CognitiveContext")

        try:
            self._cognitive_state_store.create(
                context.task.objective, task_id=context.task.task_id
            )
        except Exception:
            # Task execution and authorization must not become unavailable if
            # an optional observability write fails.
            logger.exception("brain.coordinator cognitive_state_create_failed task_id=%s", context.task.task_id)

        def persist_transition(event: OrchestrationTransition) -> None:
            stages = {
                OrchestrationStage.RECEIVED: CognitiveStage.UNDERSTANDING,
                OrchestrationStage.ASSESSED: CognitiveStage.UNDERSTANDING,
                OrchestrationStage.DISPATCHED: CognitiveStage.PLANNING,
                OrchestrationStage.EXECUTING: CognitiveStage.EXECUTING,
                OrchestrationStage.RESPONDING: CognitiveStage.RESPONDING,
                OrchestrationStage.VERIFYING: CognitiveStage.VERIFYING,
                OrchestrationStage.COMPLETED: CognitiveStage.IDLE,
                OrchestrationStage.DEGRADED: CognitiveStage.IDLE,
                OrchestrationStage.BLOCKED: CognitiveStage.FAILED,
                OrchestrationStage.FAILED: CognitiveStage.FAILED,
                OrchestrationStage.CANCELLED: CognitiveStage.FAILED,
            }
            try:
                self._cognitive_state_store.transition(
                    context.task.task_id,
                    stages[event.to_stage],
                    component=event.component,
                    summary=event.status if event.reason is None else f"{event.status}: {event.reason}",
                )
            except (KeyError, ValueError, OSError):
                logger.exception("brain.coordinator cognitive_state_transition_failed task_id=%s", context.task.task_id)

        lifecycle = ExecutionLifecycle(context.task.task_id, persist_transition=persist_transition)
        degradation_reasons: list[str] = []
        lifecycle.transition(OrchestrationStage.RECEIVED, component="coordinator", status="received")
        lifecycle.transition(OrchestrationStage.ASSESSED, component="executive", status="assessed")
        self._runtime.attach_retrieved_knowledge(
            context=context,
            repository=self._learning_repository,
        )

        try:
            lifecycle.transition(OrchestrationStage.DISPATCHED, component="cognitive_runtime", status="dispatched")
            if route_assessment(context.task.assessment) is ExecutionRoute.DIRECT_RESPONSE:
                lifecycle.transition(
                    OrchestrationStage.RESPONDING,
                    component="response_generator",
                    status="started",
                )
            initial = await self._runtime.handle(context=context, system_prompt=system_prompt)

            if initial.route is ExecutionRoute.DIRECT_RESPONSE:
                terminal = (
                    OrchestrationStage.COMPLETED
                    if initial.status is ExecutionStatus.COMPLETED
                    else OrchestrationStage.FAILED
                )
                lifecycle.transition(terminal, component="cognitive_runtime", status=initial.status.value)
                return await self._finalize(lifecycle, initial, degradation_reasons, context=context)

            selection = self._runtime.select_conversation_capability(context=context)
            if selection is None:
                lifecycle.transition(
                    OrchestrationStage.BLOCKED,
                    component="conversation_capability_bridge",
                    status="unavailable",
                    reason="No registered capability supports this assessed request.",
                )
                return await self._finalize(lifecycle, initial, degradation_reasons, context=context)

            lifecycle.transition(
                OrchestrationStage.EXECUTING,
                component=selection.descriptor.name,
                status="started",
            )
            outcome = await self._runtime.execute_conversation_capability(
                context=context,
                selection=selection,
            )
            action_status = outcome.action_result.status.value
            if action_status != "completed":
                failed = self._capability_failure(initial, outcome)
                lifecycle.transition(
                    OrchestrationStage.FAILED,
                    component=outcome.action_result.capability,
                    status=action_status,
                    reason=outcome.action_result.error.code.value if outcome.action_result.error else None,
                )
                return await self._finalize(
                    lifecycle, failed, degradation_reasons, context=context, outcome=outcome,
                )

            def responding() -> None:
                lifecycle.transition(
                    OrchestrationStage.RESPONDING,
                    component="response_generator",
                    status="started",
                )

            def verifying() -> None:
                lifecycle.transition(
                    OrchestrationStage.VERIFYING,
                    component="evidence_verifier",
                    status="started",
                )

            execution = await self._runtime.complete_capability_execution(
                initial,
                outcome,
                user_input=context.working_memory.user_input,
                history=[message.model_dump() for message in context.working_memory.recent_history],
                system_prompt=system_prompt,
                on_response_generation=responding,
                on_verification=verifying,
            )
            degradation_reasons.extend(_degradation_reasons(execution))
            terminal = (
                OrchestrationStage.DEGRADED
                if degradation_reasons
                else OrchestrationStage.COMPLETED
            )
            lifecycle.transition(
                terminal,
                component="coordinator",
                status=execution.status.value,
                reason=degradation_reasons[0] if degradation_reasons else None,
            )
            return await self._finalize(
                lifecycle, execution, degradation_reasons, context=context, outcome=outcome,
            )
        except BaseException as exc:
            if isinstance(exc, anyio.get_cancelled_exc_class()):
                if lifecycle.stage not in _TERMINAL_STAGES:
                    lifecycle.transition(
                        OrchestrationStage.CANCELLED,
                        component="coordinator",
                        status="cancelled",
                        reason="The request was cancelled before completion.",
                    )
                raise
            if not isinstance(exc, Exception):
                raise
            if lifecycle.stage not in _TERMINAL_STAGES:
                lifecycle.transition(
                    OrchestrationStage.FAILED,
                    component="coordinator",
                    status="internal_error",
                    reason="An unexpected internal coordination error occurred.",
                )
            raise

    def _capability_failure(
        self,
        initial: CognitiveExecution,
        outcome: ConversationCapabilityOutcome,
    ) -> CognitiveExecution:
        """Retain the bridge's factual user response without treating it as success."""

        result = outcome.action_result
        return CognitiveExecution(
            route=initial.route,
            status=ExecutionStatus.FAILED,
            assessment=initial.assessment,
            answer=self._runtime.conversation_capability_response(outcome),
            required_capabilities=initial.required_capabilities,
            plan=initial.plan,
            capability_name=result.capability,
            capability_operation=result.operation,
            action_status=result.status.value,
            observation_outcome=outcome.observation.outcome.value,
        )

    async def _finalize(
        self,
        lifecycle: ExecutionLifecycle,
        execution: CognitiveExecution,
        degradation_reasons: Iterable[str],
        *,
        context: CognitiveContext,
        outcome: ConversationCapabilityOutcome | None = None,
    ) -> CoordinatorExecution:
        """Commit a sanitized learning event after the cognitive outcome terminates.

        Persistence is intentionally post-execution and cannot alter the already
        decided lifecycle, action result, authorization, or verification result.
        """
        assert lifecycle.stage is not None
        learning: LearningEvent | None = None
        try:
            experience = experience_from_execution(
                correlation_id=lifecycle.task_id,
                assessment=context.task.assessment,
                execution=execution,
                lifecycle_outcome=lifecycle.stage.value,
                observation=outcome.observation if outcome else None,
                action_result=outcome.action_result if outcome else None,
                failure_reason=(
                    next((transition.reason for transition in reversed(lifecycle.transitions) if transition.reason), None)
                ),
            )
            experience = self._learning_repository.commit_experience(experience)
            learning = self._learning_repository.promote(experience)
            logger.info(
                "brain.coordinator learning task_id=%s experience_id=%s status=%s knowledge_ids=%s",
                lifecycle.task_id,
                experience.experience_id,
                learning.status.value,
                [str(item) for item in learning.knowledge_ids],
            )
        except Exception:  # Learning must not falsify or reverse a completed execution.
            logger.exception("brain.coordinator learning_commit_failed task_id=%s", lifecycle.task_id)
        trace = [
            RuntimeTraceEvent(
                event=_trace_event_name(transition.to_stage),
                component=transition.component,
                status=transition.status,
                occurred_at=transition.occurred_at,
            )
            for transition in lifecycle.transitions
        ]
        if outcome is not None:
            completed = RuntimeTraceEvent(
                event="CAPABILITY_COMPLETED",
                component=outcome.action_result.capability,
                capability=outcome.action_result.capability,
                action_id=outcome.action_result.action_id,
                status=outcome.action_result.status.value,
            )
            insert_at = next(
                (index + 1 for index, event in enumerate(trace)
                 if event.event == "CAPABILITY_STARTED"),
                len(trace),
            )
            proof = _capability_proof_events(outcome)
            trace[insert_at:insert_at] = proof + [completed]
        execution = execution.model_copy(update={
            "request_id": lifecycle.task_id,
            "trace": tuple(trace),
        })
        return CoordinatorExecution(
            task_id=lifecycle.task_id,
            lifecycle_stage=lifecycle.stage,
            execution=execution,
            transitions=lifecycle.transitions,
            degradation_reasons=tuple(degradation_reasons),
            learning=learning,
        )


def _degradation_reasons(execution: CognitiveExecution) -> list[str]:
    reasons: list[str] = []
    if execution.response_generation_status in {"fallback", "corrected_from_evidence"}:
        reasons.append(f"response_generation_{execution.response_generation_status}")
    if execution.verification is not None and execution.verification.status not in {
        VerificationStatus.VERIFIED,
        VerificationStatus.NOT_REQUIRED,
    }:
        reasons.append(f"verification_{execution.verification.status.value}")
    return reasons


def _trace_event_name(stage: OrchestrationStage) -> str:
    return {
        OrchestrationStage.RECEIVED: "REQUEST_RECEIVED",
        OrchestrationStage.ASSESSED: "EXECUTIVE_ASSESSMENT_COMPLETED",
        OrchestrationStage.DISPATCHED: "ACTION_DISPATCHED",
        OrchestrationStage.EXECUTING: "CAPABILITY_STARTED",
        OrchestrationStage.RESPONDING: "RESPONSE_GENERATION_STARTED",
        OrchestrationStage.VERIFYING: "VERIFICATION_STARTED",
        OrchestrationStage.COMPLETED: "REQUEST_COMPLETED",
        OrchestrationStage.DEGRADED: "REQUEST_DEGRADED",
        OrchestrationStage.FAILED: "REQUEST_FAILED",
        OrchestrationStage.CANCELLED: "REQUEST_CANCELLED",
        OrchestrationStage.BLOCKED: "REQUEST_BLOCKED",
    }[stage]


def _capability_proof_events(outcome: ConversationCapabilityOutcome) -> list[RuntimeTraceEvent]:
    """Expose only capability-owned execution facts, never model assertions."""
    result = outcome.action_result
    if result.capability != "document_analysis" or not isinstance(result.output, dict):
        return []
    proof = result.output.get("execution_proof")
    if not isinstance(proof, (tuple, list)):
        return []
    allowed = {
        "CORPUS_RESOLVED",
        "SOURCE_RETRIEVED",
        "TEXT_EXTRACTED",
        "TOKENS_COMPUTED",
        "DETERMINISTIC_INDEX_COMPLETED",
    }
    return [
        RuntimeTraceEvent(
            event=event,
            component="document_analysis",
            capability=result.capability,
            action_id=result.action_id,
            status="completed",
        )
        for event in proof
        if isinstance(event, str) and event in allowed
    ]
