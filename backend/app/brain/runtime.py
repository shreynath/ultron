"""Deterministic execution routing for Ultron's cognitive layer.

Step 2 consumes an already-validated Executive TaskAssessment. It either
invokes the existing direct-response generator or truthfully reports that the
request needs a future cognitive capability. It does not plan, execute tools,
retrieve memory, or verify results.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from enum import Enum
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionRequest, ActionResult
from app.brain.conversation_bridge import (
    CapabilitySelection,
    ConversationCapabilityBridge,
    ConversationCapabilityOutcome,
)
from app.brain.errors import CognitiveRuntimeError
from app.brain.evidence_providers import create_application_evidence_providers
from app.brain.executor import ActionExecutor
from app.brain.observation import ExecutionObservation, ObservationInterpreter
from app.brain.planner import CognitivePlanner, ExecutionPlan
from app.brain.plan_execution import PlanExecution, PlanExecutionCoordinator
from app.brain.schemas import TaskAssessment, TaskType
from app.brain.verification import (
    EvidenceCollection,
    EvidenceProviderRegistry,
    EvidenceVerifier,
    VerificationResult,
)
from app.brain.working_memory import CognitiveContext, create_cognitive_context
from app.brain.learning import KnowledgeScope, LearningRepository
from app.services.local_model_client import chat_response
from app.services.ollama_client import OllamaError

logger = logging.getLogger(__name__)

_LIVE_INFORMATION_REQUEST = re.compile(
    r"\b(?:current|currently|latest|today|now|most recent|live)\b", re.IGNORECASE
)


def require_live_research(assessment: TaskAssessment, user_input: str) -> TaskAssessment:
    """Conservative deterministic freshness guard for triage/provider fallback.

    A model/provider outage or an under-classified Executive result must not
    turn a request explicitly asking for current information into an ungrounded
    direct answer. This only adds the already-registered retrieval requirement;
    it never grants a new capability.
    """
    if not _LIVE_INFORMATION_REQUEST.search(user_input):
        return assessment
    if assessment.requires_tools and assessment.task_type in {TaskType.RETRIEVAL, TaskType.RESEARCH}:
        return assessment
    logger.info("brain.runtime live_research_required task_type=%s", assessment.task_type.value)
    return assessment.model_copy(update={
        "task_type": TaskType.RETRIEVAL,
        "requires_tools": True,
        "requires_verification": True,
        "direct_response_allowed": False,
        "rationale": "Explicit temporal language requires the registered live research capability.",
    })


class ExecutionRoute(str, Enum):
    """The only execution paths available to the Step 2 runtime."""

    DIRECT_RESPONSE = "direct_response"
    COGNITIVE_EXECUTION_REQUIRED = "cognitive_execution_required"


class ExecutionStatus(str, Enum):
    """Truthful completion state for one runtime invocation."""

    COMPLETED = "completed"
    REQUIRES_FUTURE_CAPABILITY = "requires_future_capability"
    FAILED = "failed"


class RuntimeTraceEvent(BaseModel):
    """A compact, user-safe proof of a runtime event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = Field(min_length=1, max_length=80)
    component: str = Field(min_length=1, max_length=100)
    status: str = Field(min_length=1, max_length=100)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action_id: UUID | None = None
    capability: str | None = Field(default=None, max_length=64)


class CognitiveExecution(BaseModel):
    """Typed result of routing and, when available, direct execution."""

    model_config = ConfigDict(extra="forbid")

    route: ExecutionRoute
    status: ExecutionStatus
    assessment: TaskAssessment
    answer: str = Field(min_length=1)
    required_capabilities: tuple[str, ...] = ()
    plan: ExecutionPlan | None = None
    capability_name: str | None = None
    capability_operation: str | None = None
    action_status: str | None = None
    observation_outcome: str | None = None
    response_generation_status: str | None = None
    verification: VerificationResult | None = None
    request_id: UUID | None = None
    trace: tuple[RuntimeTraceEvent, ...] = Field(default=(), max_length=16)


class ResponseGenerator(Protocol):
    """Existing text-generation contract used by the direct path."""

    async def __call__(
        self,
        *,
        user_message: str,
        system_prompt: str,
        history: list[dict[str, str]] | None,
    ) -> str: ...


def required_capabilities(assessment: TaskAssessment) -> tuple[str, ...]:
    """Return future capabilities required by an assessment in fixed order."""

    requirements: list[str] = []
    if assessment.requires_planning:
        requirements.append("planning")
    if assessment.requires_tools:
        requirements.append("tools")
    if assessment.requires_memory:
        requirements.append("memory")
    if assessment.requires_verification:
        requirements.append("verification")
    return tuple(requirements)


def route_assessment(assessment: TaskAssessment) -> ExecutionRoute:
    """Select a route using only validated Executive control fields.

    Explicit capability requirements take precedence over direct response.
    This prevents an unusual but capability-dependent assessment from being
    treated as completed without the required work. The Executive fallback
    carries no requirements and therefore remains eligible for direct reply.
    """

    if required_capabilities(assessment):
        return ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED
    if assessment.direct_response_allowed:
        return ExecutionRoute.DIRECT_RESPONSE
    return ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED


def _future_capability_message(
    requirements: tuple[str, ...],
    assessment: TaskAssessment,
) -> str:
    if assessment.task_type.value == "retrieval" and "tools" in requirements:
        return (
            "I can identify that this request needs live retrieval, but Ultron's "
            "current runtime does not have a matching retrieval capability connected yet."
        )
    names = ", ".join(requirements) if requirements else "additional cognitive execution"
    return (
        "I can't complete that request truthfully in this version of Ultron "
        f"because it requires {names}, which is not available yet."
    )


class CognitiveRuntime:
    """Route a validated assessment without reclassifying or performing future work."""

    def __init__(
        self,
        response_generator: ResponseGenerator = chat_response,
        planner: CognitivePlanner | None = None,
        action_executor: ActionExecutor | None = None,
        observation_interpreter: ObservationInterpreter | None = None,
        conversation_bridge: ConversationCapabilityBridge | None = None,
        evidence_verifier: EvidenceVerifier | None = None,
        evidence_providers: EvidenceProviderRegistry | None = None,
        learning_repository: LearningRepository | None = None,
    ) -> None:
        self._response_generator = response_generator
        self._planner = planner or CognitivePlanner()
        # The default executor has an empty allowlist and registry.  It is an
        # internal boundary, not an invitation for normal chat to run actions.
        self._action_executor = action_executor or ActionExecutor()
        self._observation_interpreter = observation_interpreter or ObservationInterpreter()
        self._conversation_bridge = conversation_bridge or ConversationCapabilityBridge()
        self._evidence_verifier = evidence_verifier or EvidenceVerifier()
        self._evidence_providers = evidence_providers or create_application_evidence_providers()
        self._learning_repository = learning_repository

    def create_context(
        self,
        *,
        user_input: str,
        history: list[dict] | None,
        assessment: TaskAssessment,
    ) -> CognitiveContext:
        """Build bounded task-local state for this runtime invocation."""

        return create_cognitive_context(
            user_input=user_input,
            history=history,
            assessment=require_live_research(assessment, user_input),
        )

    def attach_retrieved_knowledge(
        self,
        *,
        context: CognitiveContext,
        repository: LearningRepository | None = None,
    ) -> None:
        """Place only verified, typed learning data into task-local context.

        A repository is optional at the runtime seam so focused tests and
        embedders can remain entirely in-memory. Production coordination
        supplies the durable repository.
        """
        active_repository = repository or self._learning_repository
        if active_repository is None:
            return
        items = list(active_repository.retrieve(
            query=f"{context.task.objective} {context.working_memory.user_input}"
        ))
        items.extend(active_repository.retrieve(
            query=f"{context.task.objective} {context.working_memory.user_input}",
            scope=KnowledgeScope.WORLD,
            limit=3,
        ))
        context.working_memory.add_retrieved_knowledge(items)

    async def execute_action(
        self,
        *,
        context: CognitiveContext,
        action: ActionRequest,
    ) -> ActionResult:
        """Explicitly dispatch one action; never iterate or re-plan a plan."""

        if not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Runtime requires a validated CognitiveContext")
        return await self._action_executor.execute(context=context, action=action)

    async def execute_plan(
        self,
        *,
        context: CognitiveContext,
        plan: ExecutionPlan,
        actions: dict[str, ActionRequest],
    ) -> PlanExecution:
        """Run an application-built multi-step graph through existing authority.

        This intentionally requires explicit ActionRequests; planner output is
        descriptive and cannot choose capability names, operations, or inputs.
        """
        return await PlanExecutionCoordinator(
            executor=self._action_executor,
            observer=self._observation_interpreter,
        ).execute(context=context, plan=plan, actions=actions)

    def observe_action_result(
        self,
        *,
        context: CognitiveContext,
        action_result: ActionResult,
    ) -> ExecutionObservation:
        """Explicitly record one result; this cannot advance execution."""

        if not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Runtime requires a validated CognitiveContext")
        return self._observation_interpreter.observe(
            context=context,
            action_result=action_result,
        )

    def select_conversation_capability(
        self, *, context: CognitiveContext
    ) -> CapabilitySelection | None:
        """Expose the existing application-owned capability selection boundary."""

        if not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Runtime requires a validated CognitiveContext")
        return self._conversation_bridge.select_for_context(context)

    async def execute_conversation_capability(
        self,
        *,
        context: CognitiveContext,
        selection: CapabilitySelection,
    ) -> ConversationCapabilityOutcome:
        """Dispatch one selected conversation action through Steps 5 and 6."""

        if not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Runtime requires a validated CognitiveContext")
        return await self._conversation_bridge.execute_selection(
            context=context,
            selection=selection,
        )

    def conversation_capability_response(
        self, outcome: ConversationCapabilityOutcome
    ) -> str:
        """Render the bridge's deterministic response from an actual action result."""

        return self._conversation_bridge.response_for(outcome)

    async def handle(
        self,
        *,
        user_input: str | None = None,
        history: list[dict] | None = None,
        assessment: TaskAssessment | None = None,
        context: CognitiveContext | None = None,
        system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    ) -> CognitiveExecution:
        """Route one already-assessed request and execute only a direct reply.

        The runtime intentionally accepts an assessment instead of an
        Executive dependency. This makes the assessment/execution boundary
        explicit and guarantees routing never triggers another model call.
        """

        if context is None:
            if not isinstance(assessment, TaskAssessment) or user_input is None:
                raise CognitiveRuntimeError(
                    "Runtime requires a validated TaskAssessment and user input"
                )
            context = self.create_context(
                user_input=user_input,
                history=history,
                assessment=assessment,
            )
        elif not isinstance(context, CognitiveContext):
            raise CognitiveRuntimeError("Runtime requires a validated CognitiveContext")

        assessment = context.task.assessment
        user_input = context.working_memory.user_input
        history = [
            message.model_dump()
            for message in context.working_memory.recent_history
        ]

        started = time.perf_counter()
        route = route_assessment(assessment)
        requirements = required_capabilities(assessment)
        logger.info(
            "brain.runtime start route=%s task_type=%s complexity=%s tools=%s "
            "memory=%s planning=%s verification=%s direct=%s fallback=%s",
            route.value,
            assessment.task_type.value,
            assessment.complexity.value,
            assessment.requires_tools,
            assessment.requires_memory,
            assessment.requires_planning,
            assessment.requires_verification,
            assessment.direct_response_allowed,
            assessment.used_fallback,
        )

        if route is ExecutionRoute.COGNITIVE_EXECUTION_REQUIRED:
            plan = None
            if assessment.requires_planning:
                plan = await self._planner.create_plan(context)
            execution = CognitiveExecution(
                route=route,
                status=ExecutionStatus.REQUIRES_FUTURE_CAPABILITY,
                assessment=assessment,
                answer=_future_capability_message(requirements, assessment),
                required_capabilities=requirements,
                plan=plan,
            )
        else:
            try:
                answer = await self._response_generator(
                    user_message=user_input,
                    system_prompt=system_prompt,
                    history=history or [],
                )
            except OllamaError:
                logger.warning("brain.runtime direct_response_failed kind=ollama")
                execution = CognitiveExecution(
                    route=route,
                    status=ExecutionStatus.FAILED,
                    assessment=assessment,
                    answer="I encountered an issue generating a response. Please try again.",
                )
            except Exception:  # noqa: BLE001 - response adapters are externally supplied
                logger.error("brain.runtime direct_response_failed kind=unexpected")
                execution = CognitiveExecution(
                    route=route,
                    status=ExecutionStatus.FAILED,
                    assessment=assessment,
                    answer="I encountered an issue generating a response. Please try again.",
                )
            else:
                execution = CognitiveExecution(
                    route=route,
                    status=ExecutionStatus.COMPLETED,
                    assessment=assessment,
                    answer=answer,
                )

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "brain.runtime complete route=%s status=%s required_capabilities=%s "
            "fallback=%s latency_ms=%.1f",
            execution.route.value,
            execution.status.value,
            list(execution.required_capabilities),
            assessment.used_fallback,
            latency_ms,
        )
        return execution

    async def handle_conversation(
        self,
        *,
        user_input: str,
        history: list[dict] | None,
        assessment: TaskAssessment,
        system_prompt: str = "You are a helpful assistant. Keep replies concise and conversational.",
    ) -> CognitiveExecution:
        """Run the public conversation seam through the Step 8 coordinator.

        The local import prevents a module-import cycle: Step 8 composes this
        runtime, while this compatibility entry point remains the stable
        surface used by existing routers and diagnostic callers.
        """

        from app.brain.coordinator import BoundedExecutionCoordinator

        outcome = await BoundedExecutionCoordinator(runtime=self).run(
            user_input=user_input,
            history=history,
            assessment=assessment,
            system_prompt=system_prompt,
        )
        return outcome.execution

    async def complete_capability_execution(
        self,
        initial: CognitiveExecution,
        outcome: ConversationCapabilityOutcome,
        *,
        user_input: str,
        history: list[dict[str, str]],
        system_prompt: str,
        on_response_generation: Callable[[], None] | None = None,
        on_verification: Callable[[], None] | None = None,
    ) -> CognitiveExecution:
        """Finish one completed bridge action with response and evidence checks.

        This is deliberately limited to an action result already produced by
        the Step 5/6 boundary.  It neither selects nor executes an action,
        which permits Step 8 to expose the execution lifecycle without
        duplicating response or verification behavior.
        """

        result = outcome.action_result
        response_generation_status: str | None = None
        answer = self._conversation_bridge.response_for(outcome)
        verification: VerificationResult | None = None
        if result.status.value == "completed":
            # A capability result is the execution authority.  Do not ask a
            # general-purpose model to restate it: even a well-prompted model
            # could introduce an unsupported retrieval, computation, source,
            # or tool-use claim.  Registered capability formatters may use
            # only the schema-validated ActionResult produced in this turn.
            if on_response_generation is not None:
                on_response_generation()
            response_generation_status = "grounded_capability"
            provider = self._evidence_providers.get(result.capability, result.operation)
            if on_verification is not None:
                on_verification()
            if provider is None:
                verification = self._evidence_verifier.verify(
                    task_id=result.task_id, candidate_answer=answer, claims=(),
                    evidence=EvidenceCollection(task_id=result.task_id, items=()),
                )
            else:
                try:
                    evidence = provider.produce_evidence(task_id=result.task_id, action_result=result)
                    verification = self._evidence_verifier.verify(
                        task_id=result.task_id, candidate_answer=answer,
                        claims=provider.claims(result), evidence=evidence,
                    )
                except (TypeError, ValueError):
                    verification = self._evidence_verifier.verify(
                        task_id=result.task_id, candidate_answer=answer, claims=(),
                        evidence=EvidenceCollection(task_id=result.task_id, items=()),
                    )
            if verification.status.value == "contradicted":
                # Provider evidence wins over model prose.  This is a single
                # deterministic correction, not regeneration or a retry loop.
                answer = provider.fallback_answer(result) if provider is not None else self._conversation_bridge.response_for(outcome)
                response_generation_status = "corrected_from_evidence"
        remaining_requirements = tuple(
            requirement
            for requirement in initial.required_capabilities
            if requirement != "tools"
        )
        return CognitiveExecution(
            route=initial.route,
            status=(
                ExecutionStatus.COMPLETED
                if result.status.value == "completed"
                else ExecutionStatus.FAILED
            ),
            assessment=initial.assessment,
            answer=answer,
            required_capabilities=remaining_requirements,
            plan=initial.plan,
            capability_name=result.capability,
            capability_operation=result.operation,
            action_status=result.status.value,
            observation_outcome=outcome.observation.outcome.value,
            response_generation_status=response_generation_status,
            verification=verification,
        )
