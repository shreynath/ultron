"""Bounded bridge from a capability-required conversation to one action.

This is deliberately not an agent loop.  It selects at most one explicitly
registered action, executes it once, records one observation, and stops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.brain.actions import ActionRequest, ActionResult, ActionStatus
from app.brain.conversation_capabilities import (
    CapabilityDescriptor,
    ConversationAction,
    ConversationCapabilityRegistry,
)
from app.brain.document_analysis import DocumentAnalysisCapability
from app.brain.executor import ActionExecutor
from app.brain.observation import ExecutionObservation, ObservationInterpreter
from app.brain.reference_capability import ReferenceCapability
from app.brain.research_capability import ResearchCapability
from app.brain.schemas import TaskAssessment, TaskType
from app.brain.weather import WeatherCapability
from app.brain.working_memory import CognitiveContext

_WEATHER_FOLLOW_UP = re.compile(
    r"^\s*(?:what|how)\s+about\s+(?:the\s+)?(?:weather|temperature|forecast)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CapabilitySelection:
    descriptor: CapabilityDescriptor
    action: ConversationAction


@dataclass(frozen=True)
class ConversationCapabilityOutcome:
    selection: CapabilitySelection
    action_result: ActionResult
    observation: ExecutionObservation


class ConversationCapabilityBridge:
    """Coordinate a single action without reclassification, retries, or loops."""

    def __init__(
        self,
        *,
        registry: ConversationCapabilityRegistry | None = None,
        executor: ActionExecutor | None = None,
        observer: ObservationInterpreter | None = None,
    ) -> None:
        self._registry = registry or ConversationCapabilityRegistry(
            (WeatherCapability(), ReferenceCapability(), DocumentAnalysisCapability(), ResearchCapability())
        )
        self._executor = executor or ActionExecutor(
            registry=self._registry.executor_registry,
            allowed_capabilities=self._registry.names(),
        )
        self._observer = observer or ObservationInterpreter()

    def select(
        self,
        *,
        user_input: str,
        assessment: TaskAssessment,
    ) -> CapabilitySelection | None:
        """Select only a capability that application code explicitly registered."""

        if not assessment.requires_tools:
            return None
        for capability in self._registry.capabilities():
            descriptor = capability.descriptor
            if assessment.task_type not in descriptor.supported_task_types:
                continue
            action = capability.build_conversation_action(
                user_input=user_input,
                assessment=assessment,
            )
            if action is None:
                continue
            if action.operation not in descriptor.operations:
                continue
            return CapabilitySelection(descriptor=descriptor, action=action)
        return None

    def select_for_context(self, context: CognitiveContext) -> CapabilitySelection | None:
        """Select one application-owned action for an already-created task.

        Keeping follow-up resolution here gives higher-level coordinators a
        truthful selection boundary without making them parse conversation
        text or duplicate capability policy.
        """

        # A narrowly recognized weather follow-up such as "what about the
        # temperature?" has no location of its own. Reuse only an explicit
        # prior user location to perform one *new* lookup; an unrelated request
        # must never be interpreted as a follow-up or inherit a prior action.
        selection = None
        if _WEATHER_FOLLOW_UP.fullmatch(context.working_memory.user_input):
            for message in reversed(context.working_memory.recent_history):
                if message.role != "user":
                    continue
                selection = self.select(
                    user_input=message.content,
                    assessment=context.task.assessment,
                )
                if selection is not None:
                    break
        if selection is None:
            selection = self.select(
                user_input=context.working_memory.user_input,
                assessment=context.task.assessment,
            )
        # A corpus selection naturally follows an earlier exact-position
        # question. Carry that earlier target as untrusted request data; only
        # the registered deterministic capability can interpret it.
        if selection is None and re.search(r"\b(?:king\s+james|kjv)\b", context.working_memory.user_input, re.I):
            for message in reversed(context.working_memory.recent_history):
                if message.role != "user":
                    continue
                if re.search(r"\b\d+(?:st|nd|rd|th)\s+word\b|\bwords?\s+\d+\s+(?:through|to|-)\s+\d+\b", message.content, re.I):
                    selection = self.select(
                        user_input=f"{message.content}\nCorpus selection: {context.working_memory.user_input}",
                        assessment=context.task.assessment.model_copy(update={
                            "requires_tools": True,
                            "task_type": TaskType.RETRIEVAL,
                            "direct_response_allowed": False,
                        }),
                    )
                    break
        return selection

    async def execute(
        self,
        *,
        context: CognitiveContext,
    ) -> ConversationCapabilityOutcome | None:
        selection = self.select_for_context(context)
        if selection is None:
            return None
        return await self.execute_selection(context=context, selection=selection)

    async def execute_selection(
        self,
        *,
        context: CognitiveContext,
        selection: CapabilitySelection,
    ) -> ConversationCapabilityOutcome:
        """Execute exactly one previously selected, registered action once."""

        request = ActionRequest(
            task_id=context.task.task_id,
            capability=selection.descriptor.name,
            operation=selection.action.operation,
            parameters=selection.action.parameters,
        )
        result = await self._executor.execute(context=context, action=request)
        observation = self._observer.observe(context=context, action_result=result)
        return ConversationCapabilityOutcome(selection, result, observation)

    def response_for(self, outcome: ConversationCapabilityOutcome) -> str:
        """Produce a concise, truthful response from the actual action result."""

        capability = self._registry.get(outcome.selection.descriptor.name)
        if outcome.action_result.status is ActionStatus.COMPLETED and capability is not None:
            rendered = capability.format_conversation_result(outcome.action_result)
            if rendered:
                return rendered
            return "I completed the requested lookup, but I couldn't safely format its result."
        if outcome.action_result.status is ActionStatus.TIMED_OUT:
            return "I couldn't complete that lookup before the provider timed out."
        if outcome.action_result.status is ActionStatus.UNSUPPORTED:
            return "I can't complete that request because the selected capability is not supported."
        if outcome.action_result.status is ActionStatus.REJECTED:
            return "I can't complete that request because the selected capability is not permitted."
        return "I couldn't complete that lookup right now."
