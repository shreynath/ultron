"""Application-owned capability metadata for the conversational bridge.

These interfaces make capability selection explicit.  A conversation request
never supplies a capability name, operation, or callable: registered
application code decides whether it can safely construct one bounded action.
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult
from app.brain.capabilities import Capability, CapabilityRegistry
from app.brain.schemas import TaskAssessment, TaskType


class CapabilityDescriptor(BaseModel):
    """Static, application-controlled metadata used for safe selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    operations: tuple[str, ...] = Field(min_length=1)
    supported_task_types: tuple[TaskType, ...] = Field(min_length=1)
    single_action_only: bool = True


class ConversationAction(BaseModel):
    """The operation a registered capability has selected for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    parameters: dict[str, Any] = Field(default_factory=dict, max_length=32)


class ConversationCapability(Capability, Protocol):
    """A normal capability with deterministic conversational affordances."""

    descriptor: CapabilityDescriptor

    def build_conversation_action(
        self,
        *,
        user_input: str,
        assessment: TaskAssessment,
    ) -> ConversationAction | None:
        """Return one safe action when this capability supports the request."""

    def format_conversation_result(self, result: ActionResult) -> str | None:
        """Return a result-grounded response, or ``None`` when unavailable."""


class ConversationCapabilityRegistry:
    """Immutable-by-construction conversational view over capability instances."""

    def __init__(self, capabilities: tuple[ConversationCapability, ...] = ()) -> None:
        self._capabilities: dict[str, ConversationCapability] = {}
        for capability in capabilities:
            self.register(capability)
        self._executor_registry = CapabilityRegistry(tuple(self._capabilities.values()))

    def register(self, capability: ConversationCapability) -> None:
        descriptor = getattr(capability, "descriptor", None)
        if not isinstance(descriptor, CapabilityDescriptor):
            raise TypeError("Conversation capabilities require a CapabilityDescriptor")
        if descriptor.name != getattr(capability, "name", None):
            raise ValueError("Capability descriptor name must match the capability name")
        if descriptor.name in self._capabilities:
            raise ValueError(f"Capability already registered: {descriptor.name}")
        required = ("supports", "parameter_model", "execute", "build_conversation_action", "format_conversation_result")
        if not all(callable(getattr(capability, attribute, None)) for attribute in required):
            raise TypeError("Capability does not implement the conversation capability interface")
        self._capabilities[descriptor.name] = capability

    @property
    def executor_registry(self) -> CapabilityRegistry:
        return self._executor_registry

    def get(self, name: str) -> ConversationCapability | None:
        return self._capabilities.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def capabilities(self) -> tuple[ConversationCapability, ...]:
        return tuple(self._capabilities[name] for name in self.names())
