"""Explicit capability interface and deterministic registry.

This module deliberately contains no dynamic imports, callable lookup, or
registration from model/user data.  Each registry is an ordinary request-safe
application object supplied through dependency injection.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from app.brain.actions import CapabilityResult


class Capability(Protocol):
    """An application-owned implementation of a small named operation set."""

    name: str

    def supports(self, operation: str) -> bool:
        """Whether this exact, application-defined operation exists."""

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        """Return the strict input schema for an operation, if supported."""

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        """Perform exactly one already-validated operation."""


class CapabilityRegistry:
    """A small deterministic registry with no process-global mutable state."""

    def __init__(self, capabilities: tuple[Capability, ...] = ()) -> None:
        self._capabilities: dict[str, Capability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        name = getattr(capability, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Capabilities require a non-empty stable name")
        if name in self._capabilities:
            raise ValueError(f"Capability already registered: {name}")
        if not all(hasattr(capability, attribute) for attribute in ("supports", "parameter_model", "execute")):
            raise TypeError("Capability does not implement the required interface")
        self._capabilities[name] = capability

    def get(self, name: str) -> Capability | None:
        return self._capabilities.get(name)

    def has(self, name: str) -> bool:
        return name in self._capabilities

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def supported_operations(self, name: str) -> tuple[str, ...]:
        """Expose declared operations only when a capability opts in.

        The executor does not need enumeration to dispatch; this helper is for
        diagnostics and tests and never invokes an operation.
        """

        capability = self.get(name)
        if capability is None:
            return ()
        operations = getattr(capability, "operations", ())
        if not isinstance(operations, tuple):
            return ()
        return tuple(item for item in operations if isinstance(item, str))
