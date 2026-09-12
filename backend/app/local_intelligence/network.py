"""Explicit policy gate for application-owned network tools."""
from __future__ import annotations

from app.config import settings
from app.local_intelligence.core import NetworkMode


class NetworkToolRequired(RuntimeError):
    """A task requested a network tool while the selected mode disallows it."""


def require_network_tool(tool: str) -> None:
    try:
        mode = NetworkMode(settings.local_network_mode)
    except ValueError as exc:
        raise NetworkToolRequired("Invalid local network mode configuration.") from exc
    if mode is not NetworkMode.LOCAL_PLUS_TOOLS:
        raise NetworkToolRequired(f"NETWORK TOOL REQUIRED: {tool} is disabled in {mode.value} mode.")
