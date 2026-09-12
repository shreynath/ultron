"""Local-only inference, storage, and model lifecycle primitives.

This package deliberately has no HTTP client and no provider SDK dependency.
It is the boundary between Ultron's frozen cognitive layer and local weights.
"""

from app.local_intelligence.core import (
    BenchmarkResult,
    LocalModelError,
    LocalModelOrchestrator,
    LocalModelRegistry,
    LocalModelRuntime,
    LocalModelUnavailable,
    MlxLocalRuntime,
    ModelCapability,
    ModelOrigin,
    ModelRecord,
    ModelStatus,
    NetworkMode,
    get_local_orchestrator,
)

__all__ = [
    "BenchmarkResult",
    "LocalModelError",
    "LocalModelOrchestrator",
    "LocalModelRegistry",
    "LocalModelRuntime",
    "LocalModelUnavailable",
    "MlxLocalRuntime",
    "ModelCapability",
    "ModelOrigin",
    "ModelRecord",
    "ModelStatus",
    "NetworkMode",
    "get_local_orchestrator",
]
