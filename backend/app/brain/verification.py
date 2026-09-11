"""Step 7 bounded evidence and verification for completed actions.

This module evaluates only evidence that already exists.  It never searches,
plans, executes an action, calls a model, or retries an answer.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult
from app.brain.schemas import TaskAssessment, TaskType


class EvidenceSourceType(str, Enum):
    CAPABILITY_RESULT = "capability_result"
    RETRIEVED_DOCUMENT = "retrieved_document"
    STRUCTURED_DATASET = "structured_dataset"
    SCIENTIFIC_REFERENCE = "scientific_reference"
    DETERMINISTIC_COMPUTATION = "deterministic_computation"
    USER_PROVIDED = "user_provided"
    MODEL_GENERATED = "model_generated"


class EvidenceReliability(str, Enum):
    AUTHORITATIVE = "authoritative"
    TRUSTED = "trusted"
    UNVERIFIED = "unverified"


class ClaimImportance(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class ClaimType(str, Enum):
    FACT = "fact"
    EXACT_VALUE = "exact_value"
    NUMERIC_VALUE = "numeric_value"
    EXACT_SEQUENCE = "exact_sequence"
    POSITIONAL_VALUE = "positional_value"
    RANGE = "range"
    COMPUTATION = "computation"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    STRUCTURED_OBJECT = "structured_object"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=400)
    importance: ClaimImportance = ClaimImportance.PRIMARY
    claim_type: ClaimType = ClaimType.FACT
    expected_value: str | None = Field(default=None, max_length=400)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: UUID
    source_type: EvidenceSourceType
    source_identifier: str = Field(min_length=1, max_length=200)
    claim: str = Field(min_length=1, max_length=400)
    content: str = Field(min_length=1, max_length=2_000)
    reliability: EvidenceReliability
    retrieved_at: datetime


class EvidenceCollection(BaseModel):
    """Immutable, task-local evidence passed once to a verifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: UUID
    items: tuple[Evidence, ...] = Field(max_length=8)


class EvidenceProvider(Protocol):
    """Capability-owned adapter. It only converts an existing action result."""

    capability: str

    def produce_evidence(self, *, task_id: UUID, action_result: ActionResult) -> EvidenceCollection:
        """Produce bounded evidence without retrieval, execution, or model calls."""

    def claims(self, action_result: ActionResult) -> tuple[Claim, ...]: ...

    def fallback_answer(self, action_result: ActionResult) -> str: ...


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[Claim, ...] = Field(max_length=8)
    evidence: EvidenceCollection
    fallback_answer: str = Field(min_length=1, max_length=1_000)


class EvidenceProviderRegistry:
    """Application-owned mapping from a stable capability operation to evidence."""

    def __init__(self, providers: tuple[EvidenceProvider, ...] = ()) -> None:
        self._providers: dict[str, EvidenceProvider] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: EvidenceProvider) -> None:
        key = provider.capability
        if not key or key in self._providers:
            raise ValueError("Evidence provider keys must be unique non-empty strings")
        self._providers[key] = provider

    def get(self, capability: str, operation: str) -> EvidenceProvider | None:
        return self._providers.get(f"{capability}.{operation}")


class WeatherEvidenceProvider:
    """Weather-domain parsing isolated behind the generic evidence boundary."""

    capability = "weather.current_weather"

    def produce_evidence(self, *, task_id: UUID, action_result: ActionResult) -> EvidenceCollection:
        output = action_result.output
        if not isinstance(output, dict):
            raise TypeError("Weather evidence requires structured output")
        required = ("provider", "location", "temperature_f", "condition", "retrieved_at")
        if any(key not in output for key in required):
            raise ValueError("Weather result is missing provenance")
        retrieved_at = datetime.fromisoformat(str(output["retrieved_at"]))
        if retrieved_at.tzinfo is None:
            raise ValueError("Weather retrieval timestamp must be timezone-aware")
        evidence = Evidence(
            evidence_id=uuid4(), source_type=EvidenceSourceType.CAPABILITY_RESULT,
            source_identifier=f"{output['provider']}:weather.current_weather",
            claim=f"Current weather for {output['location']}",
            content=json.dumps({key: output[key] for key in required}, sort_keys=True),
            reliability=EvidenceReliability.AUTHORITATIVE, retrieved_at=retrieved_at,
        )
        return EvidenceCollection(task_id=task_id, items=(evidence,))

    def claims(self, action_result: ActionResult) -> tuple[Claim, ...]:
        output = action_result.output
        assert isinstance(output, dict)
        return (
            Claim(text=f"Current temperature in {output['location']}: {float(output['temperature_f']):.0f}°F", claim_type=ClaimType.NUMERIC_VALUE, expected_value=str(round(float(output['temperature_f'])))),
            Claim(text=f"Current condition in {output['location']}: {output['condition']}", importance=ClaimImportance.SECONDARY, claim_type=ClaimType.CATEGORICAL, expected_value=str(output['condition']).lower()),
        )

    def fallback_answer(self, action_result: ActionResult) -> str:
        output = action_result.output
        assert isinstance(output, dict)
        return f"It’s currently {float(output['temperature_f']):.0f}°F and {str(output['condition']).lower()} in {output['location']}."


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verification_id: UUID
    task_id: UUID
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1)
    claims: tuple[Claim, ...] = Field(max_length=8)
    supporting_evidence: tuple[Evidence, ...] = Field(max_length=8)
    contradicting_evidence: tuple[Evidence, ...] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=500)
    latency_ms: float = Field(ge=0)


_EXACT_REQUEST = re.compile(
    r"\b(?:first\s+\d+\s+digits?|\d+(?:st|nd|rd|th)\s+"
    r"(?:word|character|item|verse|record|digit)|exact\s+(?:value|answer|date)|verify)\b",
    re.IGNORECASE,
)


class VerificationPolicy:
    """Decide if already-obtained evidence must be evaluated."""

    def requires(self, assessment: TaskAssessment, user_input: str) -> bool:
        return bool(
            assessment.requires_verification
            or assessment.requires_tools
            or assessment.task_type is TaskType.RETRIEVAL
            or _EXACT_REQUEST.search(user_input)
        )


class EvidenceVerifier:
    """Compare bounded, capability-provided claims against bounded evidence.

    The verifier is deliberately domain-agnostic: capabilities own the
    conversion from their result to claims and evidence.  This class makes one
    deterministic pass and never executes actions, calls a model, or retries.
    """

    def verify(self, *, task_id: UUID, candidate_answer: str, claims: tuple[Claim, ...], evidence: EvidenceCollection) -> VerificationResult:
        started = datetime.now(UTC)
        if evidence.task_id != task_id:
            return self._result(task_id, VerificationStatus.FAILED, claims, (), (), "Evidence belongs to a different task.", started)
        if not claims or not evidence.items:
            return self._result(task_id, VerificationStatus.UNVERIFIED, claims, (), (), "No evidence-backed claims are available.", started)
        answer = candidate_answer.lower()
        supported = tuple(claim for claim in claims if claim.expected_value and claim.expected_value.lower() in answer)
        if len(supported) == len(claims):
            return self._result(task_id, VerificationStatus.VERIFIED, claims, evidence.items, (), "All evidence-backed claims match the candidate answer.", started)
        if supported:
            return self._result(task_id, VerificationStatus.PARTIALLY_VERIFIED, claims, evidence.items, (), "Only some evidence-backed claims match the candidate answer.", started)
        return self._result(task_id, VerificationStatus.CONTRADICTED, claims, (), evidence.items, "The candidate answer conflicts with evidence-backed claims.", started)

    @staticmethod
    def exact_value_matches(*, candidate: str, expected: str) -> bool:
        return candidate.strip() == expected.strip()

    @staticmethod
    def exact_sequence_status(*, candidate: str, expected: str) -> VerificationStatus:
        candidate_normalized = "".join(character for character in candidate if not character.isspace())
        expected_normalized = "".join(character for character in expected if not character.isspace())
        if candidate_normalized == expected_normalized:
            return VerificationStatus.VERIFIED
        if expected_normalized.startswith(candidate_normalized):
            return VerificationStatus.PARTIALLY_VERIFIED
        return VerificationStatus.CONTRADICTED

    @staticmethod
    def range_matches(*, candidate: float, minimum: float, maximum: float) -> bool:
        return minimum <= candidate <= maximum

    @staticmethod
    def _result(task_id: UUID, status: VerificationStatus, claims: tuple[Claim, ...], supporting: tuple[Evidence, ...], contradicting: tuple[Evidence, ...], reason: str, started: datetime) -> VerificationResult:
        latency = max(0.0, (datetime.now(UTC) - started).total_seconds() * 1000)
        confidence = 1.0 if status is VerificationStatus.VERIFIED else 0.0
        return VerificationResult(verification_id=uuid4(), task_id=task_id, status=status, confidence=confidence, claims=claims, supporting_evidence=supporting, contradicting_evidence=contradicting, reason=reason, latency_ms=latency)
