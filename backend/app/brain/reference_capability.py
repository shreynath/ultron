"""Bounded local deterministic/reference capability used to exercise Step 7."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult, CapabilityResult
from app.brain.conversation_capabilities import CapabilityDescriptor, ConversationAction
from app.brain.schemas import TaskAssessment, TaskType
from app.brain.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceCollection,
    EvidenceReliability,
    EvidenceSourceType,
)

PI_100 = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
TOKENS = ("alpha", "beta", "gamma", "delta", "epsilon")

class ReferenceParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    topic: str = Field(min_length=1, max_length=32)
    position: int | None = Field(default=None, ge=1, le=len(TOKENS))

class ReferenceCapability:
    name = "reference"
    operations = ("pi_prefix", "positional_lookup", "science_reference")
    descriptor = CapabilityDescriptor(name=name, operations=operations, supported_task_types=(TaskType.RETRIEVAL,))
    def supports(self, operation: str) -> bool: return operation in self.operations
    def parameter_model(self, operation: str) -> type[BaseModel] | None: return ReferenceParameters if self.supports(operation) else None
    def build_conversation_action(self, *, user_input: str, assessment: TaskAssessment) -> ConversationAction | None:
        text = user_input.lower()
        if not assessment.requires_tools: return None
        if "digits of pi" in text:
            return ConversationAction(operation="pi_prefix", parameters={"topic":"pi"})
        match = re.search(r"(\d+)(?:st|nd|rd|th)\s+(?:word|token)", text)
        if "synthetic" in text and match:
            return ConversationAction(operation="positional_lookup", parameters={"topic":"synthetic", "position":int(match.group(1))})
        if "moon" in text or "sun" in text:
            return ConversationAction(operation="science_reference", parameters={"topic":"moon_sun"})
        return None
    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        assert isinstance(parameters, ReferenceParameters)
        now = datetime.now(UTC).isoformat()
        if operation == "pi_prefix": return CapabilityResult(output={"value":PI_100,"requested_digits":100,"source":"pi-constant-v1","retrieved_at":now})
        if operation == "positional_lookup": return CapabilityResult(output={"value":TOKENS[(parameters.position or 1)-1],"position":parameters.position,"corpus_id":"synthetic-test-corpus","corpus_version":"1","tokenizer_id":"whitespace-v1","retrieved_at":now})
        if operation == "science_reference": return CapabilityResult(output={"moon_surface_temperature_k":{"minimum":100,"maximum":390},"sun_photosphere_temperature_k":5772,"source":"NASA-facts-v1","retrieved_at":now})
        raise ValueError("unsupported operation")
    def format_conversation_result(self, result: ActionResult) -> str | None:
        if not isinstance(result.output, dict): return None
        output=result.output
        if result.operation == "pi_prefix": return output["value"]
        if result.operation == "positional_lookup": return f"Position {output['position']} is {output['value']}."
        return "The Moon's surface spans roughly 100–390 K; the Sun's photosphere is about 5772 K."

class ReferenceEvidenceProvider:
    capability = "reference.pi_prefix"  # registry instances below override this stable key
    def __init__(self, operation: str): self.capability=f"reference.{operation}"
    def produce_evidence(
        self, *, task_id: UUID, action_result: ActionResult
    ) -> EvidenceCollection:
        output = action_result.output
        if not isinstance(output, dict):
            raise TypeError("structured output required")
        retrieved_at = datetime.fromisoformat(str(output["retrieved_at"]))
        if retrieved_at.tzinfo is None:
            raise ValueError("retrieval timestamp must be timezone-aware")
        if action_result.operation == "pi_prefix":
            source_type = EvidenceSourceType.DETERMINISTIC_COMPUTATION
            source_identifier = str(output["source"])
        elif action_result.operation == "positional_lookup":
            source_type = EvidenceSourceType.STRUCTURED_DATASET
            source_identifier = f"{output['corpus_id']}:v{output['corpus_version']}"
        else:
            source_type = EvidenceSourceType.SCIENTIFIC_REFERENCE
            source_identifier = str(output["source"])
        evidence = Evidence(
            evidence_id=uuid4(), source_type=source_type,
            source_identifier=source_identifier, claim=action_result.operation,
            content=str(output), reliability=EvidenceReliability.AUTHORITATIVE,
            retrieved_at=retrieved_at,
        )
        return EvidenceCollection(task_id=task_id, items=(evidence,))
    def claims(self, result: ActionResult) -> tuple[Claim, ...]:
        output = result.output
        assert isinstance(output, dict)
        if result.operation == "pi_prefix":
            return (Claim(text="Pi prefix", claim_type=ClaimType.EXACT_SEQUENCE, expected_value=output["value"]),)
        if result.operation == "positional_lookup":
            return (Claim(text="Positional value", claim_type=ClaimType.POSITIONAL_VALUE, expected_value=output["value"]),)
        return (
            Claim(text="Moon temperature range", claim_type=ClaimType.RANGE, expected_value="100"),
            Claim(text="Sun photosphere temperature", claim_type=ClaimType.NUMERIC_VALUE, expected_value="5772"),
        )

    def fallback_answer(self, result: ActionResult) -> str:
        return ReferenceCapability().format_conversation_result(result) or "Reference result unavailable."
