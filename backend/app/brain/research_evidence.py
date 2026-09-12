"""Evidence adapter for an already completed bounded research action."""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.brain.actions import ActionResult
from app.brain.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceCollection,
    EvidenceReliability,
    EvidenceSourceType,
)


class ResearchEvidenceProvider:
    """Verify response-to-retrieval consistency; it does not fetch or infer."""

    capability = "web_research.research"

    def produce_evidence(self, *, task_id: UUID, action_result: ActionResult) -> EvidenceCollection:
        output = action_result.output
        if not isinstance(output, dict):
            raise TypeError("structured research output required")
        claim = output.get("evidence_claim")
        sources = output.get("sources")
        if not isinstance(claim, str) or not claim or not isinstance(sources, list) or not sources:
            raise ValueError("research did not retrieve a claim with provenance")
        source = sources[0]
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            raise ValueError("research provenance is malformed")
        evidence = Evidence(
            evidence_id=uuid4(),
            source_type=EvidenceSourceType.RETRIEVED_DOCUMENT,
            source_identifier=source["url"][:200],
            claim="retrieved research claim",
            content=claim[:2_000],
            reliability=EvidenceReliability.TRUSTED,
            retrieved_at=datetime.now(UTC),
        )
        return EvidenceCollection(task_id=task_id, items=(evidence,))

    def claims(self, action_result: ActionResult) -> tuple[Claim, ...]:
        output = action_result.output
        assert isinstance(output, dict)
        claim = output.get("evidence_claim")
        return (Claim(text="Retrieved research claim", claim_type=ClaimType.FACT, expected_value=claim),) if isinstance(claim, str) and claim else ()

    def fallback_answer(self, action_result: ActionResult) -> str:
        output = action_result.output
        if isinstance(output, dict) and isinstance(output.get("answer"), str):
            return output["answer"]
        return "I couldn't verify the retrieved research result."
