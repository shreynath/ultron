"""Explicit curation and replay planning for controlled continual learning."""
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CurationDecision(str, Enum):
    INCLUDE = "INCLUDE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class ExperienceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experience_id: str
    provenance_verified: bool
    privacy_safe: bool
    duplicate: bool
    outcome_verified: bool
    contamination_free: bool
    quality_score: float = Field(ge=0, le=1)
    failure_class: str | None = None


class CuratedExperience(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experience_id: str
    decision: CurationDecision
    reasons: tuple[str, ...]


class ReplayMixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    historical_foundation: float = Field(gt=0, le=1)
    new_curated_data: float = Field(ge=0, le=1)
    verified_experiences: float = Field(ge=0, le=1)

    def model_post_init(self, __context: object, /) -> None:
        if abs(self.historical_foundation + self.new_curated_data + self.verified_experiences - 1.0) > 1e-6:
            raise ValueError("Replay mixture must sum to 1.0")


def curate(candidate: ExperienceCandidate) -> CuratedExperience:
    failures: list[str] = []
    if not candidate.provenance_verified:
        failures.append("PROVENANCE")
    if not candidate.privacy_safe:
        failures.append("PRIVACY")
    if candidate.duplicate:
        failures.append("DUPLICATE")
    if not candidate.contamination_free:
        failures.append("CONTAMINATION")
    if failures:
        return CuratedExperience(experience_id=candidate.experience_id, decision=CurationDecision.REJECT, reasons=tuple(failures))
    if not candidate.outcome_verified or candidate.quality_score < 0.7:
        return CuratedExperience(experience_id=candidate.experience_id, decision=CurationDecision.REVIEW, reasons=("OUTCOME_OR_QUALITY_INSUFFICIENT",))
    return CuratedExperience(experience_id=candidate.experience_id, decision=CurationDecision.INCLUDE, reasons=("VERIFIED_CURATED",))


def curate_all(candidates: Iterable[ExperienceCandidate]) -> tuple[CuratedExperience, ...]:
    return tuple(curate(candidate) for candidate in candidates)
