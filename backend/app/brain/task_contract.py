"""Typed, non-secret execution contract compiled before capability selection."""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.brain.schemas import TaskAssessment, TaskType


class Precision(str, Enum):
    ORDINARY = "ordinary"
    EXACT = "exact"
    COMPREHENSIVE = "comprehensive"


class TimeRequirement(str, Enum):
    STATIC = "static"
    HISTORICAL = "historical"
    CURRENT = "current"


class TaskContract(BaseModel):
    """Application-visible definition of completion, not model chain-of-thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deliverable: str = Field(min_length=1, max_length=400)
    scope: str = Field(min_length=1, max_length=300)
    success_conditions: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_operations: tuple[str, ...] = Field(min_length=1, max_length=8)
    precision: Precision
    time_requirement: TimeRequirement
    requires_external_evidence: bool
    validation_checks: tuple[str, ...] = Field(min_length=1, max_length=8)
    recovery_policy: str = Field(min_length=1, max_length=240)


_EXACT = re.compile(r"\b(?:exact(?:ly)?|\d+(?:st|nd|rd|th)|count|total|percentage|ranking|position)\b", re.I)
_COMPREHENSIVE = re.compile(r"\b(?:every|all|complete|comprehensive|exhaustive|entire)\b", re.I)
_CURRENT = re.compile(r"\b(?:current|currently|latest|today|now|recent(?:ly)?|this week|as of)\b", re.I)
_HISTORICAL = re.compile(r"\b(?:historical|history|ancient|century|archive|original|transcription)\b", re.I)


def build_task_contract(*, user_input: str, assessment: TaskAssessment) -> TaskContract:
    """Compile bounded completion criteria from validated task metadata.

    This intentionally derives controls from task structure and assessment
    fields, not from capability names or a catalog of example subjects.
    """
    if _COMPREHENSIVE.search(user_input):
        precision = Precision.COMPREHENSIVE
    elif _EXACT.search(user_input) or assessment.task_type is TaskType.COMPUTATION:
        precision = Precision.EXACT
    else:
        precision = Precision.ORDINARY
    if _CURRENT.search(user_input):
        timing = TimeRequirement.CURRENT
    elif _HISTORICAL.search(user_input):
        timing = TimeRequirement.HISTORICAL
    else:
        timing = TimeRequirement.STATIC
    operations: list[str] = ["understand", "classify"]
    if assessment.requires_planning or precision is Precision.COMPREHENSIVE:
        operations.append("plan")
    if assessment.requires_tools:
        operations.append("execute")
    if assessment.task_type is TaskType.COMPUTATION or precision is Precision.EXACT:
        operations.append("compute")
    if assessment.requires_verification or precision is not Precision.ORDINARY:
        operations.append("validate")
    operations.extend(("synthesize", "verify"))
    evidence = assessment.requires_tools or timing is TimeRequirement.CURRENT or assessment.task_type in {TaskType.RETRIEVAL, TaskType.RESEARCH}
    checks = ["deliverable_matches_user_goal"]
    if evidence:
        checks.append("evidence_matches_domain_object_and_scope")
    if precision is Precision.EXACT:
        checks.append("calculation_or_indexing_is_reproducible")
    if precision is Precision.COMPREHENSIVE:
        checks.append("defined_category_coverage_is_complete")
    return TaskContract(
        deliverable=assessment.user_goal,
        scope="User-requested scope; use a defensible convention only where it does not materially change the deliverable.",
        success_conditions=("required operations completed", "validation checks passed", "response matches requested deliverable"),
        required_operations=tuple(dict.fromkeys(operations)), precision=precision,
        time_requirement=timing, requires_external_evidence=evidence,
        validation_checks=tuple(checks),
        recovery_policy="Reject domain/object/scope-mismatched evidence; change retrieval strategy before retrying, and never synthesize rejected evidence.",
    )
