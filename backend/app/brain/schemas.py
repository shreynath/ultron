from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TaskType(str, Enum):
    CASUAL = "casual"
    QUESTION = "question"
    CONVERSATION = "conversation"
    COMPUTATION = "computation"
    RETRIEVAL = "retrieval"
    RESEARCH = "research"
    CODING = "coding"
    CREATIVE = "creative"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    INSTRUCTION = "instruction"
    TRANSFORMATION = "transformation"
    UNKNOWN = "unknown"


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ReasoningBudget(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Urgency(str, Enum):
    NORMAL = "normal"
    URGENT = "urgent"


class ExecutiveOutput(BaseModel):
    """Structured fields produced by the Executive model call."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    # The primary type remains the routing key. Additional types retain a
    # hybrid request's shape without allowing model prose to select actions.
    task_types: tuple[TaskType, ...] = Field(default=(), max_length=6)
    complexity: Complexity
    requires_tools: bool
    requires_memory: bool
    requires_planning: bool
    requires_verification: bool
    direct_response_allowed: bool
    reasoning_budget: ReasoningBudget
    urgency: Urgency
    user_goal: str = Field(min_length=1, max_length=400)
    rationale: str = Field(min_length=1, max_length=600)


class TaskAssessment(ExecutiveOutput):
    """Validated Executive result consumed by later cognitive stages."""

    used_fallback: bool = False


def fallback_assessment() -> TaskAssessment:
    """Conservative assessment used when the model call or parse fails."""

    return TaskAssessment(
        task_type=TaskType.UNKNOWN,
        complexity=Complexity.MODERATE,
        requires_tools=False,
        requires_memory=False,
        requires_planning=False,
        requires_verification=False,
        direct_response_allowed=True,
        reasoning_budget=ReasoningBudget.MEDIUM,
        urgency=Urgency.NORMAL,
        user_goal="Unable to assess the request; continue with a direct conversational reply.",
        rationale="Executive assessment failed; using a conservative direct-response fallback.",
        used_fallback=True,
    )


def executive_json_schema() -> dict:
    """JSON Schema sent to the model. Excludes application-only fields."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "task_type": {"type": "string", "enum": [item.value for item in TaskType]},
            "task_types": {"type": "array", "items": {"type": "string", "enum": [item.value for item in TaskType]}, "maxItems": 6},
            "complexity": {"type": "string", "enum": [item.value for item in Complexity]},
            "requires_tools": {"type": "boolean"},
            "requires_memory": {"type": "boolean"},
            "requires_planning": {"type": "boolean"},
            "requires_verification": {"type": "boolean"},
            "direct_response_allowed": {"type": "boolean"},
            "reasoning_budget": {
                "type": "string",
                "enum": [item.value for item in ReasoningBudget],
            },
            "urgency": {"type": "string", "enum": [item.value for item in Urgency]},
            "user_goal": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "task_type",
            "task_types",
            "complexity",
            "requires_tools",
            "requires_memory",
            "requires_planning",
            "requires_verification",
            "direct_response_allowed",
            "reasoning_budget",
            "urgency",
            "user_goal",
            "rationale",
        ],
    }
