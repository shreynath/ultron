from __future__ import annotations

import re

from app.brain.schemas import (
    Complexity,
    ExecutiveOutput,
    ReasoningBudget,
    TaskType,
    Urgency,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_URGENCY_MARKERS = re.compile(
    r"\b("
    r"urgent|urgently|asap|immediately|right now|right away|"
    r"emergency|time[- ]sensitive|hurry|quickly as possible|"
    r"deadline|before tonight|this minute"
    r")\b",
    re.IGNORECASE,
)
_LIVE_INFORMATION_MARKERS = re.compile(
    r"\b(?:weather|forecast|temperature)(?:\s+(?:like|is|be))?\s+(?:in|for|at)\b"
    r"|\b(?:current|latest|live|right now|today)\b.{0,48}\b"
    r"(?:weather|forecast|temperature|stock|price|score|news|traffic|flight|availability)\b"
    r"|\b(?:weather|forecast)\b.*\b(?:what(?:'s|\s+is)|how(?:'s|\s+is)|tell me|show me|check)\b",
    re.IGNORECASE,
)
_WEATHER_FOLLOW_UP_MARKERS = re.compile(
    r"\b(?:what|how)\s+about\s+(?:the\s+)?(?:temperature|weather|forecast)\b",
    re.IGNORECASE,
)
_EXACT_EVIDENCE_MARKERS = re.compile(
    r"\b(?:first\s+\d+\s+digits?|\d+(?:st|nd|rd|th)\s+(?:word|character|item|verse|record|digit)|"
    r"(?:words?|tokens?)\s+\d+\s*(?:through|to|-)\s*\d+|"
    r"surface\s+temperature|temperature\s+of\s+(?:the\s+)?(?:moon|sun)|king\s+james\s+bible)\b",
    re.IGNORECASE,
)
_BROAD_RESEARCH_MARKERS = re.compile(
    r"\b(?:list|identify|catalog|enumerate)\b.{0,80}\b(?:every|all)\b.{0,40}\b(?:component|part|subsystem|element|feature)s?\b"
    r"|\b(?:every|all)\s+(?:single\s+)?(?:component|part|subsystem|element|feature)s?\b",
    re.IGNORECASE,
)

CASUAL_TYPES = frozenset({TaskType.CASUAL, TaskType.CONVERSATION})


def clip_text(value: str, max_length: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 1].rstrip() + "…"


def clip_rationale(value: str, max_length: int = 400) -> str:
    collapsed = " ".join(value.split())
    sentences = [part for part in _SENTENCE_SPLIT.split(collapsed) if part]
    shortened = " ".join(sentences[:2]) if sentences else collapsed
    return clip_text(shortened, max_length)


def signals_urgency(user_input: str) -> bool:
    return bool(_URGENCY_MARKERS.search(user_input))


def requires_live_information(user_input: str, history: list[dict] | None = None) -> bool:
    """Identify requests whose answer must come from a live capability.

    This is Executive-level safety normalization, not capability selection.
    It prevents a structured-model miss or its conservative fallback from
    routing an explicitly current-information request to a text-only model.
    """

    if _LIVE_INFORMATION_MARKERS.search(user_input):
        return True
    if not _WEATHER_FOLLOW_UP_MARKERS.search(user_input) or not history:
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and _LIVE_INFORMATION_MARKERS.search(message["content"])
        for message in history
    )


def normalize_output(
    output: ExecutiveOutput,
    user_input: str,
    history: list[dict] | None = None,
) -> ExecutiveOutput:
    """Apply conservative invariants. See docs/BRAIN_EXECUTIVE.md."""

    data = output.model_dump()
    data["user_goal"] = clip_text(data["user_goal"], 300)
    data["rationale"] = clip_rationale(data["rationale"])

    if requires_live_information(user_input, history):
        data["task_type"] = TaskType.RETRIEVAL
        data["complexity"] = Complexity.SIMPLE
        data["requires_tools"] = True
        data["requires_memory"] = False
        data["requires_planning"] = False
        data["requires_verification"] = False
        data["direct_response_allowed"] = False
        data["reasoning_budget"] = ReasoningBudget.LOW

    if _EXACT_EVIDENCE_MARKERS.search(user_input):
        data["task_type"] = TaskType.RETRIEVAL
        data["complexity"] = Complexity.SIMPLE
        data["requires_tools"] = True
        data["requires_memory"] = False
        data["requires_verification"] = True
        data["direct_response_allowed"] = False
        data["reasoning_budget"] = ReasoningBudget.LOW

    if _BROAD_RESEARCH_MARKERS.search(user_input):
        data["task_type"] = TaskType.RESEARCH
        data["complexity"] = Complexity.COMPLEX
        data["requires_tools"] = True
        data["requires_memory"] = False
        data["requires_planning"] = True
        data["requires_verification"] = True
        data["direct_response_allowed"] = False
        data["reasoning_budget"] = ReasoningBudget.HIGH

    if data["task_type"] in CASUAL_TYPES:
        over_escalated = (
            data["requires_tools"]
            and data["requires_planning"]
            and data["requires_verification"]
            and data["reasoning_budget"] == ReasoningBudget.MAXIMUM
        )
        heavy_budget = data["complexity"] in {
            Complexity.TRIVIAL,
            Complexity.SIMPLE,
        } and data["reasoning_budget"] in {
            ReasoningBudget.HIGH,
            ReasoningBudget.MAXIMUM,
        }
        if over_escalated or heavy_budget:
            data["requires_tools"] = False
            data["requires_memory"] = False
            data["requires_planning"] = False
            data["requires_verification"] = False
            data["direct_response_allowed"] = True
            data["reasoning_budget"] = ReasoningBudget.MINIMAL
            if data["complexity"] in {Complexity.COMPLEX, Complexity.VERY_COMPLEX}:
                data["complexity"] = Complexity.SIMPLE

    if data["direct_response_allowed"]:
        data["requires_planning"] = False

    if data["urgency"] == Urgency.URGENT and not signals_urgency(user_input):
        data["urgency"] = Urgency.NORMAL

    if data["complexity"] == Complexity.TRIVIAL and data["reasoning_budget"] in {
        ReasoningBudget.HIGH,
        ReasoningBudget.MAXIMUM,
    }:
        data["reasoning_budget"] = ReasoningBudget.MINIMAL

    return ExecutiveOutput.model_validate(data)
