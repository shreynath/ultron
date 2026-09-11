from __future__ import annotations

import asyncio
import os

import pytest

from app.brain.executive import Executive
from app.brain.schemas import Complexity, ReasoningBudget, TaskType

pytestmark = pytest.mark.live_ollama

LIVE_ENABLED = os.environ.get("ULTRON_LIVE_BRAIN") == "1"


def _require_live() -> None:
    if not LIVE_ENABLED:
        pytest.skip("Set ULTRON_LIVE_BRAIN=1 to run live Executive tests")


def _assess(text: str):
    return asyncio.run(Executive().assess(text, history=[]))


def test_live_casual():
    _require_live()
    assessment = _assess("Hey Ultron, how are you?")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.CASUAL, TaskType.CONVERSATION}
    assert assessment.complexity is Complexity.TRIVIAL
    assert assessment.requires_tools is False
    assert assessment.requires_planning is False
    assert assessment.requires_verification is False
    assert assessment.direct_response_allowed is True
    assert assessment.reasoning_budget is ReasoningBudget.MINIMAL


def test_live_factual_question():
    _require_live()
    assessment = _assess("What is the capital of France?")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.QUESTION, TaskType.UNKNOWN}
    assert assessment.complexity in {Complexity.TRIVIAL, Complexity.SIMPLE}
    assert assessment.requires_tools is False
    assert assessment.requires_planning is False
    assert assessment.direct_response_allowed is True


def test_live_arithmetic():
    _require_live()
    assessment = _assess("What is 37 × 284?")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.COMPUTATION, TaskType.QUESTION}
    assert assessment.requires_tools is True
    assert assessment.requires_verification is True
    assert assessment.direct_response_allowed is False


def test_live_exact_retrieval():
    _require_live()
    assessment = _assess("What is the 18th word of the 10th line of this document?")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.RETRIEVAL, TaskType.QUESTION}
    assert assessment.requires_tools is True
    assert assessment.requires_planning is True
    assert assessment.requires_verification is True
    assert assessment.direct_response_allowed is False


def test_live_research():
    _require_live()
    assessment = _assess(
        "Research whether this scientific claim is true. Compare multiple sources "
        "and explain any contradictions."
    )
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.RESEARCH, TaskType.ANALYSIS}
    assert assessment.requires_tools is True
    assert assessment.requires_planning is True
    assert assessment.requires_verification is True
    assert assessment.reasoning_budget in {ReasoningBudget.HIGH, ReasoningBudget.MAXIMUM}


def test_live_coding():
    _require_live()
    assessment = _assess(
        "Write a Python function that sorts these records and confirm that it works."
    )
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.CODING, TaskType.INSTRUCTION}
    assert assessment.requires_tools is True
    assert assessment.requires_planning is True
    assert assessment.requires_verification is True
    assert assessment.direct_response_allowed is False


def test_live_creative():
    _require_live()
    assessment = _assess("Write me a short poem about the moon.")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.CREATIVE, TaskType.INSTRUCTION}
    assert assessment.requires_tools is False
    assert assessment.requires_verification is False
    assert assessment.direct_response_allowed is True


def test_live_planning():
    _require_live()
    assessment = _assess("Help me plan a three-day trip to Chicago with a $500 budget.")
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.PLANNING, TaskType.INSTRUCTION}
    assert assessment.requires_planning is True


def test_live_analysis():
    _require_live()
    assessment = _assess(
        "Analyze this dataset, identify the strongest relationship, test whether it "
        "is statistically significant, and explain the limitations."
    )
    assert assessment.used_fallback is False
    assert assessment.task_type in {TaskType.ANALYSIS, TaskType.RESEARCH}
    assert assessment.requires_tools is True
    assert assessment.requires_planning is True
    assert assessment.requires_verification is True
    assert assessment.reasoning_budget in {ReasoningBudget.HIGH, ReasoningBudget.MAXIMUM}
