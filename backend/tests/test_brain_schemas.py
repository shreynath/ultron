from pydantic import ValidationError

from app.brain.schemas import (
    Complexity,
    ExecutiveOutput,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
    executive_json_schema,
    fallback_assessment,
)


def _valid_payload(**overrides) -> dict:
    payload = {
        "task_type": "question",
        "complexity": "simple",
        "requires_tools": False,
        "requires_memory": False,
        "requires_planning": False,
        "requires_verification": False,
        "direct_response_allowed": True,
        "reasoning_budget": "low",
        "urgency": "normal",
        "user_goal": "Identify the capital of France.",
        "rationale": "This is a simple factual question that can be answered directly.",
    }
    payload.update(overrides)
    return payload


def test_valid_assessment_round_trip():
    assessment = TaskAssessment.model_validate({**_valid_payload(), "used_fallback": False})
    assert assessment.task_type is TaskType.QUESTION
    assert assessment.complexity is Complexity.SIMPLE
    assert assessment.reasoning_budget is ReasoningBudget.LOW
    assert assessment.urgency is Urgency.NORMAL
    assert assessment.used_fallback is False


def test_missing_field_rejected():
    payload = _valid_payload()
    del payload["task_type"]
    try:
        ExecutiveOutput.model_validate(payload)
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_invalid_enum_rejected():
    try:
        ExecutiveOutput.model_validate(_valid_payload(task_type="agi"))
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_extra_fields_rejected():
    try:
        ExecutiveOutput.model_validate(_valid_payload(confidence=0.9))
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_model_cannot_set_used_fallback_on_output_schema():
    try:
        ExecutiveOutput.model_validate(_valid_payload(used_fallback=True))
        raise AssertionError("expected ValidationError")
    except ValidationError:
        pass


def test_fallback_never_looks_like_success():
    assessment = fallback_assessment()
    assert assessment.used_fallback is True
    assert assessment.task_type is TaskType.UNKNOWN
    assert assessment.direct_response_allowed is True
    assert assessment.requires_tools is False
    assert assessment.requires_planning is False


def test_json_schema_omits_used_fallback():
    schema = executive_json_schema()
    assert "used_fallback" not in schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
