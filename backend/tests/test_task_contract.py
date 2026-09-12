from app.brain.schemas import Complexity, ReasoningBudget, TaskAssessment, TaskType, Urgency
from app.brain.task_contract import Precision, TimeRequirement, build_task_contract


def _assessment(**updates: object) -> TaskAssessment:
    values: dict[str, object] = {
        "task_type": TaskType.RETRIEVAL, "complexity": Complexity.MODERATE,
        "requires_tools": True, "requires_memory": False, "requires_planning": False,
        "requires_verification": True, "direct_response_allowed": False,
        "reasoning_budget": ReasoningBudget.MEDIUM, "urgency": Urgency.NORMAL,
        "user_goal": "Produce the requested result.", "rationale": "Evidence is required.",
    }
    values.update(updates)
    return TaskAssessment(**values)


def test_contract_requires_reproducible_work_for_exact_current_request() -> None:
    contract = build_task_contract(
        user_input="What is the current exact 500th item in this dataset?",
        assessment=_assessment(),
    )
    assert contract.precision is Precision.EXACT
    assert contract.time_requirement is TimeRequirement.CURRENT
    assert contract.requires_external_evidence is True
    assert "compute" in contract.required_operations
    assert "calculation_or_indexing_is_reproducible" in contract.validation_checks


def test_contract_requires_coverage_validation_for_comprehensive_work() -> None:
    contract = build_task_contract(
        user_input="Create a comprehensive inventory of every documented subsystem.",
        assessment=_assessment(requires_planning=True),
    )
    assert contract.precision is Precision.COMPREHENSIVE
    assert "plan" in contract.required_operations
    assert "defined_category_coverage_is_complete" in contract.validation_checks
