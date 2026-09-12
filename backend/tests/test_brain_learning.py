from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.brain.learning import (
    ExperienceRecord,
    KnowledgeStatus,
    LearningEventStatus,
    LearningRepository,
    Provenance,
    sanitize_for_persistence,
)
from app.brain.schemas import Complexity, ReasoningBudget, TaskAssessment, TaskType, Urgency
from app.brain.runtime import CognitiveRuntime
from app.brain.verification import (
    Claim,
    Evidence,
    EvidenceCollection,
    EvidenceReliability,
    EvidenceSourceType,
    EvidenceVerifier,
)


def _assessment() -> TaskAssessment:
    return TaskAssessment(
        task_type=TaskType.RETRIEVAL,
        complexity=Complexity.SIMPLE,
        requires_tools=True,
        requires_memory=False,
        requires_planning=False,
        requires_verification=True,
        direct_response_allowed=False,
        reasoning_budget=ReasoningBudget.LOW,
        urgency=Urgency.NORMAL,
        user_goal="Retrieve verified weather.",
        rationale="Test learning.",
    )


def _experience(
    claim_text: str = "Temperature is 72",
    expected_value: str = "72",
    knowledge_key: str | None = None,
) -> ExperienceRecord:
    task_id = uuid4()
    evidence = Evidence(
        evidence_id=uuid4(),
        source_type=EvidenceSourceType.CAPABILITY_RESULT,
        source_identifier="test.weather",
        claim=claim_text,
        content=expected_value,
        reliability=EvidenceReliability.AUTHORITATIVE,
        retrieved_at=datetime.now(UTC),
    )
    verification = EvidenceVerifier().verify(
        task_id=task_id,
        candidate_answer=claim_text,
        claims=(Claim(text=claim_text, expected_value=expected_value, knowledge_key=knowledge_key),),
        evidence=EvidenceCollection(task_id=task_id, items=(evidence,)),
    )
    return ExperienceRecord(
        correlation_id=task_id,
        created_at=datetime.now(UTC),
        normalized_goal="Retrieve verified weather.",
        assessment=_assessment(),
        route="cognitive_execution_required",
        execution_status="completed",
        lifecycle_outcome="completed",
        capability="weather",
        operation="current_weather",
        action_status="completed",
        action={"authorization": "[REDACTED]", "duration_ms": 10.0},
        verification=verification,
        confidence=verification.confidence,
        provenance=Provenance(source="test", correlation_id=task_id, capability="weather"),
    )


def test_sanitization_redacts_nested_secrets_and_bounds_text() -> None:
    sanitized = sanitize_for_persistence({"api_key": "top-secret", "safe": {"token": "nope", "message": "Bearer abcdefghijklmnop"}})
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["safe"]["token"] == "[REDACTED]"
    assert sanitized["safe"]["message"] == "[REDACTED]"


def test_verified_experience_promotes_retrievable_knowledge_and_is_idempotent(tmp_path) -> None:
    repository = LearningRepository(tmp_path / "learning.sqlite3")
    experience = _experience()

    stored = repository.commit_experience(experience)
    again = repository.commit_experience(experience)
    event = repository.promote(stored)
    retrieved = repository.retrieve(query="what temperature")

    assert again.experience_id == stored.experience_id
    assert event.status is LearningEventStatus.PROMOTED
    assert len(event.knowledge_ids) == 1
    assert [item.knowledge.status for item in retrieved] == [KnowledgeStatus.VERIFIED]
    assert retrieved[0].instruction_safe is True
    assert repository.capability_reliability("weather")["success_rate"] == 1.0


def test_unverified_experience_never_promotes_to_verified_knowledge(tmp_path) -> None:
    repository = LearningRepository(tmp_path / "learning.sqlite3")
    experience = _experience().model_copy(update={"verification": None, "confidence": 0.15})

    repository.commit_experience(experience)
    event = repository.promote(experience)

    assert event.status is LearningEventStatus.NOT_ELIGIBLE
    assert repository.retrieve(query="temperature") == ()


def test_conflicting_explicitly_verified_claims_preserve_revision_and_hide_old_claim(tmp_path) -> None:
    repository = LearningRepository(tmp_path / "learning.sqlite3")
    first_experience = _experience("Provider supports region A", "region a", "provider_regions")
    second_experience = _experience("Provider supports region B", "region b", "provider_regions")
    repository.commit_experience(first_experience)
    first_event = repository.promote(first_experience)
    repository.commit_experience(second_experience)
    second_event = repository.promote(second_experience)
    retrieved = repository.retrieve(query="provider supports region")
    first_history = repository.knowledge_history(first_event.knowledge_ids[0])

    assert first_event.knowledge_ids[0] != second_event.knowledge_ids[0]
    assert first_history[-1].status is KnowledgeStatus.CONTRADICTED
    assert len(retrieved) == 1
    assert retrieved[0].knowledge.knowledge_id == second_event.knowledge_ids[0]


def test_verified_retrieval_is_attached_as_typed_untrusted_task_data(tmp_path) -> None:
    repository = LearningRepository(tmp_path / "learning.sqlite3")
    experience = _experience()
    repository.commit_experience(experience)
    repository.promote(experience)
    runtime = CognitiveRuntime()
    context = runtime.create_context(
        user_input="What temperature was learned?",
        history=[],
        assessment=_assessment(),
    )

    runtime.attach_retrieved_knowledge(context=context, repository=repository)

    assert len(context.working_memory.retrieved_knowledge) == 1
    assert context.working_memory.retrieved_knowledge[0].instruction_safe is True
    assert context.working_memory.retrieved_knowledge[0].knowledge.status is KnowledgeStatus.VERIFIED
