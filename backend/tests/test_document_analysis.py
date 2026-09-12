from __future__ import annotations

import pytest

from app.brain.conversation_bridge import ConversationCapabilityBridge
from app.brain.document_analysis import (
    DeterministicDocumentAnalyzer,
    DocumentAnalysisCapability,
    _work_reference,
)
from app.brain.runtime import CognitiveRuntime
from app.brain.schemas import (
    Complexity,
    ReasoningBudget,
    TaskAssessment,
    TaskType,
    Urgency,
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
        user_goal="Find exact corpus positions.",
        rationale="Deterministic document analysis is required.",
    )


def test_analyzer_counts_explicit_whitespace_tokens_reproducibly() -> None:
    text = " ".join(["prefix"] * 494 + ["the", "earth", "was", "without", "form,", "and", "void;", "and", "darkness", "was", "upon"])
    result = DeterministicDocumentAnalyzer().analyze(
        corpus_id="fixture:v1", source_url="https://example.test/kjv", text=text, start=495, end=505,
    )
    assert [(row["index"], row["token"]) for row in result["tokens"]] == [
        (495, "the"), (496, "earth"), (497, "was"), (498, "without"), (499, "form,"),
        (500, "and"), (501, "void;"), (502, "and"), (503, "darkness"), (504, "was"), (505, "upon"),
    ]
    assert result["tokenizer_id"] == "unicode-whitespace-v1"
    assert result["index_base"] == 1


def test_same_analyzer_handles_unrelated_literary_and_public_text_fixtures() -> None:
    analyzer = DeterministicDocumentAnalyzer()
    sonnet = analyzer.analyze(
        corpus_id="literary:sonnet", source_url="https://archive.example/sonnet",
        text="Shall I compare thee to a summer's day? Thou art more lovely.", start=10, end=10,
    )
    address = analyzer.analyze(
        corpus_id="speech:address", source_url="https://archive.example/address",
        text="Four score and seven years ago our fathers brought forth", start=5, end=8,
    )
    assert sonnet["tokens"] == ({"index": 10, "token": "art"},)
    assert [row["token"] for row in address["tokens"]] == ["years", "ago", "our", "fathers"]
    assert sonnet["tokenizer_id"] == address["tokenizer_id"] == "unicode-whitespace-v1"


def test_work_reference_parser_is_title_and_author_generic() -> None:
    assert _work_reference("What is the 10th word in Sonnet 18 by Shakespeare?") == ("Sonnet 18", "Shakespeare")
    assert _work_reference("What is the 50th word in The Raven by Edgar Allan Poe?") == ("The Raven", "Edgar Allan Poe")


@pytest.mark.anyio
async def test_registered_kjv_action_executes_and_verifies_through_bridge() -> None:
    text = " ".join(["prefix"] * 499 + ["and", "void;"])

    async def fetcher() -> tuple[str, str]:
        return "https://corpus.example/kjv/genesis-1", text

    capability = DocumentAnalysisCapability(fetcher=fetcher)
    # Replace the default bridge with the same explicit registry shape except
    # the corpus fetcher is deterministic for this end-to-end boundary test.
    from app.brain.conversation_capabilities import ConversationCapabilityRegistry
    from app.brain.executor import ActionExecutor
    from app.brain.observation import ObservationInterpreter

    registry = ConversationCapabilityRegistry((capability,))
    bridge = ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(registry=registry.executor_registry, allowed_capabilities=registry.names()),
        observer=ObservationInterpreter(),
    )
    runtime = CognitiveRuntime(conversation_bridge=bridge)
    execution = await runtime.handle_conversation(
        user_input="What is the 500th word in the King James Version?",
        history=[], assessment=_assessment(),
    )
    assert execution.capability_name == "document_analysis"
    assert execution.capability_operation == "analyze_corpus"
    assert execution.verification is not None
    assert execution.verification.status.value == "verified"
    assert "500 and" in execution.answer
    assert execution.request_id is not None
    events = [event.event for event in execution.trace]
    assert "CAPABILITY_COMPLETED" in events
    assert {
        "CORPUS_RESOLVED", "SOURCE_RETRIEVED", "TEXT_EXTRACTED",
        "TOKENS_COMPUTED", "DETERMINISTIC_INDEX_COMPLETED",
    }.issubset(events)


@pytest.mark.anyio
async def test_follow_up_corpus_selection_reuses_prior_target() -> None:
    text = " ".join(["prefix"] * 499 + ["and"])

    async def fetcher() -> tuple[str, str]:
        return "https://corpus.example/kjv/genesis-1", text

    capability = DocumentAnalysisCapability(fetcher=fetcher)
    from app.brain.conversation_capabilities import ConversationCapabilityRegistry
    from app.brain.executor import ActionExecutor
    from app.brain.observation import ObservationInterpreter
    registry = ConversationCapabilityRegistry((capability,))
    bridge = ConversationCapabilityBridge(registry=registry, executor=ActionExecutor(registry=registry.executor_registry, allowed_capabilities=registry.names()), observer=ObservationInterpreter())
    runtime = CognitiveRuntime(conversation_bridge=bridge)
    execution = await runtime.handle_conversation(
        user_input="do the King James version",
        history=[{"role": "user", "content": "what is the 500th word in the bible"}],
        assessment=_assessment(),
    )
    assert execution.capability_name == "document_analysis"
    assert "500 and" in execution.answer


@pytest.mark.anyio
async def test_corpus_retrieval_failure_never_returns_a_successful_token_answer() -> None:
    async def unavailable() -> tuple[str, str]:
        raise ValueError("source offline")

    capability = DocumentAnalysisCapability(fetcher=unavailable)
    from app.brain.conversation_capabilities import ConversationCapabilityRegistry
    from app.brain.executor import ActionExecutor
    from app.brain.observation import ObservationInterpreter

    registry = ConversationCapabilityRegistry((capability,))
    bridge = ConversationCapabilityBridge(
        registry=registry,
        executor=ActionExecutor(registry=registry.executor_registry, allowed_capabilities=registry.names()),
        observer=ObservationInterpreter(),
    )
    execution = await CognitiveRuntime(conversation_bridge=bridge).handle_conversation(
        user_input="What is the 500th word in the King James Version?",
        history=[], assessment=_assessment(),
    )
    assert execution.status.value == "failed"
    assert execution.action_status == "failed"
    assert "500 and" not in execution.answer
    assert "couldn't complete" in execution.answer.lower()
    assert "CORPUS_RESOLVED" not in [event.event for event in execution.trace]
