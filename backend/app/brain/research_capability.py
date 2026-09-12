"""Explicit application-owned capability for bounded public-web research."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult, ActionStatus, CapabilityResult
from app.brain.conversation_capabilities import CapabilityDescriptor, ConversationAction
from app.brain.learning import LearningRepository
from app.brain.research import ResearchEngine, ResearchTask
from app.brain.schemas import TaskAssessment, TaskType
from app.config import settings


class ResearchParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=3, max_length=500)


class ResearchCapability:
    name = "web_research"
    operations = ("research", "clarify_corpus")
    descriptor = CapabilityDescriptor(
        name=name,
        operations=operations,
        # Executive triage uses RETRIEVAL for a factual lookup and RESEARCH for
        # an explicitly exploratory request. Both are fulfilled by the same
        # application-owned, authorized public-web capability.
        supported_task_types=(TaskType.RETRIEVAL, TaskType.RESEARCH, TaskType.QUESTION),
    )

    def __init__(self, engine: ResearchEngine | None = None) -> None:
        self._engine = engine or ResearchEngine(
            repository=LearningRepository(settings.brain_learning_db)
        )

    def supports(self, operation: str) -> bool:
        return operation in self.operations

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return ResearchParameters if self.supports(operation) else None

    def build_conversation_action(
        self, *, user_input: str, assessment: TaskAssessment
    ) -> ConversationAction | None:
        if assessment.task_type not in self.descriptor.supported_task_types:
            return None
        # Exact positions in a named corpus are under-specified unless a source
        # or edition is supplied. This is intentionally corpus-generic, not a
        # Bible-specific exception, and avoids fake deterministic counting.
        if _requires_corpus_clarification(user_input):
            return ConversationAction(operation="clarify_corpus", parameters={"question": user_input})
        # A bare corpus choice is meaningful only with an earlier positional
        # request. Let the bridge resolve that follow-up to document analysis
        # instead of sending the corpus name to broad web research.
        if re.search(r"\b(?:king\s+james|kjv)\b", user_input, re.IGNORECASE) and not _has_position(user_input):
            return None
        if not (assessment.requires_tools or assessment.task_type is TaskType.RESEARCH):
            return None
        return ConversationAction(operation="research", parameters={"question": user_input})

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        if not isinstance(parameters, ResearchParameters):
            raise ValueError("Unsupported bounded research operation.")
        if operation == "clarify_corpus":
            return CapabilityResult(output={
                "clarification": "Which edition, translation, or source URL should I use? Exact positions depend on the corpus text and tokenization rules.",
                "requires_source_identity": True,
            })
        if operation != "research":
            raise ValueError("Unsupported bounded research operation.")
        result = await self._engine.run(ResearchTask(question=parameters.question))
        # Full sources/claims/evidence are durably persisted by ResearchStore.
        # The action boundary carries only a bounded response projection so it
        # cannot be truncated before the grounded response layer sees it.
        return CapabilityResult(output={
            "research_id": str(result.research_id),
            "answer": result.answer,
            "verification_status": result.verification_status.value if result.verification_status else None,
            "confidence": result.confidence,
            "source_count": len(result.sources),
            "claim_count": len(result.claims),
            "uncertainty": list(result.uncertainty),
            # Provenance is deliberately bounded to retrieved public sources;
            # page bodies never cross the conversation action boundary.
            "sources": [
                {
                    "url": source.canonical_url or source.url,
                    "domain": source.domain,
                    "title": source.title,
                    "retrieval_status": source.retrieval_status.value,
                }
                for source in result.sources
                if source.retrieval_status.value in {"retrieved", "unchanged"}
            ][:4],
            "evidence_claim": result.claims[0].claim_text if result.claims else None,
        })

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is not ActionStatus.COMPLETED or not isinstance(result.output, dict):
            return None
        clarification = result.output.get("clarification")
        if isinstance(clarification, str):
            return clarification
        answer = result.output.get("answer")
        if not isinstance(answer, str) or not answer:
            return None
        sources = result.output.get("sources")
        if not isinstance(sources, list) or not sources:
            return answer
        rendered = []
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("url"), str):
                continue
            label = source.get("title") or source.get("domain") or "Retrieved source"
            rendered.append(f"- {label}: {source['url']}")
        return f"{answer}\n\nSources retrieved:\n" + "\n".join(rendered) if rendered else answer


def _requires_corpus_clarification(user_input: str) -> bool:
    """Detect an exact-document request without a reproducible corpus identity."""
    text = user_input.lower()
    positional = any(token in text for token in ("word", "character", "sentence", "line")) and any(
        marker in text for marker in ("st ", "nd ", "rd ", "th ", "first ", "last ")
    )
    has_source = "http://" in text or "https://" in text or "edition" in text or "translation" in text
    return positional and not has_source


def _has_position(user_input: str) -> bool:
    return bool(
        re.search(r"\b\d+(?:st|nd|rd|th)\s+word\b", user_input, re.IGNORECASE)
        or re.search(r"\bwords?\s+\d+\s+(?:through|to|-)\s+\d+\b", user_input, re.IGNORECASE)
    )
