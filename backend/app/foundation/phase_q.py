"""Phase Q: controlled data acquisition and curriculum construction for ULTRON.

This module creates training *data*, never a runtime answer path.  Teacher
outputs are untrusted candidate records that require validation before they can
enter an immutable CorpusPipeline release.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DatasetRelease,
    DocumentCategory,
    Provenance,
    SourceType,
    normalize_text,
)
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilitySuite,
    LearningOpportunity,
)
from app.foundation.tokenizer import UltronTokenizer


class LearningTargetStatus(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    ANALYZING = "ANALYZING"
    READY_FOR_DATA = "READY_FOR_DATA"
    DATA_AVAILABLE = "DATA_AVAILABLE"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    SOLVED = "SOLVED"
    UNSOLVED = "UNSOLVED"
    DEFERRED = "DEFERRED"
    INVALIDATED = "INVALIDATED"


class DataSourceType(str, Enum):
    USER_OWNED = "USER_OWNED"
    PUBLIC_LICENSED = "PUBLIC_LICENSED"
    OPEN_DATASET = "OPEN_DATASET"
    STRUCTURED_DATABASE = "STRUCTURED_DATABASE"
    APPROVED_WEB_SOURCE = "APPROVED_WEB_SOURCE"
    TEACHER_GENERATED = "TEACHER_GENERATED"
    SYNTHETIC = "SYNTHETIC"
    MANUALLY_VERIFIED = "MANUALLY_VERIFIED"


class SourceEligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class RecordDisposition(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class TeacherRequestKind(str, Enum):
    ANSWER = "ANSWER"
    EXPLANATION = "EXPLANATION"
    EXAMPLE = "EXAMPLE"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    ALTERNATIVE_SOLUTION = "ALTERNATIVE_SOLUTION"
    QUESTION_GENERATION = "QUESTION_GENERATION"
    DIFFICULTY_ESCALATION = "DIFFICULTY_ESCALATION"
    CRITIQUE = "CRITIQUE"
    ERROR_ANALYSIS = "ERROR_ANALYSIS"


class TeacherValidationStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    DISAGREEMENT = "DISAGREEMENT"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"


class LearningTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    source: str
    domain: CapabilityDomain
    failure_cluster_id: str | None = None
    description: str = Field(min_length=1, max_length=2_000)
    importance: float = Field(gt=0, le=1)
    frequency: float = Field(gt=0, le=1)
    confidence: float = Field(gt=0, le=1)
    expected_generalization: float = Field(gt=0, le=1)
    estimated_training_cost: int = Field(gt=0)
    status: LearningTargetStatus = LearningTargetStatus.IDENTIFIED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DataRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requirement_id: str
    target_id: str
    domain: CapabilityDomain
    required_example_kinds: tuple[str, ...]
    required_subdomains: tuple[str, ...]
    minimum_records: int = Field(gt=0)
    difficulty_bands: tuple[int, ...]
    require_generalization_variants: bool = True
    maximum_teacher_fraction: float = Field(default=0.5, ge=0, le=1)
    maximum_synthetic_fraction: float = Field(default=0.5, ge=0, le=1)
    expected_value: float = Field(ge=0)
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DataSource(BaseModel):
    """A declared source; unknown licenses cannot become eligible automatically."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    source_type: DataSourceType
    provider: str
    location: str
    license_status: str
    usage_permissions: str
    domain: CapabilityDomain
    language: str = "en"
    trust_level: float = Field(ge=0, le=1)
    acquisition_method: str
    eligibility: SourceEligibility
    eligibility_reason: str
    last_checked: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RawArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str
    source_id: str
    content_sha256: str
    byte_count: int = Field(ge=0)
    path: str
    acquisition_status: str
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TeacherRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str
    target_id: str
    kind: TeacherRequestKind
    provider: str
    model: str
    version: str | None = None
    configuration: dict[str, str | int | float | bool] = Field(default_factory=dict)
    prompt_sha256: str
    generation_run: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TeacherCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    request_id: str
    teacher_provider: str
    teacher_model: str
    teacher_version: str | None = None
    teacher_configuration: dict[str, str | int | float | bool] = Field(
        default_factory=dict
    )
    prompt_sha256: str
    generation_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generation_run: str
    source_target: str
    instruction: str
    input: str = ""
    output: str
    difficulty: int = Field(ge=1, le=10)
    domain: CapabilityDomain
    validation_status: TeacherValidationStatus = TeacherValidationStatus.PENDING


class TrainingRecord(BaseModel):
    """Canonical record retained separately from the corpus text release."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str
    target_id: str
    source_id: str
    instruction: str
    input: str = ""
    output: str
    domain: CapabilityDomain
    subdomain: str | None = None
    difficulty: int = Field(ge=1, le=10)
    difficulty_method: str
    reasoning_depth: int = Field(default=1, ge=1)
    required_knowledge: tuple[str, ...] = ()
    estimated_value: float = Field(ge=0, le=1)
    provenance: dict[str, str]
    validation_method: str
    validation_status: TeacherValidationStatus
    disposition: RecordDisposition
    teacher_candidate_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def canonical_text(self) -> str:
        parts = [f"Instruction: {self.instruction}"]
        if self.input:
            parts.append(f"Input: {self.input}")
        parts.append(f"Output: {self.output}")
        return "\n".join(parts)


class CurriculumPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    curriculum_id: str
    target_id: str
    requirement_id: str
    record_ids: tuple[str, ...]
    ordered_record_ids: tuple[str, ...]
    difficulty_distribution: dict[str, int]
    planned_examples: int
    actual_examples: int
    planned_tokens: int
    actual_tokens: int
    teacher_fraction: float = Field(ge=0, le=1)
    synthetic_fraction: float = Field(ge=0, le=1)
    source_derived_fraction: float = Field(ge=0, le=1)
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DatasetContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contribution_id: str
    dataset_release_id: str
    target_id: str
    experiment_id: str
    target_delta: float = Field(ge=-1, le=1)
    regression_rate: float = Field(ge=0, le=1)
    training_tokens: int = Field(ge=0)
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DataPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_id: str
    maximum_source_count: int = Field(default=20, ge=1)
    maximum_raw_bytes: int = Field(default=100_000_000, ge=1)
    maximum_records_per_release: int = Field(default=100_000, ge=1)
    maximum_teacher_calls: int = Field(default=0, ge=0)
    maximum_teacher_tokens: int = Field(default=0, ge=0)
    maximum_preprocessing_seconds: float = Field(default=3_600, gt=0)
    allow_teacher_generation: bool = False


class TeacherAdapter(Protocol):
    """An optional offline data-generation adapter; never used for evaluation."""

    def generate(
        self, request: TeacherRequest, prompt: str
    ) -> tuple[TeacherCandidate, ...]: ...


class DeterministicFixtureTeacher:
    """Test-only adapter proving the teacher interface without external execution."""

    def __init__(self, candidates: tuple[TeacherCandidate, ...]) -> None:
        self.candidates = candidates

    def generate(
        self, request: TeacherRequest, prompt: str
    ) -> tuple[TeacherCandidate, ...]:
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != request.prompt_sha256:
            raise ValueError(
                "Teacher request prompt hash does not match supplied prompt."
            )
        return self.candidates


class PhaseQStore:
    _CATEGORIES = frozenset(
        {
            "policies",
            "targets",
            "requirements",
            "sources",
            "raw",
            "teacher_requests",
            "teacher_candidates",
            "records",
            "curricula",
            "releases",
            "contributions",
        }
    )

    def __init__(self, root: Path) -> None:
        self.root = root / "phase_q"
        for category in self._CATEGORIES:
            (self.root / category).mkdir(parents=True, exist_ok=True)

    def write_immutable(self, category: str, identifier: str, value: BaseModel) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase Q artifact category: {category}")
        path = self.root / category / f"{identifier}.json"
        payload = value.model_dump_json(indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError(
                    f"Immutable Phase Q artifact already exists: {identifier}"
                )
            return
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)

    def reads(self, category: str, model: type[BaseModel]) -> tuple[BaseModel, ...]:
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown Phase Q artifact category: {category}")
        return tuple(
            model.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / category).glob("*.json"))
        )


def eligibility_for(source: DataSource) -> SourceEligibility:
    if source.license_status in {"", "UNKNOWN", "UNSPECIFIED"}:
        return SourceEligibility.UNKNOWN
    if source.usage_permissions not in {
        "TRAINING_PERMITTED",
        "USER_OWNED",
        "PUBLIC_DOMAIN",
    }:
        return SourceEligibility.REQUIRES_REVIEW
    return SourceEligibility.ELIGIBLE


def target_from_opportunity(opportunity: LearningOpportunity) -> LearningTarget:
    return LearningTarget(
        target_id=f"target-{opportunity.opportunity_id}"[:128],
        source="PHASE_P_LEARNING_OPPORTUNITY",
        domain=opportunity.domain,
        failure_cluster_id=opportunity.opportunity_id.removeprefix("opportunity-"),
        description=f"Target the measured {opportunity.domain.value} failure cluster: {', '.join(opportunity.task_ids)}.",
        importance=opportunity.importance,
        frequency=opportunity.frequency,
        confidence=opportunity.confidence,
        expected_generalization=opportunity.generalization_potential,
        estimated_training_cost=opportunity.estimated_compute_tokens,
        status=LearningTargetStatus.READY_FOR_DATA,
    )


def requirement_for(
    target: LearningTarget,
    *,
    minimum_records: int,
    required_example_kinds: tuple[str, ...],
    required_subdomains: tuple[str, ...] = (),
) -> DataRequirement:
    if not required_example_kinds:
        raise ValueError("A data requirement needs one or more explicit example kinds.")
    value = (
        target.importance
        * target.expected_generalization
        * target.confidence
        / target.estimated_training_cost
    )
    return DataRequirement(
        requirement_id=f"requirement-{target.target_id}"[:128],
        target_id=target.target_id,
        domain=target.domain,
        required_example_kinds=required_example_kinds,
        required_subdomains=required_subdomains,
        minimum_records=minimum_records,
        difficulty_bands=(2, 4, 6, 8),
        expected_value=value,
        rationale="target importance × expected generalization × confidence ÷ estimated training cost",
    )


class PhaseQEngine:
    """A controlled target-to-release bridge over the existing corpus pipeline."""

    def __init__(self, root: Path) -> None:
        self.root, self.store = root, PhaseQStore(root)

    def configure(self, policy: DataPolicy) -> DataPolicy:
        self.store.write_immutable("policies", policy.policy_id, policy)
        return policy

    def create_target(self, opportunity: LearningOpportunity) -> LearningTarget:
        target = target_from_opportunity(opportunity)
        self.store.write_immutable("targets", target.target_id, target)
        return target

    def create_requirement(
        self,
        target: LearningTarget,
        *,
        minimum_records: int,
        required_example_kinds: tuple[str, ...],
        required_subdomains: tuple[str, ...] = (),
    ) -> DataRequirement:
        requirement = requirement_for(
            target,
            minimum_records=minimum_records,
            required_example_kinds=required_example_kinds,
            required_subdomains=required_subdomains,
        )
        self.store.write_immutable(
            "requirements", requirement.requirement_id, requirement
        )
        return requirement

    def register_source(self, source: DataSource) -> DataSource:
        expected = eligibility_for(source)
        if source.eligibility is not expected:
            raise ValueError(
                f"Source eligibility must be {expected.value} from its license and permission declaration."
            )
        self.store.write_immutable("sources", source.source_id, source)
        return source

    def acquire_text(
        self, *, source: DataSource, content: str, policy: DataPolicy
    ) -> RawArtifact:
        if source.eligibility is not SourceEligibility.ELIGIBLE:
            raise PermissionError(
                "Only ELIGIBLE sources may be acquired into an automatic training path."
            )
        registered_sources = self.store.reads("sources", DataSource)
        if source.source_id not in {item.source_id for item in registered_sources}:
            raise ValueError("Sources must be registered before acquisition.")
        if len(registered_sources) > policy.maximum_source_count:
            raise ValueError("Data policy maximum source count has been reached.")
        raw = content.encode("utf-8")
        if len(raw) > policy.maximum_raw_bytes:
            raise ValueError("Source exceeds the immutable data policy byte limit.")
        digest = hashlib.sha256(raw).hexdigest()
        path = self.store.root / "raw" / f"{digest}.txt"
        artifact_id = f"raw-{digest[:24]}"
        existing = next(
            (
                item
                for item in self.store.reads("raw", RawArtifact)
                if item.artifact_id == artifact_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing.source_id != source.source_id
                or existing.content_sha256 != digest
            ):
                raise ValueError("Raw artifact identity collision across sources.")
            return existing
        if (
            sum(item.byte_count for item in self.store.reads("raw", RawArtifact))
            + len(raw)
            > policy.maximum_raw_bytes
        ):
            raise ValueError(
                "Data policy total raw-artifact byte limit has been reached."
            )
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        artifact = RawArtifact(
            artifact_id=artifact_id,
            source_id=source.source_id,
            content_sha256=digest,
            byte_count=len(raw),
            path=str(path.resolve()),
            acquisition_status="AVAILABLE",
        )
        self.store.write_immutable("raw", artifact.artifact_id, artifact)
        return artifact

    def validate_teacher_candidates(
        self,
        *,
        request: TeacherRequest,
        candidates: tuple[TeacherCandidate, ...],
        deterministic_verifier: Callable[[TeacherCandidate], bool],
    ) -> tuple[TrainingRecord, ...]:
        if not candidates:
            return ()
        self.store.write_immutable("teacher_requests", request.request_id, request)
        outputs = {normalize_text(candidate.output) for candidate in candidates}
        disagreement = len(outputs) > 1
        records: list[TrainingRecord] = []
        for candidate in candidates:
            if (
                candidate.request_id != request.request_id
                or candidate.prompt_sha256 != request.prompt_sha256
            ):
                raise ValueError(
                    "Teacher candidate does not belong to the immutable teacher request."
                )
            verified = not disagreement and deterministic_verifier(candidate)
            status = (
                TeacherValidationStatus.DISAGREEMENT
                if disagreement
                else (
                    TeacherValidationStatus.VERIFIED
                    if verified
                    else TeacherValidationStatus.REJECTED
                )
            )
            disposition = (
                RecordDisposition.REQUIRES_REVIEW
                if disagreement
                else (
                    RecordDisposition.ACCEPTED
                    if verified
                    else RecordDisposition.REJECTED
                )
            )
            record = TrainingRecord(
                record_id=f"record-{candidate.candidate_id}"[:128],
                target_id=request.target_id,
                source_id=f"teacher-{candidate.teacher_provider}-{candidate.teacher_model}"[
                    :128
                ],
                instruction=candidate.instruction,
                input=candidate.input,
                output=candidate.output,
                domain=candidate.domain,
                difficulty=candidate.difficulty,
                difficulty_method="TEACHER_DECLARED_NOT_GROUND_TRUTH",
                estimated_value=0.5,
                provenance={
                    "teacher_provider": candidate.teacher_provider,
                    "teacher_model": candidate.teacher_model,
                    "teacher_version": candidate.teacher_version or "UNSPECIFIED",
                    "generation_run": candidate.generation_run,
                    "prompt_sha256": candidate.prompt_sha256,
                },
                validation_method="INDEPENDENT_DETERMINISTIC_VERIFIER",
                validation_status=status,
                disposition=disposition,
                teacher_candidate_id=candidate.candidate_id,
            )
            self.store.write_immutable(
                "teacher_candidates", candidate.candidate_id, candidate
            )
            self.store.write_immutable("records", record.record_id, record)
            records.append(record)
        return tuple(records)

    def source_record(
        self,
        *,
        record_id: str,
        target: LearningTarget,
        source: DataSource,
        instruction: str,
        input: str,
        output: str,
        difficulty: int,
        subdomain: str | None = None,
        validation_method: str = "SOURCE_PROVENANCE_AND_STRUCTURAL_VALIDATION",
    ) -> TrainingRecord:
        if source.eligibility is not SourceEligibility.ELIGIBLE:
            raise PermissionError("Ineligible source cannot create a training record.")
        disposition = (
            RecordDisposition.ACCEPTED
            if all(normalize_text(value) for value in (instruction, output))
            else RecordDisposition.REJECTED
        )
        record = TrainingRecord(
            record_id=record_id,
            target_id=target.target_id,
            source_id=source.source_id,
            instruction=instruction,
            input=input,
            output=output,
            domain=target.domain,
            subdomain=subdomain,
            difficulty=difficulty,
            difficulty_method="DECLARED_SOURCE_DIFFICULTY",
            estimated_value=target.importance,
            provenance={
                "source_id": source.source_id,
                "location": source.location,
                "license": source.license_status,
            },
            validation_method=validation_method,
            validation_status=TeacherValidationStatus.VERIFIED
            if disposition is RecordDisposition.ACCEPTED
            else TeacherValidationStatus.REJECTED,
            disposition=disposition,
        )
        self.store.write_immutable("records", record.record_id, record)
        return record

    def generate_teacher_candidates(
        self,
        *,
        request: TeacherRequest,
        prompt: str,
        adapter: TeacherAdapter,
        policy: DataPolicy,
    ) -> tuple[TeacherCandidate, ...]:
        """Call an explicit offline teacher adapter within immutable policy limits.

        Returned values remain untrusted candidates and must be independently
        validated before a curriculum can consume them.
        """
        if not policy.allow_teacher_generation or policy.maximum_teacher_calls < 1:
            raise PermissionError(
                "The immutable data policy does not permit teacher generation."
            )
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != request.prompt_sha256:
            raise ValueError(
                "Teacher request prompt hash does not match supplied prompt."
            )
        prior = self.store.reads("teacher_requests", TeacherRequest)
        if len(prior) >= policy.maximum_teacher_calls and not any(
            item.request_id == request.request_id for item in prior
        ):
            raise ValueError("Data policy maximum teacher-call count has been reached.")
        self.store.write_immutable("teacher_requests", request.request_id, request)
        candidates = adapter.generate(request, prompt)
        if any(candidate.request_id != request.request_id for candidate in candidates):
            raise ValueError(
                "Teacher adapter returned a candidate for a different request."
            )
        return candidates

    @staticmethod
    def _assert_no_eval_overlap(
        records: tuple[TrainingRecord, ...], suite: CapabilitySuite
    ) -> None:
        evaluation_phrases = {
            normalize_text(item.prompt).lower() for item in suite.items
        } | {normalize_text(item.expected_answer).lower() for item in suite.items}
        contaminated = [
            record.record_id
            for record in records
            if any(
                phrase and phrase in normalize_text(record.canonical_text).lower()
                for phrase in evaluation_phrases
            )
        ]
        if contaminated:
            raise ValueError(
                f"Training records quarantined for direct evaluation overlap: {', '.join(contaminated)}"
            )

    def build_curriculum(
        self,
        *,
        target: LearningTarget,
        requirement: DataRequirement,
        records: tuple[TrainingRecord, ...],
        tokenizer: UltronTokenizer,
        curriculum_id: str | None = None,
    ) -> CurriculumPlan:
        accepted = tuple(
            record
            for record in records
            if record.target_id == target.target_id
            and record.disposition is RecordDisposition.ACCEPTED
            and record.validation_status is TeacherValidationStatus.VERIFIED
        )
        if len(accepted) < requirement.minimum_records:
            raise ValueError(
                "Validated records do not satisfy the immutable data requirement."
            )
        ordered = tuple(
            sorted(accepted, key=lambda item: (item.difficulty, item.record_id))
        )
        teacher = sum(
            record.teacher_candidate_id is not None for record in ordered
        ) / len(ordered)
        synthetic = sum(
            record.source_id.startswith("teacher-") for record in ordered
        ) / len(ordered)
        if (
            teacher > requirement.maximum_teacher_fraction
            or synthetic > requirement.maximum_synthetic_fraction
        ):
            raise ValueError(
                "Curriculum exceeds the requirement's teacher/synthetic fraction boundary."
            )
        tokens = sum(
            len(tokenizer.encode(record.canonical_text)) + 1 for record in ordered
        )
        plan = CurriculumPlan(
            curriculum_id=(curriculum_id or f"curriculum-{target.target_id}")[:128],
            target_id=target.target_id,
            requirement_id=requirement.requirement_id,
            record_ids=tuple(record.record_id for record in accepted),
            ordered_record_ids=tuple(record.record_id for record in ordered),
            difficulty_distribution={
                str(level): sum(record.difficulty == level for record in ordered)
                for level in sorted({record.difficulty for record in ordered})
            },
            planned_examples=requirement.minimum_records,
            actual_examples=len(ordered),
            planned_tokens=target.estimated_training_cost,
            actual_tokens=tokens,
            teacher_fraction=teacher,
            synthetic_fraction=synthetic,
            source_derived_fraction=1 - synthetic,
            status="READY_FOR_IMMUTABLE_RELEASE",
        )
        self.store.write_immutable("curricula", plan.curriculum_id, plan)
        return plan

    def release_training_data(
        self,
        *,
        release_id: str,
        target: LearningTarget,
        requirement: DataRequirement,
        curriculum: CurriculumPlan,
        records: tuple[TrainingRecord, ...],
        suite: CapabilitySuite,
        config: CorpusConfig | None = None,
    ) -> DatasetRelease:
        if (
            curriculum.target_id != target.target_id
            or curriculum.requirement_id != requirement.requirement_id
        ):
            raise ValueError(
                "Curriculum does not belong to the target and data requirement."
            )
        selected = tuple(
            record
            for record in records
            if record.record_id in curriculum.ordered_record_ids
        )
        self._assert_no_eval_overlap(selected, suite)
        documents = tuple(
            CorpusDocument(
                content=record.canonical_text,
                provenance=Provenance(
                    source_type=SourceType.CURATED_EXPERIENCE
                    if record.teacher_candidate_id
                    else SourceType.LOCAL_AUTHORED,
                    source_identifier=record.source_id,
                    source_group=record.source_id,
                    license=record.provenance.get(
                        "license",
                        "TEACHER_GENERATED_VALIDATED"
                        if record.teacher_candidate_id
                        else "UNSPECIFIED",
                    ),
                    license_evidence=record.validation_method,
                    declared_category=DocumentCategory.TECHNICAL
                    if record.domain
                    in {
                        CapabilityDomain.PROGRAMMING,
                        CapabilityDomain.TECHNICAL_KNOWLEDGE,
                    }
                    else DocumentCategory.GENERAL,
                ),
            )
            for record in selected
        )
        release = CorpusPipeline(
            self.root,
            config
            or CorpusConfig(
                validation_percent=20,
                test_percent=20,
                split_strategy="source_group_balanced",
            ),
        ).release(release_id, documents)
        self.store.write_immutable("releases", release_id, release)
        return release

    def record_contribution(
        self, contribution: DatasetContribution
    ) -> DatasetContribution:
        self.store.write_immutable(
            "contributions", contribution.contribution_id, contribution
        )
        return contribution
