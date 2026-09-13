from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    EvaluationPartition,
    LearningOpportunity,
    create_capability_suite,
)
from app.foundation.phase_q import (
    DataPolicy,
    DataSource,
    DataSourceType,
    DeterministicFixtureTeacher,
    PhaseQEngine,
    RecordDisposition,
    SourceEligibility,
    TeacherCandidate,
    TeacherRequest,
    TeacherRequestKind,
    TeacherValidationStatus,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


def _opportunity() -> LearningOpportunity:
    return LearningOpportunity(
        opportunity_id="opportunity-recursion", source_run_id="candidate-run", domain=CapabilityDomain.PROGRAMMING,
        task_ids=("recursion-failure",), importance=0.9, frequency=0.6, expected_improvement=0.2,
        generalization_potential=0.7, confidence=0.8, estimated_compute_tokens=256, priority=0.0002,
        rationale="fixture weakness",
    )


def _source(index: int, *, eligibility: SourceEligibility = SourceEligibility.ELIGIBLE, license_status: str = "CC0-1.0") -> DataSource:
    return DataSource(
        source_id=f"owned-source-{index}", source_type=DataSourceType.USER_OWNED, provider="fixture-owner",
        location=f"local://owned/{index}", license_status=license_status,
        usage_permissions="USER_OWNED", domain=CapabilityDomain.PROGRAMMING, trust_level=1.0,
        acquisition_method="LOCAL_IMMUTABLE_FIXTURE", eligibility=eligibility,
        eligibility_reason="Owner granted training permission." if eligibility is SourceEligibility.ELIGIBLE else "License is unknown.",
    )


def _tokenizer(tmp_path: Path) -> UltronTokenizer:
    corpus = tmp_path / "tokenizer.txt"
    corpus.write_text("recursive function base case stack tracing debugging algorithm implementation examples with varied independent language", encoding="utf-8")
    return UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(version="phase-q-tokenizer", vocab_size=320))


def _suite(tmp_path: Path):
    return create_capability_suite(
        tmp_path / "suite.json", suite_id="phase-q-heldout", version="v1", partition=EvaluationPartition.TEST,
        items=(CapabilityItem(task_id="heldout-recursion", domain=CapabilityDomain.PROGRAMMING, partition=EvaluationPartition.TEST, prompt="heldout recursion probe", expected_answer="do not train this answer", difficulty=5, provenance={"source": "fixture"}),),
    )


def test_phase_q_builds_an_immutable_eligible_non_contaminating_release(tmp_path: Path) -> None:
    engine = PhaseQEngine(tmp_path)
    policy = engine.configure(DataPolicy(policy_id="phase-q-policy", maximum_raw_bytes=10_000, maximum_records_per_release=20))
    target = engine.create_target(_opportunity())
    requirement = engine.create_requirement(target, minimum_records=6, required_example_kinds=("worked implementation", "debugging"), required_subdomains=("recursion",))
    tokenizer, suite = _tokenizer(tmp_path), _suite(tmp_path)
    records = []
    for index in range(12):
        source = engine.register_source(_source(index))
        first = engine.acquire_text(source=source, content=f"Owned raw recursion source {index} with a distinct explanation and approved license.", policy=policy)
        second = engine.acquire_text(source=source, content=f"Owned raw recursion source {index} with a distinct explanation and approved license.", policy=policy)
        assert first == second
        records.append(engine.source_record(record_id=f"recursion-record-{index}", target=target, source=source, instruction=f"Implement recursion exercise {index} with a base case and trace.", input=f"input-{index}", output=f"Use a base case before recursive call {index}; verify the trace.", difficulty=2 + index % 4, subdomain="recursion"))
    curriculum = engine.build_curriculum(target=target, requirement=requirement, records=tuple(records), tokenizer=tokenizer)
    release = engine.release_training_data(release_id="phase-q-recursion-v1", target=target, requirement=requirement, curriculum=curriculum, records=tuple(records), suite=suite)
    assert release.document_count == 12
    assert (tmp_path / "corpora" / "releases" / release.release_id / "manifest.json").is_file()
    assert engine.store.reads("curricula", type(curriculum))[0].actual_tokens == curriculum.actual_tokens


def test_phase_q_quarantines_direct_evaluation_overlap_and_unknown_license(tmp_path: Path) -> None:
    engine, suite = PhaseQEngine(tmp_path), _suite(tmp_path)
    target = engine.create_target(_opportunity())
    unknown = _source(99, eligibility=SourceEligibility.UNKNOWN, license_status="UNKNOWN")
    with pytest.raises(ValueError, match="must be UNKNOWN"):
        engine.register_source(unknown.model_copy(update={"eligibility": SourceEligibility.ELIGIBLE}))
    source = engine.register_source(_source(100))
    record = engine.source_record(record_id="contaminated-record", target=target, source=source, instruction="heldout recursion probe", input="", output="safe output but prompt overlap", difficulty=3)
    requirement = engine.create_requirement(target, minimum_records=1, required_example_kinds=("exercise",))
    curriculum = engine.build_curriculum(target=target, requirement=requirement, records=(record,), tokenizer=_tokenizer(tmp_path))
    with pytest.raises(ValueError, match="quarantined"):
        engine.release_training_data(release_id="phase-q-contaminated-v1", target=target, requirement=requirement, curriculum=curriculum, records=(record,), suite=suite)


def test_teacher_candidates_require_independent_validation_and_disagreement_is_not_training_data(tmp_path: Path) -> None:
    engine = PhaseQEngine(tmp_path)
    target = engine.create_target(_opportunity())
    prompt = "Create a verified recursion example."
    request = TeacherRequest(request_id="teacher-request-1", target_id=target.target_id, kind=TeacherRequestKind.EXAMPLE, provider="fixture-provider", model="fixture-teacher", version="v1", prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), generation_run="fixture-run")
    first = TeacherCandidate(candidate_id="teacher-candidate-a", request_id=request.request_id, teacher_provider="fixture-provider", teacher_model="fixture-teacher", teacher_version="v1", prompt_sha256=request.prompt_sha256, generation_run="fixture-run", source_target=target.target_id, instruction="Write a recursive sum.", output="Use a base case and return n plus recursive_sum(n - 1).", difficulty=4, domain=CapabilityDomain.PROGRAMMING)
    second = first.model_copy(update={"candidate_id": "teacher-candidate-b", "output": "Use iteration instead."})
    fixture = DeterministicFixtureTeacher((first, second))
    policy = engine.configure(DataPolicy(policy_id="teacher-fixture-policy", allow_teacher_generation=True, maximum_teacher_calls=1))
    candidates = engine.generate_teacher_candidates(request=request, prompt=prompt, adapter=fixture, policy=policy)
    assert candidates == (first, second)
    records = engine.validate_teacher_candidates(request=request, candidates=candidates, deterministic_verifier=lambda _: True)
    assert {record.disposition for record in records} == {RecordDisposition.REQUIRES_REVIEW}
    assert {record.validation_status for record in records} == {TeacherValidationStatus.DISAGREEMENT}
    assert engine.store.reads("teacher_requests", TeacherRequest)[0].request_id == request.request_id
