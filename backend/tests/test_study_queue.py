import pytest

from app.local_intelligence.study_queue import (
    AutonomyTier,
    CalibrationGrade,
    CollisionRisk,
    LearningDecision,
    ReviewStatus,
    RunnerConfig,
    StudyQueueController,
    StudyQueueStore,
    StudyTarget,
    SyntheticTargetGenerator,
)

SOURCE = "RFC fixture: The DNS message identifier is a 16 bit unsigned integer assigned by the query program."


def target() -> StudyTarget:
    return StudyTarget(
        subject="DNS message identifier width",
        baseline_question="How many bits is a DNS message identifier?",
        paraphrase_question="What is the bit width of the identifier in a DNS query?",
        expected_answer="16 bit",
        candidate_information="A DNS message identifier is a 16 bit unsigned integer.",
        source_url="https://www.rfc-editor.org/rfc/rfc1035.txt",
        source_credibility="RFC Editor publication",
    )


def test_study_queue_requires_supervision_then_demonstrates_external_memory_learning(tmp_path):
    controller = StudyQueueController(tmp_path / "ultron", fetcher=lambda _: SOURCE)
    controller.queue.enqueue(target())
    with pytest.raises(PermissionError, match="supervised"):
        controller.run_next()
    event = controller.run_next(supervisor_approved=True)
    assert event is not None
    assert event.decision is LearningDecision.LEARNING_DEMONSTRATED
    assert not event.pre_test_result.correct
    assert event.post_test_result and event.post_test_result.correct
    assert event.generalization_test_result and event.generalization_test_result.correct
    assert event.source_provenance["sha256"]
    assert (tmp_path / "ultron" / "learning" / "study_sources" / f"{event.source_provenance['sha256']}.txt").exists()
    assert StudyQueueStore(tmp_path / "ultron").events() == (event,)


def test_study_queue_refuses_source_that_does_not_support_ground_truth(tmp_path):
    controller = StudyQueueController(tmp_path / "ultron", fetcher=lambda _: "Unrelated external material.")
    controller.queue.enqueue(target())
    event = controller.run_next(supervisor_approved=True)
    assert event is not None
    assert event.decision is LearningDecision.UNKNOWN
    assert event.validation_status == "FAILED"


def test_stage_one_needs_low_collision_human_graded_round_and_marks_review(tmp_path):
    root = tmp_path / "ultron"
    controller = StudyQueueController(root)
    events = []
    for index in range(10):
        item = SyntheticTargetGenerator.generate(index)
        controller.queue.enqueue(item)
        assert controller.register_synthetic_collision_clearance(item).risk is CollisionRisk.SYNTHETIC
        event = controller.run_next(supervisor_approved=True)
        assert event is not None
        events.append(event)
    round_ = StudyQueueStore(root).record_calibration_round(
        stage=AutonomyTier.STAGE_1,
        grades=tuple(CalibrationGrade(event_id=event.event_id, human_decision=event.decision, reviewer_id="test-human") for event in events),
    )
    assert round_.passed and round_.agreement == 1.0
    assert round_.wilson_interval[0] < 1.0
    assert StudyQueueStore(root).tier() is AutonomyTier.STAGE_1
    extra = SyntheticTargetGenerator.generate(99)
    controller.queue.enqueue(extra)
    controller.register_synthetic_collision_clearance(extra)
    event = controller.run_available()
    assert event and event[0].review_status is ReviewStatus.REQUIRED


def test_public_collision_probe_excludes_known_answer_from_calibration(tmp_path):
    controller = StudyQueueController(tmp_path / "ultron", fetcher=lambda _: SOURCE)
    item = target()
    controller.queue.enqueue(item)
    probe = controller.assess_collision(item, prober=lambda _: "The identifier has 16 bit width.", prober_id="fixture-local-model")
    assert probe.risk is CollisionRisk.HIGH


def test_circuit_breaker_pauses_after_configured_terminal_failure(tmp_path):
    root = tmp_path / "ultron"
    controller = StudyQueueController(root, fetcher=lambda _: "unrelated")
    controller.queue.set_runner_config(RunnerConfig(consecutive_terminal_failures=1))
    controller.queue.enqueue(target())
    event = controller.run_next(supervisor_approved=True)
    assert event and event.decision is LearningDecision.UNKNOWN
    assert controller.queue.circuit_breaker().tripped
    with pytest.raises(PermissionError, match="circuit breaker"):
        controller.run_next(supervisor_approved=True)
