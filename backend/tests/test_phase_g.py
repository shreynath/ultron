from __future__ import annotations

from datetime import UTC, datetime

from app.brain.acquisition import (
    AcquisitionEngine,
    AcquisitionItem,
    AcquisitionItemStatus,
    CuratedSource,
)
from app.brain.cognitive_execution import (
    CognitiveLoop,
    CognitiveTask,
    CognitiveTaskStatus,
    CognitiveTaskStore,
    ModelOutput,
    ModelOutputKind,
)
from app.brain.research import RetrievalStatus, SourceRecord
from app.foundation.capabilities import (
    Capability,
    CapabilityStatus,
    EvaluationTask,
    TaskKind,
    capability_matrix,
    create_suite,
    evaluate_outputs,
)
from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    Provenance,
    SourceType,
)
from app.foundation.model import UltronModelConfig
from app.foundation.scaling import ResourceRisk, resource_preflight


def test_source_group_balanced_split_keeps_groups_isolated_and_seals_test(tmp_path):
    documents = []
    for group, size in (("trainable", 1000), ("validation", 800), ("testable", 600), ("small", 100)):
        terms = [f"term{chr(97 + index % 26)}{chr(97 + (index // 26) % 26)}" for index in range(size)]
        documents.append(CorpusDocument(content=f"{group} source document " + " ".join(terms), provenance=Provenance(source_type=SourceType.PUBLIC_DATASET, source_identifier=group, source_group=group, license="CC0-1.0", license_evidence="fixture")))
    release = CorpusPipeline(tmp_path, CorpusConfig(split_strategy="source_group_balanced", validation_percent=25, test_percent=20)).release("phase-g-split-v1", documents)
    directory = tmp_path / "corpora" / "releases" / release.release_id
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = {line.split('"source_group":"')[1].split('"')[0] for line in (directory / f"{split}.jsonl").read_text().splitlines()}
    assert splits["train"].isdisjoint(splits["validation"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["validation"].isdisjoint(splits["test"])
    assert release.test_document_count > 0


def test_preflight_is_an_estimate_not_a_feasibility_claim():
    model = UltronModelConfig(vocab_size=320, hidden_size=64, num_heads=4, num_layers=2, intermediate_size=192, max_sequence_length=64)
    report = resource_preflight(model, batch_size=1, sequence_length=64, memory_bytes=24 * 1024**3)
    assert report.risk is ResourceRisk.SAFE
    assert report.total_bytes_estimate > report.parameter_bytes


def test_capability_suite_is_immutable_and_reports_insufficient_evidence(tmp_path):
    tasks = (EvaluationTask(task_id="math-1", capability=Capability.MATH, kind=TaskKind.EXACT, prompt="ignored", expected="4"),)
    suite = create_suite(tmp_path / "suite.json", suite_id="phase-g", version="1", tasks=tasks, tokenizer_version="0.0.1", source_manifest_hash="a" * 64, preprocessing_version="test")
    baseline = evaluate_outputs(suite, candidate_id="baseline", outputs={})
    candidate = evaluate_outputs(suite, candidate_id="candidate", outputs={"math-1": "4"})
    assert candidate.results[0].outcome == "PASS"
    assert capability_matrix(baseline, candidate)["math"] is CapabilityStatus.INSUFFICIENT_EVIDENCE


class _ContractModel:
    def produce(self, kind: ModelOutputKind, task: CognitiveTask) -> ModelOutput:
        return ModelOutput(kind=kind, content=f"{kind.value} for {task.goal}", tool_name="inspect_task" if kind is ModelOutputKind.TOOL_REQUEST else None)


class _LocalTool:
    def execute(self, name: str, arguments: dict[str, str | int | float | bool]) -> str:
        assert name == "inspect_task"
        return "controlled local observation"


def test_bounded_cognitive_contract_persists_a_completed_task(tmp_path):
    store = CognitiveTaskStore(tmp_path / "tasks.sqlite3")
    completed = CognitiveLoop(store, _ContractModel(), tool_executor=_LocalTool()).run(CognitiveTask(goal="Run a controlled contract test", required_tools=("inspect_task",)))
    assert completed.status is CognitiveTaskStatus.COMPLETED
    assert store.get(completed.task_id).results


def test_acquisition_source_registry_persists_explainable_entry(tmp_path):
    source = CuratedSource(url="https://example.test/data", source_identifier="fixture-source", license="CC0-1.0", license_evidence="fixture-license", category="TECHNICAL")
    item = AcquisitionItem(source=source, status=AcquisitionItemStatus.FETCHED, content_hash="a" * 64, extracted_characters=5000, cache_path=str(tmp_path / "cache.txt"), source_record=SourceRecord(url=source.url, domain="example.test", retrieved_at=datetime.now(UTC), content_hash="a" * 64, retrieval_status=RetrievalStatus.RETRIEVED))
    engine = AcquisitionEngine(tmp_path)
    engine.store.register_source(engine.source_registry_entry(item))
    entry = engine.store.sources()[0]
    assert entry.source_id == "fixture-source"
    assert entry.quality.score > 0
