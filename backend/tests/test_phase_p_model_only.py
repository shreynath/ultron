from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    Provenance,
    SourceType,
)
from app.foundation.model import UltronModelConfig
from app.foundation.phase_p import (
    CapabilityDomain,
    CapabilityItem,
    EvaluationPartition,
    ItemRole,
    PhasePEngine,
    create_capability_suite,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


def test_phase_p_measures_a_locally_trained_checkpoint_without_a_teacher(tmp_path: Path) -> None:
    documents = [
        CorpusDocument(
            content=f"ULTRON owned training record {index}. The system learns only from this locally authored fixture sequence {index}.",
            provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier=f"phase-p-{index}", source_group=f"phase-p-source-{index}", license="CC0-1.0", license_evidence="test fixture"),
        )
        for index in range(12)
    ]
    release = CorpusPipeline(tmp_path, CorpusConfig(validation_percent=20, test_percent=20)).release("phase-p-model-only-release", documents)
    corpus = tmp_path / "tokenizer-corpus.txt"
    corpus.write_text("\n".join(item.content for item in documents), encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(version="phase-p-tokenizer", vocab_size=320))
    model = UltronModelConfig(model_name="ultron-phase-p-fixture", version="v1", vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    train = (tmp_path / "corpora" / "releases" / release.release_id / "train.jsonl").read_text(encoding="utf-8")
    validation = (tmp_path / "corpora" / "releases" / release.release_id / "validation.jsonl").read_text(encoding="utf-8")
    config = TrainingConfig(run_id="phase-p-model-only-run", model_id=model.model_name, steps=5, batch_size=1, checkpoint_every=5, validation_every=1, maximum_wall_seconds=0.000001, dataset_release_id=release.release_id, dataset_manifest_sha256=release.manifest_sha256 or "")
    trained = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=train, validation_text=validation, config=config)
    assert trained.steps_completed < config.steps
    assert (trained.early_stop_reason or "").startswith("WALL_CLOCK_LIMIT:")

    suite = create_capability_suite(
        tmp_path / "owned-capability-suite.json", suite_id="phase-p-owned-capability", version="v1", partition=EvaluationPartition.TEST,
        items=(CapabilityItem(task_id="owned-task-1", domain=CapabilityDomain.REASONING, partition=EvaluationPartition.TEST, role=ItemRole.TARGET, prompt="ULTRON", expected_answer="a deliberately unlikely exact fixture answer", difficulty=3, provenance={"source": "owned test suite"}),),
    )
    run = PhasePEngine(tmp_path).evaluate_model_only(checkpoint=Path(trained.checkpoint_path), tokenizer=tokenizer, model_config=model, suite=suite, candidate_id="phase-p-owned-candidate", max_new_tokens=4)

    assert run.runtime_dependency == "NONE_LOCAL_ULTRON_CHECKPOINT_ONLY"
    assert run.checkpoint_id
    assert len(run.results) == 1
    assert (tmp_path / "phase_p" / "runs" / f"{run.run_id}.json").is_file()
