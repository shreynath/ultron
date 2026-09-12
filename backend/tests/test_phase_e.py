from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")

from app.brain.cognitive_state import CognitiveStage, CognitiveStateStore, WorkingItem
from app.foundation.evaluation import (
    FoundationEvaluator,
    PromotionDecision,
    PromotionPolicy,
    suite_from_text,
)
from app.foundation.model import UltronModelConfig
from app.foundation.scaling import Feasibility, ScalingResult, recommend
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


def test_cognitive_state_is_persisted_with_bounded_working_memory(tmp_path):
    store = CognitiveStateStore(tmp_path / "state.sqlite3")
    task = store.create("Verify a local task")
    store.transition(task.task_id, CognitiveStage.UNDERSTANDING, component="executive", summary="Task classified.")
    updated = store.put_working_item(task.task_id, WorkingItem(key="source", value="local test", source="test", priority=4))
    assert updated.stage is CognitiveStage.UNDERSTANDING
    assert store.get(task.task_id).working_memory[0].key == "source"
    assert store.self_model(model_version="unknown", available_tools=("web_research",), capabilities=("retrieval",), limitations=("no consciousness claim",)).active_task_count == 1


def test_scaling_recommendation_is_measured_and_explained():
    first = ScalingResult(experiment_id="small", created_at="2026-01-01T00:00:00Z", model={"parameter_estimate": 1_000_000}, training={"tokens_per_second": 100.0}, memory={"mlx_peak_bytes": 5, "swap_delta_bytes": 0}, hardware={}, status=Feasibility.FEASIBLE)
    second = ScalingResult(experiment_id="large", created_at="2026-01-01T00:00:00Z", model={"parameter_estimate": 10_000_000}, training={"tokens_per_second": 20.0}, memory={"mlx_peak_bytes": 10, "swap_delta_bytes": 0}, hardware={}, status=Feasibility.FEASIBLE)
    report = recommend((first, second))
    assert report[-1].recommendation.value == "RECOMMENDED"
    assert report[-1].tokens_per_hour_estimate == 72_000.0


def test_evaluation_compares_checkpoint_without_promoting(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = "ULTRON evaluates held out local text with a deterministic checkpoint. " * 50
    corpus.write_text(text, encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(vocab_size=320))
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, max_sequence_length=8, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32)
    result = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=TrainingConfig(run_id="eval", steps=2, batch_size=1, checkpoint_every=2, validation_every=2))
    suite = suite_from_text("local-token-loss", "1", corpus, max_tokens=8)
    evaluator = FoundationEvaluator()
    report = evaluator.evaluate(Path(result.checkpoint_path), tokenizer, model, suite)
    comparison = evaluator.compare(report, report)
    decision, _ = evaluator.promotion_decision(comparison, Path(result.checkpoint_path), PromotionPolicy())
    assert decision is PromotionDecision.REJECT
