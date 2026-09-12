from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

pytest.importorskip("mlx")

from app.foundation.capabilities import (
    Capability,
    EvaluationTask,
    TaskKind,
    execute_task,
)
from app.foundation.generation import (
    GenerationConfig,
    GenerationStatus,
    TerminationReason,
    UltronGenerator,
)
from app.foundation.model import UltronModelConfig
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer
from app.foundation.training import FoundationTrainer, TrainingConfig


@pytest.fixture()
def generator(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = ("local generation verifies an owned checkpoint and tokenizer.\n" * 80)
    corpus.write_text(text, encoding="utf-8")
    tokenizer = UltronTokenizer.train(corpus, tmp_path / "tokenizer", TokenizerConfig(vocab_size=320))
    model = UltronModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_heads=4, num_layers=1, intermediate_size=32, max_sequence_length=16)
    result = FoundationTrainer(tmp_path).train(model_config=model, tokenizer=tokenizer, train_text=text, validation_text=text, config=TrainingConfig(run_id="phase-h-generation", model_id="phase-h-generation", steps=2, batch_size=1, checkpoint_every=2, validation_every=2))
    return UltronGenerator(Path(result.checkpoint_path), tokenizer, model)


def test_greedy_generation_is_deterministic_and_bounded(generator):
    first = generator.generate("local", GenerationConfig(max_new_tokens=3, temperature=0))
    second = generator.generate("local", GenerationConfig(max_new_tokens=3, temperature=0))
    assert first.status is GenerationStatus.COMPLETED
    assert first.generated_token_ids == second.generated_token_ids
    assert first.generated_tokens <= 3


def test_generation_supports_eos_cancellation_and_context_limits(generator, monkeypatch):
    monkeypatch.setattr(generator, "_select", lambda *_: generator.tokenizer.processor.eos_id())
    eos = generator.generate("local", GenerationConfig(max_new_tokens=3))
    assert eos.termination_reason is TerminationReason.EOS
    cancelled = Event()
    cancelled.set()
    assert generator.generate("local", GenerationConfig(max_new_tokens=3), cancel=cancelled).status is GenerationStatus.CANCELLED
    long_prompt = " ".join(["local"] * 100)
    with pytest.raises(ValueError, match="context"):
        generator.generate(long_prompt, GenerationConfig(max_new_tokens=1))


def test_sampling_is_seeded_and_invalid_decoding_is_rejected(generator):
    first = generator.generate("local", GenerationConfig(max_new_tokens=3, temperature=0.8, top_k=8, seed=12))
    second = generator.generate("local", GenerationConfig(max_new_tokens=3, temperature=0.8, top_k=8, seed=12))
    assert first.generated_token_ids == second.generated_token_ids
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig(temperature=0, top_k=2)


def test_generation_rejects_incompatible_architecture(generator):
    incompatible = UltronModelConfig(vocab_size=generator.tokenizer.vocab_size, hidden_size=32, num_heads=4, num_layers=1, intermediate_size=64, max_sequence_length=16)
    with pytest.raises(ValueError, match="architecture"):
        UltronGenerator(generator.checkpoint, generator.tokenizer, incompatible)


def test_python_capability_task_runs_inside_macos_sandbox():
    task = EvaluationTask(task_id="sandbox", capability=Capability.CODING, kind=TaskKind.PYTHON, prompt="print", expected="4")
    result = execute_task(task, "print(4)")
    assert result.outcome == "PASS", result.detail


def test_python_capability_sandbox_blocks_network_access():
    task = EvaluationTask(task_id="sandbox-network", capability=Capability.CODING, kind=TaskKind.PYTHON, prompt="network", expected="never")
    result = execute_task(task, "import socket\nsocket.create_connection(('127.0.0.1', 1), timeout=0.1)")
    assert result.outcome == "RUNTIME_ERROR"
