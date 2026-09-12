from __future__ import annotations

import asyncio

import pytest

from app.brain.research import SafeWebClient
from app.config import settings
from app.local_intelligence.core import (
    LocalModelOrchestrator,
    LocalModelRegistry,
    LocalModelUnavailable,
    ModelCapability,
    ModelRecord,
    ModelStatus,
)
from app.local_intelligence.memory import (
    FeedbackEvent,
    FeedbackKind,
    LearningEventStore,
    LearningExampleStatus,
    LocalMemory,
    LocalMemoryStore,
    MemoryKind,
)
from app.local_intelligence.network import NetworkToolRequired


def test_registry_hashes_and_promotes_local_artifact(tmp_path):
    async def run():
        root = tmp_path / "ultron"
        weights = root / "models" / "base" / "fixture"
        weights.mkdir(parents=True)
        (weights / "config.json").write_text("{}")
        registry = LocalModelRegistry(root)
        record = await registry.register(ModelRecord(
            model_id="fixture-v1", base_model="fixture", version="1", location=str(weights),
            runtime="mlx", status=ModelStatus.CANDIDATE,
        ))
        assert record.artifact_sha256
        promoted = await registry.promote("fixture-v1")
        assert promoted.status is ModelStatus.PROMOTED
        assert registry.active(ModelCapability.GENERAL).model_id == "fixture-v1"
    asyncio.run(run())


def test_orchestrator_fails_locally_when_no_model_is_promoted(tmp_path):
    async def run():
        orchestrator = LocalModelOrchestrator(registry=LocalModelRegistry(tmp_path / "ultron"))
        with pytest.raises(LocalModelUnavailable, match="No promoted local model"):
            await orchestrator.generate(system_prompt="sys", user_message="hello")
    asyncio.run(run())


def test_local_memory_is_namespaced_retrievable_and_recoverably_forgotten(tmp_path):
    store = LocalMemoryStore(tmp_path / "ultron")
    saved = store.save(LocalMemory(kind=MemoryKind.USER, content="Prefers concise technical answers.", source="explicit_user"))
    assert store.retrieve("technical concise", kinds=(MemoryKind.USER,)) == (saved,)
    forgotten = store.forget(saved.memory_id)
    assert forgotten is not None
    assert store.retrieve("technical concise") == ()


def test_feedback_requires_curation_before_dataset_creation(tmp_path):
    events = LearningEventStore(tmp_path / "ultron")
    feedback = events.record(FeedbackEvent(kind=FeedbackKind.CORRECTION, original_output="wrong", corrected_output="right"))
    assert feedback.status is LearningExampleStatus.REVIEW_REQUIRED
    with pytest.raises(ValueError, match="No curated"):
        events.training_dataset(dataset_id="v1")
    events.curate(feedback.feedback_id, approve=True)
    dataset = events.training_dataset(dataset_id="v1")
    assert '"content":"right"' in dataset.read_text()


def test_memory_refuses_credentials(tmp_path):
    store = LocalMemoryStore(tmp_path / "ultron")
    with pytest.raises(ValueError, match="Sensitive credentials"):
        store.save(LocalMemory(kind=MemoryKind.SEMANTIC, content="api_key=supersecretvalue", source="test"))


def test_local_only_blocks_explicit_network_tools(monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "local_network_mode", "local_only")
        with pytest.raises(NetworkToolRequired, match="NETWORK TOOL REQUIRED"):
            await SafeWebClient().search("current weather", 1)
    asyncio.run(run())
