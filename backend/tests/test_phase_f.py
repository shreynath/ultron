from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.brain.acquisition import (
    AcquisitionEngine,
    AcquisitionItem,
    AcquisitionItemStatus,
    AcquisitionJob,
    CuratedSource,
)
from app.brain.research import RetrievalStatus, SourceRecord
from app.foundation.data import CorpusConfig, CorpusDocument, CorpusPipeline, Provenance
from app.foundation.data import SourceType as CorpusSourceType


class _FakeWebClient:
    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, url: str):
        self.calls += 1
        source = SourceRecord(url=url, domain="example.test", retrieved_at=datetime.now(UTC), content_hash="a" * 64, retrieval_status=RetrievalStatus.RETRIEVED)
        return source, "A sufficiently long public document. " * 100


def test_acquisition_caches_explicit_source_and_creates_provenance(tmp_path):
    async def run() -> None:
        client = _FakeWebClient()
        engine = AcquisitionEngine(tmp_path, client=client)
        item = AcquisitionItem(source=CuratedSource(url="https://example.test/document", source_identifier="fixture", license="CC0-1.0", category="TECHNICAL", minimum_characters=100))
        result = await engine.run(AcquisitionJob(items=[item]))
        assert result.items[0].status is AcquisitionItemStatus.FETCHED
        documents = engine.corpus_documents(result)
        assert documents[0].provenance.source_type is CorpusSourceType.PUBLIC_DATASET
        rerun = await engine.run(AcquisitionJob(items=[item]))
        assert rerun.items[0].status is AcquisitionItemStatus.CACHED
    asyncio.run(run())


def test_source_identifier_split_does_not_split_a_source_across_partitions(tmp_path):
    documents = [
        CorpusDocument(content=f"Long source one section {index} contains enough text for deterministic processing and separation.", provenance=Provenance(source_type=CorpusSourceType.LOCAL_AUTHORED, source_identifier="source-one", license="CC0-1.0"))
        for index in range(3)
    ] + [
        CorpusDocument(content=f"Long source two section {index} contains enough text for deterministic processing and separation.", provenance=Provenance(source_type=CorpusSourceType.LOCAL_AUTHORED, source_identifier="source-two", license="CC0-1.0"))
        for index in range(3)
    ]
    release = CorpusPipeline(tmp_path, CorpusConfig(validation_percent=50)).release("source-split-v1", documents)
    release_dir = tmp_path / "corpora" / "releases" / release.release_id
    train_sources = {line.split('"source_identifier":"')[1].split('"')[0] for line in (release_dir / "train.jsonl").read_text().splitlines()}
    validation_sources = {line.split('"source_identifier":"')[1].split('"')[0] for line in (release_dir / "validation.jsonl").read_text().splitlines()}
    assert not train_sources & validation_sources
