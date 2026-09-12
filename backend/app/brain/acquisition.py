"""Bounded, resumable acquisition of explicitly curated public documents."""
from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.brain.research import RetrievalStatus, SafeWebClient, SourceRecord, SourceType
from app.foundation.data import CorpusDocument, Provenance
from app.foundation.data import SourceType as CorpusSourceType


class AcquisitionStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class AcquisitionItemStatus(str, Enum):
    PENDING = "PENDING"
    CACHED = "CACHED"
    FETCHED = "FETCHED"
    FAILED = "FAILED"


class CuratedSource(BaseModel):
    """A human/configuration-selected source; discovery snippets are never input."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: str
    source_identifier: str
    license: str
    category: str
    minimum_characters: int = Field(default=2_000, ge=100)
    source_type: SourceType = SourceType.UNKNOWN
    title: str | None = None
    publisher: str | None = None
    license_evidence: str | None = None
    source_concentration_group: str | None = None
    parser_version: str = "ultron.web-text.v1"
    acquisition_version: str = "ultron.acquisition.v2"


class SourceQuality(BaseModel):
    """Explainable acquisition quality factors, never a truth/confidence score."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    authority: float = Field(ge=0, le=1)
    provenance_completeness: float = Field(ge=0, le=1)
    structural_quality: float = Field(ge=0, le=1)
    machine_readability: float = Field(ge=0, le=1)
    stability: float = Field(ge=0, le=1)
    licensing_clarity: float = Field(ge=0, le=1)
    duplication_risk: float = Field(ge=0, le=1)
    noise_risk: float = Field(ge=0, le=1)
    content_density: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)


class SourceRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    canonical_url: str
    title: str | None = None
    publisher: str | None = None
    source_type: str
    category: str
    license: str
    license_evidence: str | None = None
    retrieved_at: datetime
    published_at: datetime | None = None
    content_hash: str
    retrieval_status: str
    parser_version: str
    acquisition_version: str
    provenance: dict[str, str]
    quality: SourceQuality
    source_concentration_group: str


class AcquisitionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: CuratedSource
    status: AcquisitionItemStatus = AcquisitionItemStatus.PENDING
    attempts: int = 0
    retry_count: int = 0
    last_error: str | None = None
    fetched_at: datetime | None = None
    content_hash: str | None = None
    extracted_characters: int = 0
    cache_path: str | None = None
    source_record: SourceRecord | None = None


class AcquisitionJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: UUID = Field(default_factory=uuid4)
    status: AcquisitionStatus = AcquisitionStatus.QUEUED
    items: list[AcquisitionItem]
    max_retries: int = Field(default=2, ge=0, le=5)
    backoff_seconds: float = Field(default=0.1, ge=0, le=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    requests: int = 0
    bytes_downloaded: int = 0
    elapsed_seconds: float = 0.0


class AcquisitionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "acquisition" / "acquisition.sqlite3"
        self.cache = root / "acquisition" / "cache"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS acquisition_jobs (job_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS source_registry (source_id TEXT PRIMARY KEY, canonical_url TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, job: AcquisitionJob) -> AcquisitionJob:
        updated = job.model_copy(update={"updated_at": datetime.now(UTC)})
        with self._connect() as connection:
            connection.execute("INSERT INTO acquisition_jobs VALUES (?, ?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload", (str(updated.job_id), updated.status.value, updated.updated_at.isoformat(), updated.model_dump_json()))
        return updated

    def get(self, job_id: UUID) -> AcquisitionJob | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM acquisition_jobs WHERE job_id = ?", (str(job_id),)).fetchone()
        return AcquisitionJob.model_validate_json(row["payload"]) if row else None

    def register_source(self, entry: SourceRegistryEntry) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO source_registry VALUES (?, ?, ?, ?) ON CONFLICT(source_id) DO UPDATE SET canonical_url=excluded.canonical_url, updated_at=excluded.updated_at, payload=excluded.payload",
                (entry.source_id, entry.canonical_url, entry.retrieved_at.isoformat(), entry.model_dump_json()),
            )

    def sources(self) -> tuple[SourceRegistryEntry, ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM source_registry ORDER BY source_id").fetchall()
        return tuple(SourceRegistryEntry.model_validate_json(row["payload"]) for row in rows)


class AcquisitionEngine:
    """Fetch only supplied sources, with a content cache and explicit retry state."""

    def __init__(self, root: Path, client: SafeWebClient | None = None) -> None:
        self.store = AcquisitionStore(root)
        self.client = client or SafeWebClient()

    async def run(self, job: AcquisitionJob) -> AcquisitionJob:
        started = time.perf_counter()
        job = self.store.save(job.model_copy(update={"status": AcquisitionStatus.RUNNING}))
        items = list(job.items)
        for index, item in enumerate(items):
            if item.status in {AcquisitionItemStatus.CACHED, AcquisitionItemStatus.FETCHED}:
                continue
            items[index] = await self._fetch_item(item, job.max_retries, job.backoff_seconds)
            job = self.store.save(job.model_copy(update={"items": items, "requests": job.requests + items[index].attempts, "bytes_downloaded": sum(candidate.extracted_characters for candidate in items), "elapsed_seconds": time.perf_counter() - started}))
        complete = sum(item.status in {AcquisitionItemStatus.CACHED, AcquisitionItemStatus.FETCHED} for item in items)
        status = AcquisitionStatus.COMPLETED if complete == len(items) else (AcquisitionStatus.PARTIAL if complete else AcquisitionStatus.FAILED)
        return self.store.save(job.model_copy(update={"items": items, "status": status, "elapsed_seconds": time.perf_counter() - started}))

    async def _fetch_item(self, item: AcquisitionItem, max_retries: int, backoff_seconds: float) -> AcquisitionItem:
        for attempt in range(item.attempts + 1, max_retries + 2):
            source, text = await self.client.fetch(item.source.url)
            if source.retrieval_status is RetrievalStatus.RETRIEVED and text and len(text) >= item.source.minimum_characters:
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                cache_path = self.store.cache / f"{content_hash}.txt"
                cached = cache_path.exists()
                if not cached:
                    cache_path.write_text(text, encoding="utf-8")
                updated = item.model_copy(update={"status": AcquisitionItemStatus.CACHED if cached else AcquisitionItemStatus.FETCHED, "attempts": attempt, "retry_count": attempt - 1, "fetched_at": source.retrieved_at, "content_hash": content_hash, "extracted_characters": len(text), "cache_path": str(cache_path), "source_record": source.model_copy(update={"source_type": item.source.source_type})})
                self.store.register_source(self.source_registry_entry(updated))
                return updated
            error = f"{source.retrieval_status.value}: content unavailable or below {item.source.minimum_characters} characters"
            if attempt <= max_retries:
                await asyncio.sleep(backoff_seconds * 2 ** (attempt - 1))
        return item.model_copy(update={"status": AcquisitionItemStatus.FAILED, "attempts": max_retries + 1, "retry_count": max_retries, "last_error": error})

    def corpus_documents(self, job: AcquisitionJob) -> tuple[CorpusDocument, ...]:
        documents: list[CorpusDocument] = []
        for item in job.items:
            if item.status not in {AcquisitionItemStatus.CACHED, AcquisitionItemStatus.FETCHED} or not item.cache_path or not item.source_record:
                continue
            text = Path(item.cache_path).read_text(encoding="utf-8")
            documents.append(CorpusDocument(content=text, provenance=Provenance(source_type=CorpusSourceType.PUBLIC_DATASET, source_identifier=item.source.source_identifier, source_url=item.source.url, license=item.source.license, license_evidence=item.source.license_evidence, source_group=item.source.source_concentration_group or item.source.source_identifier, acquisition_version=item.source.acquisition_version, parser_version=item.source.parser_version, declared_category=item.source.category, copyright_metadata=f"source_quality_score={self.source_quality(item):.3f}")))
        return tuple(documents)

    @staticmethod
    def source_quality(item: AcquisitionItem) -> float:
        return AcquisitionEngine.source_quality_factors(item).score

    @staticmethod
    def source_quality_factors(item: AcquisitionItem) -> SourceQuality:
        if item.status not in {AcquisitionItemStatus.CACHED, AcquisitionItemStatus.FETCHED} or not item.source_record:
            return SourceQuality(authority=0.0, provenance_completeness=0.0, structural_quality=0.0, machine_readability=0.0, stability=0.0, licensing_clarity=0.0, duplication_risk=1.0, noise_risk=1.0, content_density=0.0, score=0.0)
        type_score = {SourceType.GOVERNMENT: 0.9, SourceType.ACADEMIC: 0.85, SourceType.DOCUMENTATION: 0.8, SourceType.PRIMARY: 0.8}.get(item.source.source_type, 0.55)
        structure = min(1.0, item.extracted_characters / 10_000)
        provenance = 1.0 if item.source.license and item.source.source_identifier and item.source.url else 0.4
        licensing = 1.0 if item.source.license not in {"", "UNSPECIFIED"} and item.source.license_evidence else 0.5
        density = min(1.0, item.extracted_characters / 20_000)
        score = round(0.22 * type_score + 0.14 * provenance + 0.13 * structure + 0.12 * structure + 0.10 * 0.8 + 0.12 * licensing + 0.08 * density + 0.05 * 0.8 + 0.04 * 0.8, 3)
        return SourceQuality(authority=type_score, provenance_completeness=provenance, structural_quality=structure, machine_readability=structure, stability=0.8, licensing_clarity=licensing, duplication_risk=0.2, noise_risk=0.2, content_density=density, score=score)

    @staticmethod
    def source_registry_entry(item: AcquisitionItem) -> SourceRegistryEntry:
        if not item.source_record or not item.content_hash:
            raise ValueError("Only successfully retrieved sources can enter the source registry.")
        source = item.source_record
        return SourceRegistryEntry(source_id=item.source.source_identifier, canonical_url=source.canonical_url or item.source.url, title=item.source.title or source.title, publisher=item.source.publisher or source.publisher, source_type=item.source.source_type.value, category=item.source.category, license=item.source.license, license_evidence=item.source.license_evidence, retrieved_at=source.retrieved_at, published_at=source.published_at, content_hash=item.content_hash, retrieval_status=source.retrieval_status.value, parser_version=item.source.parser_version, acquisition_version=item.source.acquisition_version, provenance={"url": item.source.url, "cache_path": item.cache_path or ""}, quality=AcquisitionEngine.source_quality_factors(item), source_concentration_group=item.source.source_concentration_group or item.source.source_identifier)
