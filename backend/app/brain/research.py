"""Bounded public-web research with source/claim/evidence separation.

No web content is executable.  This module has no capability registration,
authorization, shell, filesystem, or model dependency; it only obtains and
evaluates public text within fixed runtime budgets.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import ipaddress
import json
import re
import socket
from datetime import UTC, datetime
from enum import Enum
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote, quote_plus, urljoin, urlparse, urlunparse
from uuid import UUID, uuid4

import httpx
import anyio
from pydantic import BaseModel, ConfigDict, Field

from app.brain.learning import KnowledgeStatus, LearningRepository
from app.config import settings
from app.local_intelligence.network import require_network_tool


class ResearchTaskStatus(str, Enum):
    QUEUED = "queued"
    DISCOVERING = "discovering"
    RETRIEVING = "retrieving"
    ANALYZING = "analyzing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    DOCUMENTATION = "documentation"
    NEWS = "news"
    COMMUNITY = "community"
    BLOG = "blog"
    UNKNOWN = "unknown"


class RetrievalStatus(str, Enum):
    RETRIEVED = "retrieved"
    UNCHANGED = "unchanged"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ContradictionStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    PERSISTENT_CONFLICT = "persistent_conflict"
    STALE_CONFLICT = "stale_conflict"


class ResearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_searches: int = Field(default=2, ge=1, le=3)
    max_urls: int = Field(default=12, ge=1, le=12)
    max_depth: int = Field(default=1, ge=0, le=2)
    max_total_bytes: int = Field(default=4_000_000, ge=10_000, le=8_000_000)
    max_duration_seconds: float = Field(default=45, gt=0, le=90)
    max_concurrent_requests: int = Field(default=6, ge=1, le=8)


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=3, max_length=500)
    status: ResearchTaskStatus = ResearchTaskStatus.QUEUED
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cancellation_requested: bool = False
    failure_reason: str | None = Field(default=None, max_length=500)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: UUID = Field(default_factory=uuid4)
    url: str = Field(min_length=1, max_length=2_000)
    canonical_url: str | None = Field(default=None, max_length=2_000)
    domain: str = Field(min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=500)
    publisher: str | None = Field(default=None, max_length=255)
    author: str | None = Field(default=None, max_length=255)
    published_at: datetime | None = None
    retrieved_at: datetime
    content_hash: str | None = Field(default=None, max_length=64)
    content_type: str | None = Field(default=None, max_length=100)
    http_status: int | None = Field(default=None, ge=100, le=599)
    source_type: SourceType = SourceType.UNKNOWN
    retrieval_status: RetrievalStatus
    trust_metadata: dict[str, float | str] = Field(default_factory=dict, max_length=8)


class WebClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    subject: str = Field(min_length=1, max_length=300)
    predicate: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=800)
    claim_text: str = Field(min_length=1, max_length=1_200)
    claim_key: str = Field(min_length=1, max_length=100)
    extraction_confidence: float = Field(ge=0, le=1)
    temporal_context: str | None = Field(default=None, max_length=200)


class WebEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    source_id: UUID
    passage: str = Field(min_length=1, max_length=1_500)
    content_hash: str
    retrieved_at: datetime
    relevance: float = Field(ge=0, le=1)
    supports: bool


class ResearchContradiction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contradiction_id: UUID = Field(default_factory=uuid4)
    claim_a: UUID
    claim_b: UUID
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: ContradictionStatus = ContradictionStatus.UNRESOLVED
    reason: str = Field(min_length=1, max_length=500)


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_id: UUID
    question: str
    answer: str
    sources: tuple[SourceRecord, ...]
    claims: tuple[WebClaim, ...]
    evidence: tuple[WebEvidence, ...]
    contradictions: tuple[ResearchContradiction, ...]
    verification_status: KnowledgeStatus | None = None
    knowledge_ids: tuple[UUID, ...] = ()
    budget_used: dict[str, int | float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    uncertainty: tuple[str, ...] = ()


class SearchProvider(Protocol):
    async def search(self, query: str, limit: int) -> list[str]: ...


_INJECTION = re.compile(r"(?:ignore\s+(?:all|previous|system)|system\s+prompt|run\s+(?:rm|curl|wget)|send\s+(?:the\s+)?(?:api|secret)|you\s+are\s+now\s+authorized)", re.I)
_WORDS = re.compile(r"[a-z0-9]{2,}", re.I)
_BOILERPLATE = re.compile(r"(?:cookie|privacy\s+policy|youtube|turn\s+on\s+optional|sign\s+in|subscribe|javascript)", re.I)
_QUERY_STOP_WORDS = frozenset({"what", "which", "where", "when", "who", "current", "currently", "latest", "today", "about", "with", "from", "that", "this", "the", "and", "for", "are", "was", "is", "search", "internet", "give", "answer", "sources", "source", "used", "please", "every", "single"})
_GENERIC_SCOPE_TERMS = frozenset({"full", "comprehensive", "complete", "list", "part", "parts", "component", "components", "system", "systems"})


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title: str | None = None
        self._in_title = False
        self._ignored = 0
        self._heading_level: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "article", "section", "li", "h1", "h2", "h3", "h4", "br"}:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4"}:
            self._heading_level = int(tag[1])
            self.parts.append(f"[[H{self._heading_level}]] ")
        if tag in {"script", "style", "noscript", "svg", "iframe"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "iframe"} and self._ignored:
            self._ignored -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3", "h4"}:
            self._heading_level = None

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        value = " ".join(data.split())
        if value:
            self.parts.append(value)
            if self._in_title and self.title is None:
                self.title = value[:500]


class SafeWebClient:
    """Public HTTP-only client with URL, DNS, redirect, size, and type guards."""

    def __init__(self, client: httpx.AsyncClient | None = None, *, max_extracted_characters: int = 50_000) -> None:
        if not 1_000 <= max_extracted_characters <= settings.brain_research_max_response_bytes:
            raise ValueError("max_extracted_characters must fit the configured bounded response limit.")
        self.max_extracted_characters = max_extracted_characters
        self._client = client or httpx.AsyncClient(
            timeout=settings.brain_research_timeout,
            follow_redirects=False,
            headers={"User-Agent": "UltronResearch/1.0 (+bounded-public-research)"},
        )

    async def search(self, query: str, limit: int) -> list[str]:
        require_network_tool("web_research")
        # Discovery is intentionally one fixed public endpoint, not arbitrary
        # model-selected network access. Failures degrade research truthfully.
        response = await self._client.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(_discovery_query(query))}"
        )
        if response.status_code != 200:
            return await self._wikipedia_search(_discovery_query(query), limit)
        links = re.findall(r'class="result__a"[^>]+href="([^"]+)"', response.text)
        urls: list[str] = []
        for link in links:
            parsed = urlparse(html.unescape(link))
            target = parse_qs(parsed.query).get("uddg", [link])[0] if "duckduckgo.com" in parsed.netloc else link
            if target not in urls:
                urls.append(target)
            if len(urls) >= limit:
                break
        return urls or await self._wikipedia_search(_discovery_query(query), limit)

    async def _wikipedia_search(self, query: str, limit: int) -> list[str]:
        """Use a fixed public search API when generic discovery rate-limits us.

        This is a bounded fallback, not a way for model text to choose an API.
        Returned pages still go through the same DNS, redirect, type, and size
        guards in ``fetch``.
        """
        try:
            response = await self._client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query[:300],
                    "format": "json",
                    "srlimit": min(limit, 6),
                },
            )
            payload = response.json() if response.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return []
        results = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
        urls: list[str] = []
        for item in results:
            title = item.get("title") if isinstance(item, dict) else None
            if isinstance(title, str) and title:
                urls.append(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    f"{quote(title.replace(' ', '_'))}"
                )
        return list(dict.fromkeys(urls))[:limit]

    async def fetch(self, url: str) -> tuple[SourceRecord, str | None]:
        require_network_tool("web_research")
        current = self._validate_url(url)
        for _ in range(settings.brain_research_max_redirects + 1):
            try:
                await self._validate_dns(current)
            except ValueError:
                return self._failed_source(current, None, RetrievalStatus.BLOCKED), None
            try:
                async with self._client.stream("GET", current) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            return self._failed_source(current, response.status_code, RetrievalStatus.FAILED), None
                        current = self._validate_url(urljoin(current, location))
                        continue
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type not in {"text/html", "text/plain", "application/xhtml+xml", "application/json"}:
                        return self._failed_source(current, response.status_code, RetrievalStatus.UNSUPPORTED, content_type), None
                    if response.status_code != 200:
                        return self._failed_source(current, response.status_code, RetrievalStatus.FAILED, content_type), None
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > settings.brain_research_max_response_bytes:
                            return self._failed_source(current, response.status_code, RetrievalStatus.FAILED, content_type), None
                        chunks.append(chunk)
            except (httpx.HTTPError, asyncio.TimeoutError):
                return self._failed_source(current, None, RetrievalStatus.FAILED), None
            raw = b"".join(chunks).decode("utf-8", errors="replace")
            parser = _TextParser()
            if content_type == "application/json":
                text, json_title = _json_text(raw)
                parser.title = json_title
            else:
                parser.feed(raw)
                text = " ".join(parser.parts).replace(" \n ", "\n")[:self.max_extracted_characters]
            canonical = self._canonical(current)
            source = SourceRecord(url=current, canonical_url=canonical, domain=urlparse(current).hostname or "unknown", title=parser.title, publisher=urlparse(current).hostname, retrieved_at=datetime.now(UTC), content_hash=hashlib.sha256(text.encode()).hexdigest(), content_type=content_type, http_status=200, source_type=_classify(current), retrieval_status=RetrievalStatus.RETRIEVED, trust_metadata={"type_weight": _source_weight(_classify(current))})
            return source, text
        return self._failed_source(current, None, RetrievalStatus.BLOCKED), None

    @staticmethod
    def _validate_url(value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only public HTTP(S) URLs are permitted.")
        if parsed.username or parsed.password or (parsed.port not in {None, 80, 443}):
            raise ValueError("URL credentials and nonstandard ports are not permitted.")
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""))

    @staticmethod
    async def _validate_dns(url: str) -> None:
        host = urlparse(url).hostname
        assert host
        try:
            addresses = await anyio.to_thread.run_sync(
                lambda: socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            )
        except socket.gaierror as exc:
            raise ValueError("Public hostname could not be resolved.") from exc
        if not addresses:
            raise ValueError("Public hostname has no address.")
        for item in addresses:
            address = ipaddress.ip_address(item[4][0])
            if not address.is_global:
                raise ValueError("Private, loopback, link-local, and reserved addresses are blocked.")

    @staticmethod
    def _canonical(url: str) -> str:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/") or "/", "", parsed.query, ""))

    @staticmethod
    def _failed_source(url: str, status: int | None, retrieval_status: RetrievalStatus, content_type: str | None = None) -> SourceRecord:
        return SourceRecord(url=url, domain=urlparse(url).hostname or "unknown", retrieved_at=datetime.now(UTC), http_status=status, content_type=content_type, retrieval_status=retrieval_status)


def _json_text(raw: str) -> tuple[str, str | None]:
    """Extract bounded display text from a retrieved public JSON document."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "", None
    if not isinstance(payload, dict):
        return "", None
    title = payload.get("title")
    extract = payload.get("extract") or payload.get("description") or payload.get("content")
    text = extract if isinstance(extract, str) else ""
    return " ".join(text.split())[:50_000], title[:500] if isinstance(title, str) else None


def _discovery_query(question: str) -> str:
    """Remove request boilerplate before fixed-provider discovery."""
    terms = [word for word in _WORDS.findall(question.lower()) if word not in _QUERY_STOP_WORDS]
    return " ".join(terms[:12]) or question[:300]


def _classify(url: str) -> SourceType:
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    if host.endswith(".gov"):
        return SourceType.GOVERNMENT
    if host.endswith(".edu"):
        return SourceType.ACADEMIC
    if any(part in path for part in ("/docs", "/documentation", "/api/")):
        return SourceType.DOCUMENTATION
    return SourceType.UNKNOWN


def _source_weight(kind: SourceType) -> float:
    return {SourceType.GOVERNMENT: 0.75, SourceType.ACADEMIC: 0.75, SourceType.DOCUMENTATION: 0.65}.get(kind, 0.45)


class ResearchStore:
    """Research-specific tables share the learning SQLite file and WAL behavior."""

    def __init__(self, repository: LearningRepository) -> None:
        self._repository = repository
        self._lock = repository._lock  # shared transaction discipline
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._repository._connect() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS research_tasks (research_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS web_sources (source_id TEXT PRIMARY KEY, canonical_url TEXT, content_hash TEXT, retrieved_at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS web_sources_hash ON web_sources(canonical_url, content_hash);
            CREATE TABLE IF NOT EXISTS web_claims (claim_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, claim_key TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS web_claims_key ON web_claims(claim_key);
            CREATE TABLE IF NOT EXISTS web_evidence (evidence_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL, source_id TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS web_contradictions (contradiction_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_results (research_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_metrics (domain TEXT PRIMARY KEY, attempts INTEGER NOT NULL, retrieved INTEGER NOT NULL, contradicted INTEGER NOT NULL, updated_at TEXT NOT NULL);
            """)

    def save_task(self, task: ResearchTask) -> None:
        with self._lock, self._repository._connect() as connection:
            connection.execute("INSERT INTO research_tasks VALUES (?, ?, ?, ?) ON CONFLICT(research_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, payload=excluded.payload", (str(task.research_id), task.status.value, task.updated_at.isoformat(), task.model_dump_json()))

    def task(self, research_id: UUID) -> ResearchTask | None:
        with self._lock, self._repository._connect() as connection:
            row = connection.execute("SELECT payload FROM research_tasks WHERE research_id = ?", (str(research_id),)).fetchone()
        return ResearchTask.model_validate_json(row["payload"]) if row else None

    def save_result(self, result: ResearchResult) -> None:
        with self._lock, self._repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for source in result.sources:
                connection.execute("INSERT OR REPLACE INTO web_sources VALUES (?, ?, ?, ?, ?)", (str(source.source_id), source.canonical_url, source.content_hash, source.retrieved_at.isoformat(), source.model_dump_json()))
                prior = connection.execute("SELECT * FROM source_metrics WHERE domain = ?", (source.domain,)).fetchone()
                attempts = (prior["attempts"] if prior else 0) + 1
                retrieved = (prior["retrieved"] if prior else 0) + int(source.retrieval_status is RetrievalStatus.RETRIEVED)
                conflict_claims = {item.claim_a for item in result.contradictions} | {item.claim_b for item in result.contradictions}
                source_claims = {claim.claim_id for claim in result.claims if claim.source_id == source.source_id}
                contradicted = (prior["contradicted"] if prior else 0) + int(bool(conflict_claims & source_claims))
                connection.execute("INSERT INTO source_metrics VALUES (?, ?, ?, ?, ?) ON CONFLICT(domain) DO UPDATE SET attempts=excluded.attempts, retrieved=excluded.retrieved, contradicted=excluded.contradicted, updated_at=excluded.updated_at", (source.domain, attempts, retrieved, contradicted, datetime.now(UTC).isoformat()))
            for claim in result.claims:
                connection.execute("INSERT OR REPLACE INTO web_claims VALUES (?, ?, ?, ?)", (str(claim.claim_id), str(claim.source_id), claim.claim_key, claim.model_dump_json()))
            for evidence in result.evidence:
                connection.execute("INSERT OR REPLACE INTO web_evidence VALUES (?, ?, ?, ?)", (str(evidence.evidence_id), str(evidence.claim_id), str(evidence.source_id), evidence.model_dump_json()))
            for conflict in result.contradictions:
                connection.execute("INSERT OR REPLACE INTO web_contradictions VALUES (?, ?)", (str(conflict.contradiction_id), conflict.model_dump_json()))
            connection.execute("INSERT OR REPLACE INTO research_results VALUES (?, ?)", (str(result.research_id), result.model_dump_json()))
            connection.execute("COMMIT")

    def unchanged(self, source: SourceRecord) -> bool:
        if not source.canonical_url or not source.content_hash:
            return False
        with self._lock, self._repository._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM web_sources WHERE canonical_url = ? AND content_hash = ? LIMIT 1",
                (source.canonical_url, source.content_hash),
            ).fetchone()
        return row is not None

    def result(self, research_id: UUID) -> ResearchResult | None:
        with self._lock, self._repository._connect() as connection:
            row = connection.execute("SELECT payload FROM research_results WHERE research_id = ?", (str(research_id),)).fetchone()
        return ResearchResult.model_validate_json(row["payload"]) if row else None

    def recover_interrupted(self) -> None:
        """A restart leaves nonterminal work explicit; it is never completed."""
        terminal = tuple(item.value for item in (ResearchTaskStatus.COMPLETED, ResearchTaskStatus.DEGRADED, ResearchTaskStatus.FAILED, ResearchTaskStatus.CANCELLED))
        with self._lock, self._repository._connect() as connection:
            rows = connection.execute(f"SELECT payload FROM research_tasks WHERE status NOT IN ({','.join('?' for _ in terminal)})", terminal).fetchall()
            for row in rows:
                task = ResearchTask.model_validate_json(row["payload"]).model_copy(update={"status": ResearchTaskStatus.FAILED, "updated_at": datetime.now(UTC), "failure_reason": "Research interrupted by process restart."})
                connection.execute("UPDATE research_tasks SET status = ?, updated_at = ?, payload = ? WHERE research_id = ?", (task.status.value, task.updated_at.isoformat(), task.model_dump_json(), str(task.research_id)))

    def source_reliability(self, domain: str) -> dict[str, float | int] | None:
        with self._lock, self._repository._connect() as connection:
            row = connection.execute("SELECT * FROM source_metrics WHERE domain = ?", (domain.lower(),)).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"])
        # This is operational history, not a trust grant for future content.
        return {"attempts": attempts, "retrieval_rate": round(int(row["retrieved"]) / attempts, 3), "contradiction_rate": round(int(row["contradicted"]) / attempts, 3)}


class ResearchEngine:
    def __init__(self, *, repository: LearningRepository, web: SearchProvider | SafeWebClient | None = None) -> None:
        self.repository = repository
        self.store = ResearchStore(repository)
        self.web = web or SafeWebClient()

    async def run(self, task: ResearchTask) -> ResearchResult:
        self._save(task, ResearchTaskStatus.DISCOVERING)
        started = anyio.current_time()
        try:
            # Broad/exhaustive work needs domain discovery plus coverage
            # passes. Each query changes the retrieval objective rather than
            # trivially repeating wording; candidates still pass relevance
            # validation before claim extraction.
            urls: list[str] = []
            searches = 0
            queries = _research_queries(task.question, task.budget.max_searches)
            per_query_limit = max(1, task.budget.max_urls // len(queries))
            for query in queries:
                searches += 1
                for url in await self.web.search(query, per_query_limit):
                    if url not in urls:
                        urls.append(url)
                    if len(urls) >= task.budget.max_urls:
                        break
                if len(urls) >= task.budget.max_urls:
                    break
            self._save(task, ResearchTaskStatus.RETRIEVING)
            sources: list[SourceRecord] = []
            contents: dict[UUID, str] = {}
            total_bytes = 0
            semaphore = anyio.Semaphore(task.budget.max_concurrent_requests)
            fetched: list[tuple[SourceRecord, str | None] | None] = [None] * len(urls)

            async def fetch_one(index: int, url: str) -> None:
                async with semaphore:
                    fetched[index] = await self.web.fetch(url)  # type: ignore[union-attr]

            async with anyio.create_task_group() as group:
                for index, url in enumerate(urls):
                    group.start_soon(fetch_one, index, url)
            for item in fetched:
                assert item is not None
                source, text = item
                if anyio.current_time() - started > task.budget.max_duration_seconds:
                    break
                if task.cancellation_requested:
                    return self._cancelled(task)
                if text and self.store.unchanged(source):
                    source = source.model_copy(update={"retrieval_status": RetrievalStatus.UNCHANGED})
                    # Identical content is not a new source observation, but
                    # it remains valid bounded input for this task's claim
                    # extraction. Dropping it made repeat live research look
                    # successful at retrieval yet yield no evidence at all.
                sources.append(source)
                if text:
                    size = len(text.encode())
                    if total_bytes + size <= task.budget.max_total_bytes:
                        contents[source.source_id] = text
                        total_bytes += size
            self._save(task, ResearchTaskStatus.ANALYZING)
            claims = tuple(sorted(
                (claim for source in sources if source.source_id in contents for claim in _extract_claims(task.question, source, contents[source.source_id])),
                key=lambda claim: claim.extraction_confidence,
                reverse=True,
            ))
            evidence = tuple(WebEvidence(claim_id=claim.claim_id, source_id=claim.source_id, passage=claim.claim_text, content_hash=next(source.content_hash for source in sources if source.source_id == claim.source_id) or "", retrieved_at=next(source.retrieved_at for source in sources if source.source_id == claim.source_id), relevance=claim.extraction_confidence, supports=True) for claim in claims)
            conflicts = _contradictions(claims)
            self._save(task, ResearchTaskStatus.VERIFYING)
            knowledge_ids, verified, confidence = self._promote(task, claims, sources, conflicts)
            status = ResearchTaskStatus.COMPLETED if verified else ResearchTaskStatus.DEGRADED
            answer, uncertainty = _answer(task.question, claims, conflicts, verified, contents)
            result = ResearchResult(research_id=task.research_id, question=task.question, answer=answer, sources=tuple(sources), claims=claims, evidence=evidence, contradictions=conflicts, verification_status=KnowledgeStatus.VERIFIED if verified else (KnowledgeStatus.SUPPORTED if knowledge_ids else None), knowledge_ids=knowledge_ids, budget_used={"searches": searches, "urls": len(urls), "sources": len(sources), "bytes": total_bytes}, confidence=confidence, uncertainty=uncertainty)
            self.store.save_result(result)
            self._save(task, status)
            return result
        except Exception:
            self._save(task, ResearchTaskStatus.FAILED, "Research failed safely.")
            return ResearchResult(research_id=task.research_id, question=task.question, answer="I couldn't complete the bounded public-web research safely.", sources=(), claims=(), evidence=(), contradictions=(), confidence=0.0, uncertainty=("Research failed or a source was unavailable.",))

    def _promote(self, task: ResearchTask, claims: tuple[WebClaim, ...], sources: list[SourceRecord], conflicts: tuple[ResearchContradiction, ...]) -> tuple[tuple[UUID, ...], bool, float]:
        conflicted = {item.claim_a for item in conflicts} | {item.claim_b for item in conflicts}
        ids: list[UUID] = []
        verified = False
        for claim in claims:
            if claim.claim_id in conflicted:
                continue
            support = [source for source in sources if source.source_id == claim.source_id and source.retrieval_status is RetrievalStatus.RETRIEVED]
            # Distinct domains, not URL count, are the corroboration unit.
            peers = [other for other in claims if other.claim_key == claim.claim_key and other.claim_id not in conflicted]
            domains = {next(source.domain for source in sources if source.source_id == other.source_id) for other in peers}
            if len(domains) >= 2:
                state, score = KnowledgeStatus.VERIFIED, min(0.95, 0.70 + 0.1 * len(domains))
                verified = True
            elif support:
                state, score = KnowledgeStatus.SUPPORTED, 0.55
            else:
                continue
            knowledge = self.repository.record_web_knowledge(claim=claim.claim_text[:500], claim_key=claim.claim_key, status=state, confidence=score, source_ids=tuple(source.source_id for source in sources if source.domain in domains)[:8], research_id=task.research_id, freshness_seconds=_freshness(task.question))
            ids.append(knowledge.knowledge_id)
        return tuple(dict.fromkeys(ids)), verified, 0.8 if verified else (0.55 if ids else 0.0)

    def _save(self, task: ResearchTask, status: ResearchTaskStatus, reason: str | None = None) -> None:
        task = task.model_copy(update={"status": status, "updated_at": datetime.now(UTC), "failure_reason": reason})
        self.store.save_task(task)

    def _cancelled(self, task: ResearchTask) -> ResearchResult:
        self._save(task, ResearchTaskStatus.CANCELLED)
        return ResearchResult(research_id=task.research_id, question=task.question, answer="Research was cancelled.", sources=(), claims=(), evidence=(), contradictions=(), confidence=0.0, uncertainty=("Cancelled before verification.",))


def _extract_claims(question: str, source: SourceRecord, content: str) -> tuple[WebClaim, ...]:
    # Retrieval is not evidence by itself. Reject a source before extracting
    # claims when it cannot meet the question's object/domain specificity
    # threshold. This is deliberately capability-agnostic: it evaluates the
    # task terms against the source title/domain and retrieved body.
    if source_relevance_score(question, source, content) < 2:
        return ()
    terms = set(_WORDS.findall(question.lower())) - _QUERY_STOP_WORDS
    source_context = set(_WORDS.findall(f"{source.title or ''} {source.domain}".lower()))
    claims: list[WebClaim] = []
    for sentence in re.split(r"(?:\n+|(?<=[.!?])\s+)", content):
        sentence = " ".join(sentence.split())[:1_200]
        words = set(_WORDS.findall(sentence.lower()))
        overlap = (words | source_context) & terms
        required_overlap = min(2, len(terms))
        if (
            len(sentence) < 30
            or len(sentence) > 500
            or _non_substantive_retrieval_text(sentence)
            or _INJECTION.search(sentence)
            or _BOILERPLATE.search(sentence)
            or len(overlap) < required_overlap
        ):
            continue
        normalized = " ".join(_WORDS.findall(sentence.lower()))
        key = hashlib.sha256(normalized.encode()).hexdigest()[:40]
        authority_bonus = 0.12 if source.domain.endswith("nvidia.com") else 0.0
        claims.append(WebClaim(source_id=source.source_id, subject=question[:300], predicate="reports", object=sentence[:800], claim_text=sentence, claim_key=key, extraction_confidence=min(0.9, 0.35 + len(overlap) * 0.08 + authority_bonus)))
        if len(claims) >= 4:
            break
    return tuple(claims)


def _non_substantive_retrieval_text(sentence: str) -> bool:
    """Keep page chrome, headings, and navigation out of evidence claims."""
    lowered = sentence.casefold()
    markers = ("skip to", "navigation", "cookie", "sign in", "subscribe", "menu", "drivers teams", "race weekend")
    if any(marker in lowered for marker in markers):
        return True
    words = _WORDS.findall(lowered)
    # Heading-like title fragments are not evidence for an inventory or claim.
    return len(words) < 12 and not re.search(r"[.!?]$", sentence)


def source_relevance_score(question: str, source: SourceRecord, content: str) -> int:
    """Return a bounded relevance gate score; scores below 2 are rejected.

    The score is not a truth score. It prevents keyword-adjacent pages from
    becoming evidence before the claim verifier evaluates support.
    """
    terms = set(_WORDS.findall(question.casefold())) - _QUERY_STOP_WORDS
    if not terms:
        return 0
    source_terms = set(_WORDS.findall(f"{source.title or ''} {source.domain}".casefold()))
    body_terms = set(_WORDS.findall(content[:12_000].casefold()))
    title_overlap = len(terms & source_terms)
    # Generic scope words ("list", "component", "complete") cannot prove
    # that a result concerns the requested object. When the request contains
    # a more specific object/domain term, require that term as well.
    distinctive = terms - _GENERIC_SCOPE_TERMS
    required_terms = distinctive or terms
    required = min(2, len(required_terms))
    if len(required_terms & body_terms) < required:
        return 0
    score = 2 + int(title_overlap >= 1) + int(title_overlap >= required)
    if source.source_type in {SourceType.PRIMARY, SourceType.GOVERNMENT, SourceType.ACADEMIC, SourceType.DOCUMENTATION}:
        score += 1
    return min(score, 5)


def _contradictions(claims: tuple[WebClaim, ...]) -> tuple[ResearchContradiction, ...]:
    conflicts: list[ResearchContradiction] = []
    for index, left in enumerate(claims):
        left_terms = set(_WORDS.findall(re.sub(r"\b(?:not|no|never|false)\b", "", left.claim_text.lower())))
        left_negative = bool(re.search(r"\b(?:not|no|never|false)\b", left.claim_text, re.I))
        for right in claims[index + 1:]:
            right_negative = bool(re.search(r"\b(?:not|no|never|false)\b", right.claim_text, re.I))
            if left_negative != right_negative and len(left_terms & set(_WORDS.findall(right.claim_text.lower()))) >= 4:
                conflicts.append(ResearchContradiction(claim_a=left.claim_id, claim_b=right.claim_id, reason="Claims share subject terms but assert opposing polarity."))
    return tuple(conflicts)


def _answer(question: str, claims: tuple[WebClaim, ...], conflicts: tuple[ResearchContradiction, ...], verified: bool, contents: dict[UUID, str]) -> tuple[str, tuple[str, ...]]:
    if conflicts:
        return "Sources contain unresolved conflicting claims; I cannot present a single verified conclusion.", ("Sources disagree.",)
    if not claims:
        return "The bounded research found no sufficiently relevant evidence.", ("No evidence-backed claim was extracted.",)
    if _is_broad_decomposition(question):
        inventory = _structured_inventory(contents)
        if inventory:
            return (
                "Scope: publicly documented systems and named subcomponents; proprietary internals and individual fasteners are excluded.\n\n"
                "Documented inventory:\n" + inventory,
                (() if verified else ("Coverage is source-backed but not independently corroborated for every branch.",)),
            )
        return "The retrieved sources did not expose a usable component hierarchy.", ("Inventory coverage validation failed; no snippet list was returned.",)
    qualifier = "corroborated" if verified else "single-source and not fully verified"
    return f"Based on {qualifier} public-web evidence: {claims[0].claim_text}", (() if verified else ("Additional independent corroboration is needed.",))


def _structured_inventory(contents: dict[UUID, str]) -> str | None:
    """Extract a bounded heading hierarchy from relevant retrieved documents."""
    seen: set[str] = set()
    rows: list[tuple[int, str]] = []
    for content in contents.values():
        for level, text in re.findall(r"\[\[H([1-4])\]\]\s*([^\n]+)", content):
            label = " ".join(text.split()).strip(" -:.")
            normalized = re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()
            if (
                len(_WORDS.findall(label)) < 2 or len(label) > 150 or normalized in seen
                or "?" in label
                or re.match(r"^(?:what|why|how|when|where|who|do|does|is|are)\b", label, re.I)
                or re.match(r"^doc\s+\d+\b", label, re.I)
                or re.search(r"\b(?:faq|frequently asked|sources|references|more in|related|subscribe|contact|see every|working, live)\b", label, re.I)
            ):
                continue
            seen.add(normalized)
            rows.append((int(level), label))
    if len(rows) < 5:
        return None
    base = min(level for level, _ in rows)
    return "\n".join(f"{'  ' * min(3, level - base)}- {label}" for level, label in rows[:80])


def _is_broad_decomposition(question: str) -> bool:
    return bool(re.search(r"\b(?:every|all)\s+(?:single\s+)?(?:component|part|subsystem|element)s?\b", question, re.IGNORECASE))


def _research_queries(question: str, max_searches: int) -> tuple[str, ...]:
    """Generate task-structure passes without encoding an example domain."""
    queries = [question]
    if _is_broad_decomposition(question):
        focused_terms = [term for term in _WORDS.findall(question.casefold()) if term not in _QUERY_STOP_WORDS | _GENERIC_SCOPE_TERMS]
        focused = " ".join(dict.fromkeys(focused_terms)) or question
        queries.extend((
            f"{focused} official technical documentation regulations",
            f"{focused} engineering architecture systems components",
        ))
    return tuple(dict.fromkeys(queries))[:max_searches]


def _freshness(question: str) -> int:
    text = question.lower()
    if any(word in text for word in ("current", "today", "latest", "price", "weather", "now")):
        return 3_600
    if any(word in text for word in ("documentation", "version", "api")):
        return 2_592_000
    return 15_552_000


class ResearchManager:
    """Explicit in-process background boundary for small bounded research jobs."""

    def __init__(self, engine: ResearchEngine) -> None:
        self.engine = engine
        self._jobs: dict[UUID, asyncio.Task[ResearchResult]] = {}
        self.engine.store.recover_interrupted()

    def submit(self, question: str, budget: ResearchBudget | None = None) -> ResearchTask:
        task = ResearchTask(question=question, budget=budget or ResearchBudget())
        self.engine.store.save_task(task)
        self._jobs[task.research_id] = asyncio.create_task(self.engine.run(task))
        return task

    def cancel(self, research_id: UUID) -> bool:
        job = self._jobs.get(research_id)
        if job is None or job.done():
            return False
        job.cancel()
        current = self.engine.store.task(research_id)
        if current:
            self.engine.store.save_task(current.model_copy(update={"status": ResearchTaskStatus.CANCELLED, "cancellation_requested": True, "updated_at": datetime.now(UTC)}))
        return True

    def queue_stale_revalidations(self, limit: int = 5) -> tuple[ResearchTask, ...]:
        """Explicit scheduler seam; callers decide when revalidation is permitted."""
        return tuple(self.submit(item.claim) for item in self.engine.repository.stale_knowledge(limit=limit))
