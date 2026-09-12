"""Bounded deterministic analysis for explicitly registered public corpora.

The analyzer is deliberately independent of language-model output: once a
corpus has been obtained, token positions and ranges are computed from a
document hash and a named tokenization rule.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.brain.actions import ActionResult, ActionStatus, CapabilityResult
from app.brain.conversation_capabilities import CapabilityDescriptor, ConversationAction
from app.brain.schemas import TaskAssessment, TaskType
from app.local_intelligence.network import require_network_tool
from app.brain.verification import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceCollection,
    EvidenceReliability,
    EvidenceSourceType,
)

_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+word\b", re.IGNORECASE)
_RANGE = re.compile(r"\b(?:words?|tokens?)\s+(\d+)\s*(?:through|to|-)\s*(\d+)\b", re.IGNORECASE)
_KJV = re.compile(r"\b(?:king\s+james|kjv)\b", re.IGNORECASE)
_BIBLE = re.compile(r"\bbible\b", re.IGNORECASE)
_GUTENBERG_BIBLE = re.compile(r"\b(?:project\s+)?gutenberg\s+bible\b", re.IGNORECASE)
_TOKEN = re.compile(r"\S+")
KJV_GENESIS_ONE_URL = "https://bible-api.com/genesis%201?translation=kjv"
GUTENBERG_BIBLE_URL = "https://www.gutenberg.org/files/10/10-0.txt"


class CorpusAnalysisParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_text: str = Field(min_length=3, max_length=1_000)


class DeterministicDocumentAnalyzer:
    """Pure analysis core reusable for any supplied corpus text."""

    tokenizer_id = "unicode-whitespace-v1"

    def analyze(self, *, corpus_id: str, source_url: str, text: str, start: int, end: int) -> dict[str, object]:
        tokens = tuple(_TOKEN.findall(text))
        if start < 1 or end < start:
            raise ValueError("Requested positions must be positive and ordered.")
        if end > len(tokens):
            raise ValueError(f"The requested range exceeds this corpus segment ({len(tokens)} tokens).")
        rows = tuple({"index": index, "token": tokens[index - 1]} for index in range(start, end + 1))
        return {
            "corpus_id": corpus_id,
            "source_url": source_url,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "tokenizer_id": self.tokenizer_id,
            "normalization": "source text flattened to Unicode whitespace; punctuation remains attached to tokens",
            "index_base": 1,
            "target_start": start,
            "target_end": end,
            "tokens": rows,
            "context_before": tuple(tokens[max(0, start - 4) : start - 1]),
            "context_after": tuple(tokens[end : min(len(tokens), end + 3)]),
            "corpus_token_count": len(tokens),
        }


CorpusFetcher = Callable[[], Awaitable[tuple[str, str]]]


class CorpusResolution(BaseModel):
    """A selected corpus before deterministic analysis begins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_id: str
    title: str
    author: str | None = None
    source_url: str
    text: str
    extraction_method: str
    extraction_confidence: float = Field(ge=0, le=1)


class _WorkTextParser(HTMLParser):
    """Extract visible prose while excluding script/navigation-like blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0
        self._ignored_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").casefold()
        non_corpus_class = any(marker in classes for marker in (
            "noexport", "wst-header", "plainsister", "sistersitebox",
            "mw-editsection", "mw-indicators",
        ))
        if tag in {"script", "style", "nav", "footer", "header"} or non_corpus_class:
            self._ignored += 1
            self._ignored_tags.append(tag)
        if tag in {"p", "br", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_tags and tag == self._ignored_tags[-1]:
            self._ignored_tags.pop()
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            value = " ".join(data.split())
            if value:
                # Archive markup can split one printed word across spans. Join
                # adjacent alphanumeric fragments only when the HTML supplied
                # no separating whitespace; ordinary word boundaries remain.
                if (
                    self.parts
                    and not data[:1].isspace()
                    and self.parts[-1]
                    and self.parts[-1][-1].isalnum()
                    and value[0].isalnum()
                ):
                    self.parts[-1] += value
                else:
                    self.parts.append(value)


class DocumentAnalysisCapability:
    """Resolve an identifiable public text, then compute exact positions."""

    name = "document_analysis"
    operations = ("analyze_corpus",)
    descriptor = CapabilityDescriptor(
        name=name,
        operations=operations,
        supported_task_types=(TaskType.RETRIEVAL, TaskType.QUESTION),
    )

    def __init__(self, *, fetcher: CorpusFetcher | None = None, analyzer: DeterministicDocumentAnalyzer | None = None) -> None:
        self._fetcher = fetcher or self._fetch_kjv_genesis_one
        self._analyzer = analyzer or DeterministicDocumentAnalyzer()

    def supports(self, operation: str) -> bool:
        return operation == "analyze_corpus"

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return CorpusAnalysisParameters if self.supports(operation) else None

    def build_conversation_action(self, *, user_input: str, assessment: TaskAssessment) -> ConversationAction | None:
        if not assessment.requires_tools or _positions(user_input) is None:
            return None
        # A named Project Gutenberg corpus has a registered canonical text.
        # Other unnamed Bible requests remain materially ambiguous.
        if _BIBLE.search(user_input) and not (_KJV.search(user_input) or _GUTENBERG_BIBLE.search(user_input)):
            return None
        return ConversationAction(operation="analyze_corpus", parameters={"request_text": user_input})

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        if operation != "analyze_corpus" or not isinstance(parameters, CorpusAnalysisParameters):
            raise ValueError("Unsupported deterministic document operation.")
        positions = _positions(parameters.request_text)
        if positions is None:
            raise ValueError("An ordinal word or word range is required.")
        start, end = positions
        corpus = await self._resolve(parameters.request_text)
        output = self._analyzer.analyze(
            corpus_id=corpus.corpus_id,
            source_url=corpus.source_url,
            text=corpus.text,
            start=start,
            end=end,
        )
        output.update({
            "corpus_title": corpus.title,
            "corpus_author": corpus.author,
            "extraction_method": corpus.extraction_method,
            "extraction_confidence": corpus.extraction_confidence,
            # These are emitted only after the corresponding application
            # operations completed in this invocation; no model text supplies
            # them and the response formatter does not invent them.
            "execution_proof": (
                "CORPUS_RESOLVED",
                "SOURCE_RETRIEVED",
                "TEXT_EXTRACTED",
                "TOKENS_COMPUTED",
                "DETERMINISTIC_INDEX_COMPLETED",
            ),
        })
        output["retrieved_at"] = datetime.now(UTC).isoformat()
        return CapabilityResult(output=output)

    async def _fetch_kjv_genesis_one(self) -> tuple[str, str]:
        # This is a fixed HTTPS corpus registration, not model-selected web
        # access. Limit both response type and bytes before parsing JSON.
        require_network_tool("document_analysis")
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
            response = await client.get(KJV_GENESIS_ONE_URL, headers={"Accept": "application/json"})
        if response.status_code != 200 or len(response.content) > 512_000:
            raise ValueError("The registered KJV corpus source was unavailable.")
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            raise ValueError("The registered KJV corpus source returned an unsupported type.")
        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The registered KJV corpus source had no text.")
        return KJV_GENESIS_ONE_URL, text

    async def _resolve(self, request_text: str) -> CorpusResolution:
        if _GUTENBERG_BIBLE.search(request_text):
            source_url, text = await self._fetch_gutenberg_bible()
            return CorpusResolution(
                corpus_id="project-gutenberg:10", title="Project Gutenberg eBook 10, King James Bible",
                source_url=source_url, text=text, extraction_method="registered_plaintext_corpus",
                extraction_confidence=1.0,
            )
        if _KJV.search(request_text):
            source_url, text = await self._fetcher()
            return CorpusResolution(
                corpus_id="kjv-genesis-1:bible-api", title="Genesis 1, King James Version",
                source_url=source_url, text=text, extraction_method="registered_json_corpus",
                extraction_confidence=1.0,
            )
        return await self._resolve_wikisource(request_text)

    async def _fetch_gutenberg_bible(self) -> tuple[str, str]:
        """Retrieve the canonical Project Gutenberg plain-text edition once."""
        require_network_tool("document_analysis")
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
            response = await client.get(GUTENBERG_BIBLE_URL, headers={"Accept": "text/plain"})
        if response.status_code != 200 or len(response.content) > 6_000_000:
            raise ValueError("The registered Project Gutenberg corpus source was unavailable.")
        if "text/plain" not in response.headers.get("content-type", "").lower():
            raise ValueError("The registered Project Gutenberg corpus returned an unsupported type.")
        text = response.text
        start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
        end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", text)
        corpus = text[start.end():end.start() if end else None].strip() if start else text.strip()
        if not corpus:
            raise ValueError("The registered Project Gutenberg corpus had no text.")
        return GUTENBERG_BIBLE_URL, corpus

    async def _resolve_wikisource(self, request_text: str) -> CorpusResolution:
        require_network_tool("document_analysis")
        title, author = _work_reference(request_text)
        if not title:
            raise ValueError("I couldn't identify a public text to analyze. Provide a title, author, or source URL.")
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=False,
            headers={"User-Agent": "UltronCorpusResolver/1.0 (public-text analysis)"},
        ) as client:
            search = await client.get("https://en.wikisource.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": " ".join(part for part in (title, author) if part),
                "format": "json", "srlimit": 8,
            })
            payload = search.json() if search.status_code == 200 else {}
            candidates = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
            selected = _select_wikisource_title(candidates, title, author)
            if selected is None:
                raise ValueError("No suitable public corpus was found for the identified work.")
            parsed = await client.get("https://en.wikisource.org/w/api.php", params={
                "action": "parse", "page": selected, "prop": "text", "format": "json",
            })
            parsed_payload = parsed.json() if parsed.status_code == 200 else {}
            markup = parsed_payload.get("parse", {}).get("text", {}).get("*") if isinstance(parsed_payload, dict) else None
            # Wikisource frequently returns a version index. Follow one
            # archive-work link chosen from that page, rather than counting
            # index prose as the requested corpus.
            if isinstance(markup, str) and "Versions of" in markup and "/" not in selected:
                edition = _edition_link(markup, requested_title=title)
                if edition:
                    selected = edition
                    parsed = await client.get("https://en.wikisource.org/w/api.php", params={
                        "action": "parse", "page": selected, "prop": "text", "format": "json",
                    })
                    parsed_payload = parsed.json() if parsed.status_code == 200 else {}
                    markup = parsed_payload.get("parse", {}).get("text", {}).get("*") if isinstance(parsed_payload, dict) else None
        if not isinstance(markup, str):
            raise TypeError("The selected public corpus could not be retrieved.")
        parser = _WorkTextParser()
        parser.feed(markup)
        visible_text = "\n".join(part for part in parser.parts if part).strip()
        text, extraction_method, confidence = _isolate_work_text(visible_text, title)
        if len(_TOKEN.findall(text)) < 12:
            raise ValueError("The selected page did not expose enough corpus text for deterministic analysis.")
        return CorpusResolution(
            corpus_id=f"wikisource:{selected}", title=selected, author=author,
            source_url=f"https://en.wikisource.org/wiki/{quote(selected.replace(' ', '_'))}", text=text,
            extraction_method=extraction_method, extraction_confidence=confidence,
        )

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is not ActionStatus.COMPLETED or not isinstance(result.output, dict):
            return None
        rows = result.output.get("tokens")
        if not isinstance(rows, tuple | list) or not rows:
            return None
        lines = "\n".join(f"{item['index']} {item['token']}" for item in rows if isinstance(item, dict))
        title = result.output.get("corpus_title") or result.output["corpus_id"]
        return (
            f"Using {title} ({result.output['corpus_id']}), with 1-indexed "
            f"{result.output['tokenizer_id']} tokenization (punctuation remains attached):\n{lines}\n"
            f"Source: {result.output['source_url']}"
        )


class DocumentAnalysisEvidenceProvider:
    capability = "document_analysis.analyze_corpus"

    def produce_evidence(self, *, task_id: UUID, action_result: ActionResult) -> EvidenceCollection:
        output = action_result.output
        if not isinstance(output, dict):
            raise TypeError("structured document-analysis output required")
        retrieved_at = datetime.fromisoformat(str(output["retrieved_at"]))
        evidence = Evidence(
            evidence_id=uuid4(),
            source_type=EvidenceSourceType.STRUCTURED_DATASET,
            source_identifier=f"{output['corpus_id']}:{output['content_sha256'][:12]}",
            claim="deterministic token positions",
            content=str(output["tokens"]),
            reliability=EvidenceReliability.AUTHORITATIVE,
            retrieved_at=retrieved_at,
        )
        return EvidenceCollection(task_id=task_id, items=(evidence,))

    def claims(self, result: ActionResult) -> tuple[Claim, ...]:
        output = result.output
        assert isinstance(output, dict)
        rows = output["tokens"]
        return tuple(
            Claim(text=f"Token {row['index']}", claim_type=ClaimType.POSITIONAL_VALUE, expected_value=row["token"])
            for row in rows[:8]
        )

    def fallback_answer(self, result: ActionResult) -> str:
        return DocumentAnalysisCapability().format_conversation_result(result) or "Deterministic analysis result unavailable."


def _positions(text: str) -> tuple[int, int] | None:
    range_match = _RANGE.search(text)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    ordinal = _ORDINAL.search(text)
    return (int(ordinal.group(1)), int(ordinal.group(1))) if ordinal else None


def _work_reference(text: str) -> tuple[str | None, str | None]:
    match = re.search(r"\b(?:in|from|of)\s+(.+?)\s+by\s+([^?.!]+)[?.!]*$", text, re.IGNORECASE)
    if match:
        title = re.sub(r"\b(?:the\s+)?\d+(?:st|nd|rd|th)\s+(?:word|token)s?\b", "", match.group(1), flags=re.IGNORECASE).strip(" ,")
        return (title or None), match.group(2).strip()
    match = re.search(r"\b(?:in|from|of)\s+(.+?)[?.!]*$", text, re.IGNORECASE)
    if not match:
        return None, None
    title = re.sub(r"\b(?:the\s+)?\d+(?:st|nd|rd|th)\s+(?:word|token)s?\b", "", match.group(1), flags=re.IGNORECASE).strip(" ,")
    return title or None, None


def _select_wikisource_title(candidates: object, title: str, author: str | None) -> str | None:
    if not isinstance(candidates, list):
        return None
    terms = set(re.findall(r"[a-z0-9]+", title.lower()))
    author_terms = set(re.findall(r"[a-z0-9]+", (author or "").lower()))
    scored: list[tuple[int, str]] = []
    for item in candidates:
        if not isinstance(item, dict) or not isinstance(item.get("title"), str):
            continue
        candidate = unescape(item["title"])
        blob = f"{candidate} {item.get('snippet', '')}".lower()
        score = len(terms & set(re.findall(r"[a-z0-9]+", blob))) * 10 + len(author_terms & set(re.findall(r"[a-z0-9]+", blob))) * 8
        if candidate.lower() == title.lower() or candidate.lower().startswith(f"{title.lower()} ("):
            score += 60
        if re.search(r"\(.*\)$", candidate) and "/" not in candidate:
            # Keep a modest penalty for a version index, but let an otherwise
            # exact requested title win so the resolver can inspect and follow
            # its corpus links below.
            score -= 15
        if "versions" in blob or "commentary" in blob:
            score -= 12
        scored.append((score, candidate))
    return max(scored, default=(0, None))[1]


def _isolate_work_text(visible_text: str, title: str) -> tuple[str, str, float]:
    """Apply title-derived boundaries before deterministic tokenization.

    Numbered works are common in public-domain archives. Their visible page
    text usually contains a standalone ordinal followed by the work, so this
    removes archive navigation without knowing any particular work's words.
    """
    number = re.search(r"\b(\d+)\b\s*$", title)
    if number:
        ordinal_markers = (number.group(1), _roman_numeral(int(number.group(1))))
        marker_pattern = "|".join(re.escape(marker) for marker in ordinal_markers)
        start = re.search(rf"(?:^|\n)(?:{marker_pattern})[.]?\s*\n+(?=[A-Za-z])", visible_text, re.IGNORECASE)
        if start:
            text = visible_text[start.end() :]
            numbered_prefix = re.sub(r"\s+\d+\s*$", "", title).strip()
            next_work = (
                re.search(rf"\n{re.escape(numbered_prefix)}\s+\d+\b", text, re.IGNORECASE)
                if numbered_prefix else None
            )
            if next_work:
                text = text[:next_work.start()]
            return text.strip(), "wikisource_numbered_work_boundary_v1", 0.9
    heading = re.escape(title.upper())
    matches = list(re.finditer(rf"(?:^|\n){heading}[.!]?\s*\n+", visible_text.upper()))
    if matches:
        text = visible_text[matches[-1].end() :]
        return text.strip(), "wikisource_title_heading_boundary_v1", 0.82
    return visible_text, "wikisource_visible_prose_v1", 0.55


def _roman_numeral(value: int) -> str:
    """Represent a positive ordinal for archive headings without work-specific rules."""
    if value < 1 or value > 3_999:
        return ""
    symbols = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
               (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
               (5, "V"), (4, "IV"), (1, "I"))
    parts: list[str] = []
    remainder = value
    for magnitude, symbol in symbols:
        count, remainder = divmod(remainder, magnitude)
        parts.append(symbol * count)
    return "".join(parts)


def _edition_link(markup: str, *, requested_title: str) -> str | None:
    candidates = [
        unescape(candidate)
        for candidate in re.findall(r'title="([^"]+)"', markup)
        if "/" in candidate and "commentary" not in candidate.lower()
    ]
    if not candidates:
        return None
    def score(candidate: str) -> int:
        lowered = candidate.lower()
        leaf = candidate.rsplit("/", 1)[-1].replace("_", " ").casefold()
        # A version page can list many containing books.  Prefer the link
        # whose leaf is the requested work, which is a corpus-selection rule
        # derived from the request rather than a title-specific adapter.
        exact_work_leaf = int(leaf == requested_title.casefold())
        publication_penalty = sum(term in lowered for term in ("review", "journal", "works", "encyclopedia"))
        return exact_work_leaf * 100 - publication_penalty * 10 - len(candidate)
    return max(candidates, key=score)
