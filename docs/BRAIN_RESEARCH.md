# Internet Research and Continual Knowledge

## Runtime boundary

Public-web research is an explicit `web_research.research` capability. It is
selected only for an Executive `RESEARCH` task and remains subject to the
existing capability registry, action allowlist, timeout, observation, and
coordinator lifecycle. It is not a generic HTTP or download endpoint.

The optional control API requires `BRAIN_RESEARCH_API_TOKEN` via
`X-Ultron-Research-Token`. Without that deployment secret, task submission and
audit control endpoints return no authority. Normal conversation still uses the
registered, bounded capability.

## Pipeline

```
question -> fixed search provider -> safe fetch -> SourceRecord -> WebClaim
-> WebEvidence -> corroboration/contradiction -> world knowledge -> retrieval
```

A webpage is retained as a source record; extracted text is an observation;
claims and evidence have independent IDs and source provenance. Web content is
always data. Injection-like passages are excluded from deterministic claim
extraction and cannot alter instructions, authorization, capabilities, or
network policy.

## Network safety

`SafeWebClient` permits only public `http`/`https` URLs on standard ports. It
rejects credentials, validates DNS before every request and redirect, blocks
private/loopback/link-local/reserved IPs, limits redirects, response type,
response size, and timeout. It accepts text/HTML only and never executes or
installs downloaded content. Search is a single fixed DuckDuckGo HTML discovery
endpoint; model output cannot select a provider or widen the network boundary.

## Evidence, confidence, and freshness

Source classification provides modest evidence weight, never automatic truth.
Verified world knowledge requires matching claims from at least two distinct
domains. A single retrieved source is only `SUPPORTED`; conflicting polarity is
stored as an unresolved contradiction and is not promoted. A known content hash
at the same canonical URL is retained as an `UNCHANGED` retrieval without
duplicate claim ingestion.

Freshness is question-sensitive: current/today/latest questions use a short
interval, API/version questions a medium interval, and other claims a longer
interval. `stale_knowledge()` records a `STALE` revision; the explicit
revalidation endpoint creates a bounded task for each selected claim. There is
no always-running crawler or autonomous rescheduling loop.

## Persistence and audit

The existing learning SQLite database adds `research_tasks`, `research_results`,
`web_sources`, `web_claims`, `web_evidence`, `web_contradictions`, and
`source_metrics`. WAL mode and immediate transactions remain in use. On startup
nonterminal tasks are marked failed as interrupted—never completed—and no
uncommitted research result promotes knowledge.

Read-only/restricted inspection is available through the research task result,
source-reliability, existing knowledge revision, and experience endpoints.
Research results contain the full source/claim/evidence/contradiction graph.
Normal runtime retrieval requests only `VERIFIED` knowledge. Audit callers can
explicitly request `SUPPORTED`, `STALE`, or `CONTRADICTED` statuses through the
knowledge endpoint; these states are never silently mixed into factual context.

## Test policy

The default suite uses mock search/fetch providers. It covers corroboration,
world-knowledge persistence/retrieval, unchanged-page caching, injection
isolation, and private-network blocking. Live web validation is opt-in and
bounded; it is never part of normal tests.
