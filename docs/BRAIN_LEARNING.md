# Evidence-Gated Learning Architecture

## Closed loop

```
Assessment -> retrieved verified experience -> plan/authorized action
-> observation -> evidence verification -> response -> experience -> learning
```

The coordinator commits one redacted `ExperienceRecord` after each terminal
cognitive lifecycle. The correlation ID is the existing task ID, so an audit
can connect assessment, action, observation, verification, response outcome,
and learning event without creating a second trace identity.

## Persistence and recovery

`LearningRepository` uses SQLite with WAL mode and `BEGIN IMMEDIATE`
transactions at `BRAIN_LEARNING_DB` (default `data/brain_learning.sqlite3`).
It stores immutable experience payloads, current knowledge projections,
append-only knowledge revisions, contradiction records, and capability
reliability metrics. A duplicate correlation ID returns the original stored
experience; it cannot create a second record after a retry or restart.

Interrupted work is never promoted as success: the terminal lifecycle outcome
and factual execution status are stored unchanged. Learning commit failure is
logged and cannot change the already-completed cognitive result.

## Truth and promotion

Model prose, executor result, observation, evidence, verification, and learned
knowledge are separate types. `completed` action status is not verification.
Only `VERIFIED` or `PARTIALLY_VERIFIED` Step 7 results with actual claims can
create `VERIFIED` or `SUPPORTED` knowledge. Unverified, failed, and
contradicted results remain experience, not reusable truth.

Confidence is deterministic: verification status, authoritative evidence,
revalidation count, and contradictions affect it. Model self-reported
confidence is never an input.

## Contradictions and provenance

Knowledge has statuses `candidate`, `supported`, `verified`, `contradicted`,
`stale`, `retired`, and `rejected`. Conflicting claims with the same stable
claim key preserve the older revision, mark it `contradicted`, record a
contradiction row, and exclude it from retrieval. Every item retains source
experience, verification ID, capability, timestamps, and revision.

## Retrieval and memory boundaries

Only active `VERIFIED` system experience is retrieved. Retrieval uses bounded
structured token matching today; SQLite is the source of truth, not a vector
index. Results include status, confidence, provenance, timestamps, and an
`instruction_safe` marker. The planner serializes them explicitly as untrusted
task data: retrieved text cannot change policy, authorization, tools, or
system instructions.

`system_experience`, `capability`, `user_memory`, and `world` are separate
namespaces. The runtime currently learns only system experience and capability
metrics; it never derives user preferences or world facts from a tool result.

## Safety and execution integrity

Learning has no capability registry or executor access and cannot grant
permission. `ActionExecutor` caches dispatched action identities under an
async lock, so simultaneous calls with the same action ID cross a side-effect
boundary only once. A rejected prerequisite is not cached and may be retried
after its dependency actually completes. No automatic retry, tool selection,
source-code modification, package installation, or autonomous loop exists.

When application code has an explicit safe mapping for a multi-step plan, it
uses `PlanExecutionCoordinator`. The graph accepts only prebuilt
`ActionRequest`s whose task, plan, step, and dependency metadata exactly match
the validated plan. It executes ready dependencies in order, records an action
and observation for every dispatched step, blocks dependents after failure, and
propagates cancellation. Planner descriptions never become tool calls.

The read-only diagnostic endpoints are `GET /api/brain/experiences/{task_id}`,
`GET /api/brain/knowledge?query=...`,
`GET /api/brain/knowledge/{knowledge_id}/history`, and
`GET /api/brain/capabilities/{capability}/reliability`. They do not provide
execution authority.

## Data hygiene and testing

Before persistence, secret-like keys and bearer/basic credentials are redacted;
payloads are bounded. Raw user messages and arbitrary tool text are not used
as learned claims. Tests cover redaction, idempotent experience capture,
verified promotion, rejection of unverified claims, contradiction handling,
retrieval filtering, and duplicate action execution.
