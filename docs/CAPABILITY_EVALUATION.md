# Capability evaluation

Capability suites are immutable JSON artifacts with task-owned verifiers. Exact, controlled Python, and controlled tool-trace tasks are supported. They never inject answers into a model prompt. A matrix can label a category `IMPROVED`, `REGRESSED`, `UNCHANGED`, or `INSUFFICIENT_EVIDENCE` only from reports on the same suite.

The Phase G suite has one task per category, deliberately yielding insufficient evidence rather than a premature capability label. Its current run has no foundation-model generation adapter, so empty outputs are recorded explicitly.
