# Phase Q: controlled data-to-training bridge

Phase Q connects a measured Phase P weakness to an immutable corpus release
that Phase O can train. It reuses `CorpusPipeline` for provenance, privacy,
deduplication, quality filtering, split leakage checks, and immutable release
manifests; it does not introduce another trainer or an answer-serving path.

## Data path

```
Phase P opportunity → learning target → data requirement → eligible source
→ immutable raw artifact → independently validated canonical record
→ difficulty-ordered curriculum → immutable CorpusPipeline release → Phase O
```

`DataSource` requires an explicit license/permission declaration. Unknown or
unreviewed licenses cannot enter automatic acquisition. `acquire_text` is
content-hash cached and resumable, retains raw artifacts, and applies source
count and raw-byte limits. Source text is treated as untrusted data, not as
configuration or executable instructions.

Training records retain source, target, instruction/input/output, domain,
difficulty method, validation method/status, and permanent teacher provenance
when applicable. A direct normalized overlap with any evaluation prompt or key
quarantines the record before a release can be created.

## Teacher boundary

`TeacherAdapter` is an offline candidate-data interface only. It cannot be
called by the capability evaluator or runtime response path. Teacher candidates
carry provider/model/version/configuration, prompt hash, generation run, and
target. They must pass an independent deterministic verifier; disagreement is
recorded as `DISAGREEMENT` / `REQUIRES_REVIEW`, never silently accepted.

The current repository includes `DeterministicFixtureTeacher` solely to test
the interface. No external teacher was invoked in the retained demonstration.

## Parent-weight targeted training

Phase O now supports initializing a candidate from a complete ULTRON parent
checkpoint's `model.safetensors` while resetting optimizer/data position for a
different immutable release. It records the parent checkpoint in the new
checkpoint manifest and rejects architecture/tokenizer mismatches. This is not
optimizer-state resume and cannot blur training-token accounting across data
releases.

## Retained demonstration

`tests/test_phase_q_e2e.py` ran the full path into
`backend/data/phase-q-e2e/`:

1. Model-only baseline evaluation identified a programming weakness.
2. Phase Q created a recursion target and explicit data requirement.
3. Twelve user-owned CC0 sources were registered and raw artifacts cached.
4. Twelve validated source-derived records became a difficulty-ordered
   curriculum and immutable `phase-q-recursion-v1` release.
5. The release passed source-group split leakage checks and contained 8/2/2
   train/validation/test documents with zero duplicates.
6. A new candidate initialized from the proprietary baseline's weights and
   trained for 192 tokens on that exact release.
7. Model-only capability evaluation measured zero change in every domain;
   the candidate was rejected and the data contribution was recorded as zero.

This demonstrates data construction and training integration, not model
learning. The dataset is narrow and intentionally tiny; it produced no measured
programming improvement and must not be called a successful curriculum.

## Current limits

The existing curated web acquisition engine remains available for explicitly
curated sources, but Phase Q's new autonomous path accepts registered local
content rather than performing unrestricted discovery. Semantic contamination
checks, executable code sandbox verification, live external teacher adapters,
teacher ablations, dataset-champion selection, disk/preprocessing timers, and
automatic source refresh/drift analysis remain explicit future work.
