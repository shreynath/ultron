# Study Queue calibration and unattended execution

The Study Queue has three execution tiers: `SUPERVISED`, `STAGE_1`, and
`STAGE_2`. A circuit-breaker trip is a separate terminal pause state. There is
no administrative override that changes tier without a persisted calibration
round.

Before a public-source target can enter calibration, an independent baseline
prober must answer three natural variants. Any answer containing the
deterministic answer key marks the target `HIGH_COLLISION_RISK`; only a target
where all three miss is `LOW_COLLISION_RISK`. High-risk and unassessed targets
may remain operational targets but cannot count toward calibration.

Synthetic calibration targets use invented protocol identifiers and values with
locally constructed, content-addressed source documents. They are marked
`SYNTHETIC_BY_CONSTRUCTION`, are deterministically checkable, and do not claim
to measure a base model's public-world knowledge. Calibration requires a
majority of synthetic or low-collision targets.

Stage 1 requires a persisted round of at least 10 distinct human grades, at
least 90% agreement with the deterministic evaluator, and a Wilson 95%
interval recorded alongside the raw agreement. Stage 1 permits rate-limited
execution without per-item approval, but every resulting event remains
`HUMAN_REVIEW_REQUIRED` before it can count as evidence elsewhere.

Stage 2 requires a persisted round of at least 30 distinct human grades, 90%
agreement, a second independent Stage 2 round on a different day, and the
configured Stage 1 duration or cycle evidence. It is reverted to Stage 1 when
the configured spot-check cadence is reached, pending fresh recalibration.

The runner enforces persisted limits on cycles/hour, external fetches/hour and
day. It records starts, skips, rate holds, fetches, and completions. It trips
and pauses on the configured count of consecutive `LEARNING_FAILED`/`UNKNOWN`
events or a rolling source-validation failure rate. A human reset is required;
there is no timer-based resume.

## Stage 1 Calibration Evidence (Validated)

Stage 1 autonomy was unlocked following the completion of an independently graded
15-event synthetic calibration round (`5937d7e9-e763-43f7-b2bf-05137c500a46`)
at 2026-09-12T04:02:19Z.

- Sample size: 15 distinct synthetic events (`SYNTHETIC_BY_CONSTRUCTION`).
- Composition: 10 standard synthetic protocol targets (`ZK-0100`–`ZK-0109`) and
  5 deliberately fuzzy synthetic targets (`ZK-0110`–`ZK-0114`) testing compound
  attributes, deprecated vs. active status, synchronous mode conditions, header
  scoping, and directional interface prefixes.
- Human grading methodology: Blind review against persisted content-addressed
  source snapshots on local disk without consulting system logs.
- Outcomes:
  - 13/15 demonstrated learning (clean recall after context reset).
  - 2/15 failed learning (`ZK-0111` and `ZK-0114`, where lexical retrieval drifted
    on near-miss/inverted questions). Both automated evaluator and human grader
    classified these as `LEARNING_FAILED`.
- Agreement: 15/15 (1.000 raw agreement; threshold 0.900 satisfied).
- Wilson 95% score interval: [0.7961, 1.0000].
- Execution tier: Promoted to `STAGE_1`. Mandatory human review remains active
  (`review_status=HUMAN_REVIEW_REQUIRED`) on all autonomous executions.

## Collision Probes (`qwen3-8b-4bit-v1`)

Collision probes completed against the local `qwen3-8b-4bit-v1` substrate via
`open-ended-baseline-v1`:

- IPv4 address width: 3/3 natural variants answered `32 bits`; `HIGH_COLLISION_RISK`.
- HTTP 404 semantics: 3/3 variants answered missing resource; `HIGH_COLLISION_RISK`.
- Python `len`: 3/3 variants answered item/element count; `HIGH_COLLISION_RISK`.
- DNS message identifier width: 3/3 variants answered `16 bits`; `HIGH_COLLISION_RISK`.
- DNS zone transfer transport: 3/3 variants answered `TCP`; `HIGH_COLLISION_RISK`.

All five public operational targets are classified as `HIGH_COLLISION_RISK`.
They remain in the operational backlog as Level 1 demonstrations but are
excluded from calibration rounds. The active calibration set is 100%
content-addressed synthetic protocol documents.
