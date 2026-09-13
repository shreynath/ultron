# Phase R — Learning Attribution, Generalization, and Data Efficiency

Phase R is the evidence layer after Phase O training and Phase P model-only
evaluation. It persists a bounded, immutable account of an intervention; it
does not run arbitrary data, use a teacher at inference, or bypass Phase N's
human promotion gate.

## Architecture

`Phase P weakness → Phase Q target/release/curriculum → Phase O candidate
weights → Phase P baseline/candidate runs → Phase R analysis → Phase Q
recommendation`

`LearningExperiment` binds parent/candidate checkpoint hashes, target,
immutable releases and curriculum ordering, P evaluation IDs, config hashes,
seed, token/step/wall/compute accounting, runtime, software/Git identity, and
distributed-world-size metadata. Its manifest hash excludes only its creation
time. `PhaseRStore` is write-once: a changed payload under an existing ID is
rejected, so interrupted or resumed work must record a distinct manifest/status
rather than rewrite history.

The Phase R engine is persistence-only. Phase O remains the only training-job
adapter; Phase P remains the only model-only evaluator; Phase Q remains the
data-release authority; Phase N remains the promotion authority.

## Attribution and bounded experiments

`bounded_designs` creates a full mixture and bounded leave-one-out designs,
limited by an immutable policy's experiment count, datasets per experiment,
tokens, compute, wall time, evaluation items, and concurrency. It deliberately
does not enumerate every dataset subset. `dataset_attributions` reports the
observed `full_delta - without_dataset_delta`; its wording is explicitly
non-causal and it records `INCONCLUSIVE_CONTRIBUTION` when the ablation has not
been run.

The same manifest supports ordered/randomized/reversed curriculum variants and
replicated seeds, because it records curriculum-ordering hash and seed. This
phase records no invented confidence interval: replication is only strong
evidence once actual repeat results are supplied.

## Generalization and contamination

Independent Phase P suites may label items as `IN_DISTRIBUTION`, `STRUCTURAL`,
`COMPOSITIONAL`, `COUNTERFACTUAL`, or `ADVERSARIAL`. Phase R compares the
baseline and candidate only when both ran the exact same immutable suite hash.
It reports per-class counts/deltas and excludes quarantined tasks from credit.

The deterministic overlap detectors are versioned and preserve the matching
record and explanation:

- exact prompt/key text;
- normalized text;
- normalized 3-gram overlap (conservative review signal);
- narrow number-insensitive lexical-template overlap.

These detectors catch direct/memorization-like overlap but **do not implement
semantic contamination detection**. They never execute downloaded content,
scripts, macros, or teacher output; this phase operates only on text already
accepted by Phase Q.

## Evidence, failures, and data value

Evidence levels distinguish `NO_LEARNING_EVIDENCE`, `TRAINING_SIGNAL_ONLY`,
in-distribution improvement, generalization evidence, strong replicated
evidence, contradictory evidence, and insufficient evidence. A loss decrease
without measured held-out capability is only `TRAINING_SIGNAL_ONLY`. A changed
checkpoint with zero target/generalization delta is not a learning claim.
Contamination and regressions take precedence over apparent improvements.

Efficiency records calculate measured capability/generalization gain per token,
step, and compute unit, plus tokens/compute per newly solved item. Zero-cost
denominators are represented as unavailable, not fabricated. The transparent
data-value heuristic is:

`capability_contribution × max(0, generalization_contribution) × target_importance × evidence_confidence × novelty × source_reliability × (1 - contamination_risk) / max(1, token_cost + compute_cost)`.

Every factor is stored alongside the score and formula version. Datasets can be
useful, neutral, harmful, quarantined, championed, retired, or marked dominated
without deleting historical evidence. Champion and learning-curve schemas are
immutable; Pareto/frontier selection and statistical intervals are not yet
implemented.

Failures are persisted as no-weight-change, instability, no held-out/capability
gain, memorization/contamination, regression, data/curriculum limitations, or
evaluation limitations. The structured next recommendation can ask Phase Q to
acquire more data, change curriculum, replicate, defer, accept, or reject; it
cannot promote a model.

## Retained E2E evidence

`tests/test_phase_q_e2e.py` performs actual local MLX training, not a mock: a
12,592-parameter proprietary decoder is initialized locally, trained for one
baseline step (16 tokens), then initialized from those proprietary weights and
trained on the Phase Q recursion release for 12 steps (192 tokens). Baseline and
candidate model-only Phase P runs both scored 0.0 for the target. A separate
three-item structural/compositional/counterfactual suite also scored a 0.0
delta. Checkpoint hashes changed, but Phase R records
`NO_LEARNING_EVIDENCE`, `NO_CAPABILITY_GAIN`, zero capability/token and
capability/compute, and recommends more structurally diverse validated data.

The retained generated artifacts are under
`backend/data/phase-r-e2e/test_phase_q_real_target_data_0/phase_r/`, including
the manifest, independent suite/run references, contamination check, analysis,
efficiency, failure, and recommendation. The run is intentionally not evidence
that the model learned recursion; it is evidence that the closed loop measured
no target or generalization gain and rejected promotion.

## Current limitations and next boundary

Observed: immutable manifests, real local weights, model-only evaluations,
explicit overlap quarantine, bounded designs, and conservative no-learning
reporting. Inferred/estimated: data value and compute units are transparent
heuristics, not causal or hardware-calibrated estimates. Not implemented:
semantic contamination, sandboxed code verification, live-teacher ablations,
automated repeat execution, confidence intervals, complete Pareto dominance,
and an API/dashboard view of Phase R artifacts.

`PhaseOEngine.monitor()` exposes the latest persisted learning attribution and
next learning recommendation alongside its existing candidate/job/capability
status. The next boundary is controlled multiple-release/seed execution, then
actual replications to support any stronger attribution claim.
