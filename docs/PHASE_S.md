# Phase S — Adaptive Learning and Training Optimization

Phase S is a bounded intervention-design layer. It turns measured Phase R
evidence into explicit configurations that Phase O can execute through the
existing Phase N control plane. It is not a second trainer, a retrieval layer,
or an automatic promotion system.

## Architecture and lifecycle

`Phase P weakness → Phase Q validated release → Phase R evidence → Phase S
intervention → Phase O proprietary checkpoint → Phase P/R evaluation → Phase S
adaptation`.

`TrainingIntervention` is immutable and hash-pinned. It records parent identity,
release IDs/hashes, curriculum and difficulty ordering, mixture and inspectable
dataset/target/difficulty/novelty/failure-repair weights, sampling policy,
optimizer/schedule/LR/batch/sequence/accumulation/decay/warmup, limits, seed,
checkpoint cadence, and early-stop policy. A changed configuration must use a
new intervention ID. `AdaptiveDecision` connects previous configuration,
Phase R evidence, policy rule, new configuration, and rationale.

The versioned policy bounds iterations, experiments, token budgets, LR and
curriculum variants, mixtures, concurrency, generalization/regression limits,
and difficulty thresholds. It remains a recommendation/persistence layer:
Phase N authorizes jobs and promotion; Phase O executes jobs; Phase P evaluates
local checkpoints; Q controls source/data provenance; R owns attribution and
generalization.

## Optimization controls

Supported, inspectable experiment variables are target density, validated
release mixture, replay metadata, difficulty ordering, LR candidates (0.5×,
1×, 2× base within policy), token/step bounds, seed, and checkpoint choice.
Target replay rejects non-diverse replay profiles. Difficulty advancement uses
aggregated success thresholds, and recommends prerequisites at low success.
Mixture weights must sum to one and every release stays hash-pinned.

Stability marks non-finite losses, gradient-bound violations, and trainer stop
events. Gradient data unavailable from a run is explicitly `unavailable`.
Early stop does **not** turn loss-only behavior into capability evidence: loss
degradation may only stop under a policy that has capability evaluation
available. Checkpoint selection ranks clean candidates by target capability,
generalization, regression safety, stability, then fewer tokens; it does not
assume the final checkpoint is best.

Phase S records retention risk from Phase P domain deltas, recommends rehearsal
when an explicit tolerance is crossed, records efficiency per token/compute,
and distinguishes insufficient replication from measured replication. The
conservative learning-curve rule only labels saturation from actual checkpoints.

## Retained real E2E

The retained test at `backend/tests/test_phase_q_e2e.py` begins with the Phase R
`NO_LEARNING_EVIDENCE` recursion result. Both Phase S candidates are real local
MLX proprietary checkpoints initialized from the exact same proprietary parent:

| Intervention | Dataset / configuration | Tokens / steps | Target Δ | Generalization Δ | Result |
|---|---|---:|---:|---:|---|
| 1 | 12-record validated recursion release, target weight 1.0, LR 0.0003 | 192 / 12 | 0.0 | 0.0 | NO_LEARNING_EVIDENCE |
| 2 | 16-record release (75% original + 25% newly authored structural variants), target weight 2.0, LR 0.0006 | 192 / 12 | 0.0 | 0.0 | NO_LEARNING_EVIDENCE |

The evidence-driven decision is `INCREASE_TARGET_DENSITY`; it also records the
new diverse release and LR change. No checkpoint is promoted. Because the two
bounded interventions still had zero target and generalization deltas, the
final recommendation is `PAUSE_FOR_REVIEW`, not more unbounded training.
The same tiny development model has changed weights in both runs, but that is
not a capability claim.

Generated retained artifacts are in
`backend/data/phase-s-e2e/test_phase_q_real_target_data_0/phase_s/`, including
policy, both interventions, adaptive decision, checkpoint selection, efficiency,
and final recommendation. Phase R contains the corresponding clean local
model-only evidence.

## Evidence and limits

Observed/measured: actual proprietary-weight training, immutable intervention
lineage, two independent Phase O runs, local Phase P evaluation, Phase R
generalization assessment, and zero measured deltas. Estimated: the E2E's
compute units are a transparent token × hidden-size proxy; memory/disk hardware
telemetry remains unavailable. Inferred: a mixture or LR change may be more
promising only as a recommendation, never as a causal conclusion.

Not implemented: arbitrary per-example sampling inside the current Phase O
packed-release loader, automated multi-seed execution, confidence intervals,
semantic contamination detection, dynamic in-run capability probes, live
teacher ablations, distributed training, and architecture redesign. The next
boundary is a controlled larger-development-model/data experiment with real
multiple-seed repetitions and data-mixture sampling support in Phase O.
