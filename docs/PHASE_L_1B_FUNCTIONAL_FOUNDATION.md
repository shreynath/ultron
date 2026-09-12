# Phase L.2 — 1B Functional Foundation Gate

Phase L.2 answers whether an ULTRON-owned approximately 1B model is genuinely
functional, not whether it can allocate memory or execute optimizer steps.
Only `DEVELOPMENT_SCALE` or `FORMAL_PHASE_L` records with complete lineage may
be considered. Hardware-only runs are excluded.

The machine-readable `ONE_B_FUNCTIONAL_FOUNDATION_GATE` evaluates training
stability, persistent held-out learning, generalization, native generation,
instruction following, technical/programming behavior, reasoning, and
checkpoint reproducibility. Threshold policy is versioned and visible; absent
measurements are `INSUFFICIENT_EVIDENCE`, never silently converted to failure.

Current status is `NOT_STARTED` and the scale decision is `DO_NOT_SCALE`.
The completed 1B run is execution evidence only. L.1 must first establish a
credible recipe through fixed token-budget and generation trajectories. The
only authorized next scale decision after a successful L.1 recipe is
`READY_FOR_1B`; 2B, 5B, 10B, and 20B remain policy-only.

The current L.1 trajectory has now reached 250,368 tokens on the unchanged
50.55M EOS-corrected recipe. It has not established a credible recipe:
generation has 0% EOS and punctuation-loop collapse, held-out packed-test
loss regressed to 8.391, and the fixed six-task model-only suite is 0/6.
This is development-scale negative evidence, not an eligible functional 1B
record. The gate remains fail-closed.

The intended local frontier is empirical: 1B → 2B → 5B → 10B → 20B+. Swap is
recorded as a measured performance cost, not an automatic rejection. No claim
is made that from-scratch 20B training is practical on the current M4 Pro.
