# Phase L Diagnostic Framework

`phase-l-diagnose`, `phase-l-diagnose-tokenizer`, `phase-l-diagnose-data`,
`phase-l-diagnose-packing`, `phase-l-diagnose-objective`, and
`phase-l-synthetic-overfit` produce durable, machine-readable diagnostic
evidence. `phase-l-smoke-train` and `phase-l-recovery` run bounded packed
experiments; `phase-l-progress` / `phase-l-diagnostic-report` combine their
artifacts.

The framework fails closed: target or loss-mask defects block recovery
training; synthetic-overfit failure blocks real-data recovery; and a recovery
without generation-health improvement does not authorize a longer rung.

The first cycle established that historical fixed EOS packing put every active
EOS target at position 512. `VARIABLE_EOS_MASKED_V2` preserves document
isolation while giving EOS targets multiple relative positions and excluding
padding from loss. It is a new versioned packing recipe, not a reinterpretation
of historical runs.

The branch comparator ranks functional evidence before held-out loss. A
nonzero EOS rate counts only when generation is non-degenerate (mean length at
least four and EOS rate below 95%); an immediate-EOS response is classified as
collapse, even though it can superficially improve distinctness and repetition
metrics. The matched natural, short-natural, unpacked, and v2 experiments are
serialized in `data/ultron/reports/phase-l-branch-comparison-25k-v3.json`.

The final resolution adds two fixed-representation AdamW learning-rate
controls (1e-4 and 6e-4) and a 102.4K-token corrected-recipe exposure rung.
The status command reads `reports/phase-l-final-disposition-v1.json` when it
exists, so Phase L exposes its terminal disposition and Phase M handoff rather
than leaving the current programme in an indefinite diagnostic state.
