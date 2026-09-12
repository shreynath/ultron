# Phase I — Pretraining Scale, Architecture Optimization & Emergent Capability

Phase I adds a scientific pretraining workbench: immutable corpus diagnostics,
development-only tokenizer A/B experiments, initialization and gradient
telemetry, bounded schedules/early stopping, raw-logit generation metrics,
an append-only experiment registry, and expanded model-only evaluation.

The work does not use a third-party model, synthetic teacher answers, a
retrieval fallback, a repetition penalty, or output repair. All candidate
models are randomly initialized ULTRON decoders.

The Phase I result is deliberately negative. A valid full-line tokenizer A/B
study selected the 8,192-token tokenizer for the available corpus, but the
fresh 250k-token pilot still produced repetitive greedy text, never emitted
EOS on the fixed probes, and did not justify use of the sealed test. The older
1M-token 4,096-token run is retained as failure evidence but is not a valid
selection result because its tokenizer fit skipped six long lines.

See [PHASE_I_REPORT.md](PHASE_I_REPORT.md) and
`backend/data/ultron/reports/phase-i-report.json` for the complete evidence.
