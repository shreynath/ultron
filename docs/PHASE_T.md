# Phase T — Proprietary Model Scaling and Learning Validation

Phase T extends the O/P/Q/R/S loop with an immutable model-architecture
registry, scale-campaign manifests, tokenizer profiling, memory feasibility,
and scaling evidence. It does not add a trainer: all actual updates remain in
Phase O and `FoundationTrainer`; promotion remains Phase N-gated.

## Model and resource choice

This Mac reports 25,769,803,776 bytes (24 GiB) unified memory. The first
larger model is a conservative development-scale decoder: hidden size 160,
four layers, four attention heads, FFN 640, context 64, tokenizer vocabulary
320, float32, causal MHA, RMSNorm/GELU, learned positions, and random
initialization. The constructed model has exactly **1,342,880 trainable
parameters**, verified against the instantiated MLX parameter tree.

The feasibility record estimates f32 parameters, gradients, Adam m/v, and a
conservative activation multiplier. This is an estimate, not a hardware
measurement. It leaves substantial room under half unified memory for
repeatable experiments; 10M+ models are intentionally not auto-launched.

## Mixtures and reproducibility

`DeterministicMixtureSampler` is a seeded release-level weighted sampler with
an immutable policy hash and an actual exposure report. It reports planned
weights, token counts, actual proportions, and material divergence rather than
calling concatenation weighted sampling. The first retained campaign uses one
owned release; multi-release wiring into Phase O’s packed loader remains the
next implementation boundary.

Architecture, tokenizer, campaign, mixture, and evidence artifacts are
write-once. Campaigns explicitly record dataset hashes, requested/actual token
budget, context/batch/steps, seed, checkpoint limit, and the fact that HPC
distribution is `NOT_IMPLEMENTED`.

## Retained from-scratch E2E

The actual campaign in `tests/test_phase_t.py` builds a proprietary tokenizer
from an owned corpus, randomly initializes the 1.34M model, trains through
Phase O, checkpoints with optimizer/scheduler/RNG state, and evaluates only
the local checkpoint through Phase P.

Measured result:

| Metric | One-step baseline | 16-step candidate |
|---|---:|---:|
| Tokens | 128 | 2,048 |
| Validation loss | 5.778572 | 4.209881 |
| Validation perplexity | 323.30 | 67.35 |
| Throughput | 2,187 tok/s | 3,027 tok/s |
| Capability delta | — | 0.0 |
| Generalization delta | — | unavailable |

Weights changed and validation loss decreased by 1.568691, but the held-out
model-only capability suite remained at 0.0. The correct conclusion is
`TRAINING_SIGNAL_ONLY`, not capability acquisition or proof that scaling works.
No external model or pretrained weights were used.

Artifacts: `backend/data/phase-t-e2e/test_phase_t_real_larger_from_0/phase_t/`
contains architecture, campaign, and evidence; the Phase O checkpoint and
metrics are in its `models/base/phase-t-candidate/` directory.

## Limits and next boundary

Observed: exact constructed parameter count, owned tokenizer, random
initialization, local MLX training, changed weights, lower validation loss,
checkpoint integrity, and model-only capability result. Estimated: feasibility
memory model. Not implemented: actual multi-release sampler integration inside
the Phase O loader, semantic contamination, distributed/HPC execution, a
controlled two-size campaign, and generalization results for the 1.34M run.

The next phase should create sufficiently rich, contamination-clear capability
and generalization data plus a larger token budget / repeated-seed campaign;
the present 0.0015 tokens-per-parameter is a pipeline validation budget, not a
meaningful pretraining regime.
