# Phase K — foundation scaling and compute frontier

Phase K preserves every Phase J artifact and uses the Phase I/J 8,192-piece ULTRON tokenizer unchanged (`0.5.0-phase-i-v2`, SHA-256 `d37ad9d…38a19f0`). It adds a local-only scaling path: no external model weights, teacher outputs, inference APIs, distillation, or synthetic LLM data can enter the model, corpus, checkpoint, or evaluation lineage.

## Revision: swap-aware scaling continuation

Phase K remains **ACTIVE — SCALING CONTINUATION**. Swap is an execution and performance resource, not an automatic model-size ceiling. Preflight now returns `RUN`, `RUN_WITH_SWAP`, `RUN_WITH_OPTIMIZATION`, `HIGH_COST_RUN`, or `LIKELY_OOM` with low-confidence planning estimates. Only a real failed execution establishes empirical infeasibility. Every micro-pilot records swap before/after/delta and its observed footprint; future multi-step benchmarks also record peak swap.

## Implemented controls

- Exact parameter inspection from an instantiated MLX parameter tree, with embedding, attention, FFN, normalization, and output-head breakdowns. Phase K corrects the historical Phase J configuration estimate: the preserved `~15.6M` label corresponds to an instantiated 14,750,592-parameter decoder. No Phase J artifact is rewritten.
- Mechanically derived 50M/100M/250M/500M/1B candidate configurations; their labels are never trusted in place of a calculated count.
- Fail-closed resource preflight covering parameters, gradients, AdamW state, activations, temporary tensors, checkpoint/disk budgets, approximate training FLOPs, and optional duration estimate.
- Lazy packed-shard sequence access using byte offsets, so the trainer no longer needs to materialize a release’s token arrays in Python memory.
- Append-only Phase K experiment registry, fixed Phase J baseline record, explicit failure classes, and fair-comparison analysis that refuses to fit a scaling law from incompatible runs.
- Corpus token/domain reporting, source-family/URL/fingerprint isolation checks, provenance and license-evidence coverage, and a diversity gate.
- CLI commands for exact model counting, Phase K preflight, corpus reports, scale-ladder derivation, and experiment analysis.

## Data status

`ultron-corpus-0.6.0-phase-k-balanced-pilot-v4` is a 246,370-token **balanced pilot**, not the requested 10M-token general corpus. It has 31 documents, 24 source families, 100% provenance coverage, 100% separate license-evidence coverage, and passing source-group/URL/near-duplicate isolation. The largest source domain is 42.88%; realized category token shares are TECHNICAL 19.84%, GENERAL 22.31%, CODE 14.22%, MATH 13.79%, SCIENCE 15.60%, INSTRUCTION 9.76%, and ENGINEERING 4.48%.

Earlier Phase K corpus candidates are retained as rejected evidence: v1 remained Gutenberg-dominant and had incomplete license evidence; v2/v3 remained above the configured domain concentration threshold. None trained a model. The Phase J leaked v1 corpus remains permanently excluded.

The 10M, 25M, 50M, 100M, 250M, 500M, and 1B corpus targets are supported by the release/shard/streaming interface but are **not yet reached**. Phase K does not call the pilot corpus a general foundation corpus.

## Experimental policy

Training uses packed train shards only; validation uses validation shards only; a copied immutable test split is created only after checkpoint selection. Generation and capability records are always labelled `MODEL_ONLY` unless another mechanism is explicitly supplied. A pilot must show finite loss/gradients, stable memory, checkpointing, resume compatibility, and native generation before a fuller run. Swap-heavy models are not treated as locally feasible.

Research promotion requires reproducible training, complete provenance, valid lineage, and sound evaluation; assistant promotion also requires meaningful model-only capability. Neither is automatic from lower loss or more parameters.
