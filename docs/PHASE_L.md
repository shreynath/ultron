# Phase L — Foundation Training Convergence & 1B-Scale Capability

Phase L separates two questions that Phase K could not answer: whether a
model can execute under memory pressure, and whether it can learn useful
model-only behavior from a qualified corpus.  The accepted Phase K corpus is
an integrity-valid pilot, not the Phase L scaling corpus.  No fair 50M/100M/
250M comparison is authorized until a new immutable 10M-token release passes
the Phase L corpus gate. That formal gate is deliberately unchanged.

## Parallel corpus and development tracks

Phase L now has explicit corpus tiers. `INFRASTRUCTURE` is below 1M tokens
and supports only pipeline/debugging work. `DEVELOPMENT` is a valid roughly
1M–3M-token release and supports controlled `DEVELOPMENT_SCALE` experiments.
`FORMAL` requires a valid release of at least 10M ULTRON-tokenizer tokens and
is the only tier that can make `FORMAL_PHASE_L` scaling claims. The run
registry records corpus ID/tier, token count, evidence class, model
parameters, train/validation/test token counts, and formal/development gate
status. A formal report is rejected unless the record has a formal corpus and
a `READY` 10M gate.

`phase-l-corpus-plan` reports token-realized domain, source-family, and
source-host concentrations and acquisition deficits. It plans only explicit
curated-source manifests; it is not a web scraper.

The sustained-run harness executes real optimizer steps and records ten-step
throughput and swap windows.  Random token IDs are intentionally allowed only
for execution/memory stability; they are never evidence of convergence,
generation quality, or capability.

Current measured status: the 995.9M-parameter harness completed 100 real
optimizer steps and is classified `SUSTAINED_EXECUTION_WITH_SWAP`. The 10M
corpus gate remains blocked: the strongest cache-derived candidate reaches
497,507 valid tokenizer tokens, not 10M. Therefore neither fair parameter
scaling nor a corpus-backed 1B training conclusion has begun.

Promotion remains strict: `RESEARCH_BEST` requires complete lineage and
reproducible evaluation; `DEFAULT_FOUNDATION` additionally requires useful
native model-only behavior.  Neither has been established for Phase L.

## Development-scale finding: EOS packing

`ultron-corpus-0.9.0-phase-l-development-v4-eos-packing` is the first valid
Tier 1 release: 1,012,215 tokenizer tokens, complete provenance and license
evidence, PII filtering, deduplication, source-family-isolated splits, and a
passed balance/integrity gate. Its formal status remains `SIZE_FAILURE`.

The first 50.55M-parameter, 100-step run exposed a packing defect: dropping
the terminal tail omitted EOS from all 1,925 packed examples. Raw model-only
generation therefore had 0% EOS and 98.44% repeated unigrams at every
checkpoint. That immutable release/run remains evidence. The separate v4
release preserves a same-document terminal EOS window (98 EOS-bearing
sequences, zero cross-document boundaries). Its controlled 100-step run still
had 0% EOS and 98.44% repeated unigrams at the final checkpoint. The defect
was real and fixed, but 51,200 training tokens are insufficient to establish
useful generation; the next diagnostic is a token-budget/schedule trajectory,
not 1B or HPC compute.

## L.1 and L.2 functional-foundation track

Phase L.1 is now the explicit training-recipe validation track: fixed corpus,
token-budget ladder, raw generation trajectory, packed held-out loss, and
model-only capability suite. Phase L.2 is the separate 1B functional gate.
It cannot pass from an execution benchmark, a parameter count, or a small
model result. Phase L.3 (2B through 20B+) is policy-only until L.2 passes.

The duration trajectory remains `DEVELOPMENT_SCALE`: at 100,352 tokens the
sealed packed-test loss improved to 8.097 from 8.342 at 51,200 tokens, but
generation was fully repetitive with 0% EOS termination. The controlled
250,368-token continuation then reached a worse sealed packed-test loss
(8.391), 0% EOS, 98.31% repeated-unigram rate, punctuation-loop output, and
0/6 fixed model-only capability tasks. It therefore does not justify 1B
functional training or scaling beyond 1B. The next step is an isolated
training-pipeline diagnostic, not an indefinite duration extension.

The subsequent diagnostic cycle validated the production objective with a
synthetic EOS-bearing overfit test and found a distinct historical packing
defect: EOS targets appeared only at absolute position 512. A new versioned
masked, variable-EOS packing artifact fixes target positions and padding loss
accounting without mutating historical evidence. Its 25,600-token recovery
run remained generation-degenerate and scored 0/6, so it is negative
development evidence and does not authorize a longer duration rung. The next
branch is a natural-segmentation recipe control, not 1B scaling.
