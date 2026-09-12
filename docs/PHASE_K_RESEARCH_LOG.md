# Phase K research log

## K0 — integrity and measurement infrastructure

**Hypothesis:** larger-scale work requires exact counts and fail-closed resource planning.  
**Changed:** instantiated parameter counting, complete memory/disk/FLOP preflight, lazy packed shards, registry, and corpus balance report.  
**Learned:** the legacy formula overstated Phase J by 851,968 parameters; the real decoder count is 14,750,592.  
**Next:** use instantiated counts for every pilot and preserve the legacy label only as historical context.

## K1/K3/K5 — hardware microbenchmarks

**Hypothesis:** progressively larger decoders can identify the M4 Pro practical frontier.  
**Held constant:** tokenizer vocabulary, context 512, batch 1, random-local MLX optimizer path.  
**Learned:** 50M and 100M are swap-free; 250M creates ~984MB swap growth.  
**Ruled out:** treating 24GB unified memory as a safe 250M+ AdamW training budget.  
**Next:** memory-efficient optimizer/precision/rematerialization validation before any 500M attempt.

## K1 — 50M, 100k-token corpus pilot

**Hypothesis:** the larger architecture can train stably through the full local stack.  
**Changed:** 50,553,216 parameters; balanced, provenance-complete pilot corpus.  
**Measured:** finite losses, 4,410 training tok/s, sealed loss 8.194, 97.87% repeated bigrams, 0 EOS, zero capability score.  
**Learned:** infrastructure works; 100k tokens are dramatically insufficient.  
**Next:** build the 10M-token licensed balanced corpus and run a fair multi-scale budget before interpreting generation or capability.

## K7–K10 revision — swap-assisted 500M/750M/1B micro-pilots

**Hypothesis:** static RAM estimates do not establish the largest model macOS/MLX can execute.  
**Changed:** preflight became recommendation-based; swap is classified as an observed performance property.  
**Measured:** 500M completed at 402 tok/s with moderate additional swap; 750M completed with a severe retained swap footprint; 1B completed at 48.4 tok/s with 6.06GB additional severe swap.  
**Learned:** 1B execution is empirically possible for one real step, but sustained training economics are unmeasured.  
**Next:** controlled multi-step, cooled-state 500M/1B trials; no capability conclusion until the 10M-token corpus and meaningful training budgets exist.
