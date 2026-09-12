# Phase K report

## Decision

**Active scaling continuation; no research-best or assistant model promoted.** Phase K measured the local M4 Pro scaling frontier, validated a 50.55M model training path, created a provenance-complete balanced pilot corpus, and empirically executed 500M, 750M, and 1B real optimizer micro-pilots. It did not reach the 10M-token corpus target or produce a fair multi-scale full-run comparison.

## Empirical findings

1. Exact instantiated accounting corrects the Phase J architecture count to 14,750,592; K1 is 50,553,216.
2. 50M and 100M short MLX optimizer runs are swap-free; 250M produced moderate swap growth (~983.9MB) but completed. 500M completed at 402 tok/s with 426.6MB additional swap; 750M completed at 345 tok/s with a retained 7.29GB swap footprint; 1B completed at 48.4 tok/s with 6.06GB additional severe swap.
3. The accepted pilot corpus has 246,370 tokens, 24 source families, complete provenance/license evidence, and clean split isolation, but is below the 10M data target.
4. The 50M/100k-token pilot is stable (4,410 tok/s) yet has sealed loss 8.194, 97.87% repeated bigrams, 0 EOS, and 0.0 capability score.
5. Evidence is insufficient for a parameter/data/compute scaling law. Phase J and K1 are not a fair comparison because both corpus and token budget differ.

## Next recommended experiment

Acquire and validate a 10M-token, license-evidenced, source-family-isolated balanced corpus; then run a same-token, same-corpus 50M/100M comparison with multiple seeds. Run controlled cooled-state multi-step 500M and 1B trials to measure sustained swap cost before choosing a meaningful large-model token budget.
