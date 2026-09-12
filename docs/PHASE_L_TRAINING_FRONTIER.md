# Phase L Training Frontier

| Frontier | Status | Evidence |
| --- | --- | --- |
| RAM-only execution | 100M observed in Phase K | Historical measured micro-pilot |
| Swap-assisted execution | 1B observed in Phase K | One optimizer step; not sustained training |
| Sustained 1B execution | Established with swap | 100 real optimizer steps; random-ID hardware harness only |
| Fair parameter scaling | Blocked | Requires a shared, qualified 10M-token corpus |
| Learning convergence | Not measured | Pilot corpus cannot support conclusion |

Projected token budgets must be based only on measured steady-state throughput
after a sustained run.  They remain projections, not observations.

The completed 1B run had an aggregate 116.42 tok/s over 51,200 tokens, but
this average includes a one-window 16.26 tok/s stall. Its final 50-step window
mean was 436.56 tok/s and final 10-step window was 402.86 tok/s. The observed
swap trajectory was non-monotonic (5.45 GB before, 9.85 GB peak, 6.36 GB
after), so it is not evidence of continuously escalating pressure. It has no
corpus, checkpoint, validation, generation, or capability claim.
