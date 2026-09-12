# Model scaling

The scaling harness now contains legal candidate shapes through approximately one billion parameters, but it does not infer feasibility for unmeasured configurations. Its recommendation report labels only measured results and explains feasibility, memory pressure, throughput, or failure.

Measured at context 128, batch 1, three steps on the local M4 Pro: 738,432 parameters ran at 22,699.84 tokens/s with 41.36 MB MLX peak; 8,426,368 parameters ran at 11,462.23 tokens/s with 262.18 MB MLX peak. Both had zero measured swap delta. These are short benchmark measurements, not claims about long-run training cost or model quality.
# Phase G update

Phase G added measured 8M–~91M short MLX runs plus a 128–4096 context sweep under a fixed 4096-token measurement budget. The raw machine-readable artifacts are under `backend/data/ultron/scaling`; measured feasibility does not establish useful learning at those scales.
