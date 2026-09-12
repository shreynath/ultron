# Phase K hardware frontier — swap-aware revision

Machine: Apple M4 Pro, 24GB unified memory, MLX, context 512, batch 1. Each row is a real random-input forward/backward/AdamW micro-pilot. These establish execution feasibility, not learned capability or sustained thermal performance.

| Model | actual parameters | tok/s | MLX peak | swap before → after | delta / footprint | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 50M | 50,553,216 | 5,723 | 1.70GB | 1.96 → 1.96GB | no delta | completed |
| 100M | 101,081,856 | 3,665 | 2.85GB | 1.96 → 1.96GB | no delta | completed |
| 250M | 247,983,424 | 1,433 | 5.68GB | 1.96 → 3.04GB | 983.9MB moderate | completed |
| 500M | 503,367,680 | 402 | 9.22GB | 2.96 → 3.39GB | 426.6MB moderate | completed |
| 750M | 749,826,688 | 345 | 12.93GB | 7.29 → 7.29GB | zero delta, severe retained footprint | completed |
| 1B | 995,928,960 | 48.4 | 17.05GB | 3.39 → 9.45GB | 6.06GB severe | completed |

The frontiers are separate:

- RAM-only frontier: 100M, the largest measured model with zero swap delta.
- Swap-assisted execution frontier: 1B, which completed a real optimizer step.
- Practical-throughput frontier: **not yet decided**. The 1B 10M-token wall-time projection is about 57.4 hours from one step; it needs controlled multi-step and cooled-state validation before a meaningful run is scheduled.

Swap is neither hidden nor treated as automatic failure. It is recorded as a measurable performance cost. The 750M result cannot be read as a controlled no-swap comparison because its run began after the 1B pilot had already retained substantial system swap.
