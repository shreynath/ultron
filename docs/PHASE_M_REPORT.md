# Phase M report

## Current state

`PHASE_M_CAPACITY_LIMIT_IDENTIFIED` (development-scale conclusion)

Phase L remains closed with its fixed disposition,
`PHASE_L_FOUNDATION_LIMIT_IDENTIFIED`. No Phase L branch has been reopened.

Phase M registered 30 new full, public-domain Project Gutenberg snapshots, all
with licence evidence and content-addressed cache records. The current
inventory has 137 included sources, 6,924,246 eligible tokenizer tokens, and
62 explicit exclusions: 13 missing/hash-mismatched cache records and 49
superseded snapshots of the same canonical source. The duplicate handling keeps
the largest hash-verified snapshot per canonical URL; it does not increase
scale by replaying duplicate text.

The complete machine-readable inclusion/exclusion evidence is
`backend/data/ultron/reports/phase-m-source-inventory-v2.json`.

## Corpus construction contract

The executable contract is implemented in
`backend/app/foundation/phase_m.py`; the policy is recorded in
[`PHASE_M_CORPUS_SPEC.md`](PHASE_M_CORPUS_SPEC.md). `phase-m-build` only uses
licence-evidenced, hash-verified cache entries from the existing acquisition
registry. It writes a source-decision manifest, immutable corpus release,
tokenized shards, split report, quality report, deduplication report, and
licence/source report. It rejects an undersized or failed audit rather than
silently promoting it.

The quality audit measures source distribution, document/sentence lengths,
punctuation and alphabetic ratios, code/markup/URL proportions, special-token
counts, vocabulary coverage, duplicates, boilerplate spans, shard hashes,
EOS/mask row structure, and split leakage.

## Training state

The corrected immutable release is
`ultron-corpus-1.0.1-phase-m-development-5m`: 6,399,057 tokenizer tokens
(5,178,039 train / 592,338 validation / 628,680 sealed test), 128/4/3
documents, and a passing audit with no recorded failures. The preceding
1.0.0 build is retained as rejected evidence because its over-strict long-text
quality heuristic left it under-sized.

The exact Phase L 50.55M architecture and recipe were trained from fresh random
initialization for 1,000,448 then 5,000,192 update tokens. The second rung is
an exact optimizer/scheduler/RNG continuation of the first. It completed with
training/validation losses of 5.644/6.539, no gradient clipping, and no early
stop. Sealed test loss was 6.164 across all 2,344 test sequences (perplexity
475.4). These are learning signals, not a capability promotion.

At 5M exposure the fixed generation prompts still had 85.4% repeated tokens
and 82.2% repeated bigrams. All six predeclared model-only basic capability
tasks scored zero. This is Level 0: output terminates more reliably than at 1M,
but remains repetitive and does not perform a basic requested behavior.

## Next action

Do not advance to a 10M corpus/training rung under this frozen 50.55M,
from-scratch recipe. The predeclared 5M stop trigger fired: stable optimization
and a valid corpus did not yield even Level 1 functionality or a single frozen
capability pass. The precise conclusion is scoped to this configuration and
data exposure, rather than a claim about language modeling generally. Any new
research branch requires an explicitly approved changed capacity or training
regime; it must not be labelled a continuation of this control experiment.
