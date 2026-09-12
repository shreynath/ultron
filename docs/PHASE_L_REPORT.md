# Phase L Report

Corpus: valid development release, 1,012,215 tokenizer tokens / formal target
10,000,000. It remains below the formal target and is therefore
`DEVELOPMENT_SCALE` evidence only.
Tokenizer: ULTRON Phase I v2, 8,192 vocabulary tokens.
Largest successfully executed model: 995,928,960 parameters (Phase K,
swap-assisted, one optimizer step).
Largest sustained model: 995,928,960 parameters.  The Phase L random-ID
hardware harness completed 100 optimizer steps / 51,200 tokens in 439.8 s.
It is classified `SUSTAINED_EXECUTION_WITH_SWAP`: MLX peak 17.14 GB; swap
started at 5.45 GB, peaked at 9.85 GB, and finished at 6.36 GB.  It had one
16.26 tok/s window (steps 31–40) but recovered to 402.86 tok/s in steps
91–100.  This establishes execution stability only; it does not establish
convergence, quality, generation, capability, or checkpoint portability.
Best validation/test/generation/capability: `INSUFFICIENT_EVIDENCE` for Phase
L.  No Phase L checkpoint is promoted.

Primary bottleneck: qualified corpus scale.  Secondary bottleneck: measured
throughput variability under swap.  Two immutable cache-source candidates
were measured: strict rejection reached 223,099 tokens and explicit PII
redaction reached 497,507; neither reaches the 10M gate.

The first explicit Wikimedia foundation batch acquired 35/35 sources through
the existing provenance cache. Its 780,442-token redacted candidate is also
preserved and rejected: it remains below 10M and Wikipedia exceeds the domain
concentration threshold. This is a data-acquisition increment, not a training
release.

Validation: Phase L files pass targeted lint; the full suite reports 322
passed, 10 skipped, and 2 deprecation warnings. Repository-wide `ruff check
.` is currently blocked by 1,171 pre-existing violations outside Phase L,
including vendored Voicebox sources, and is not treated as Phase L evidence.

## Corpus acquisition

The formal 10M-token gate is unchanged. The new immutable development release
`ultron-corpus-0.9.0-phase-l-development-v4-eos-packing` contains 1,012,215
ULTRON tokenizer tokens. It passed provenance, licensing, PII, deduplication,
source isolation, deterministic splitting, and balance checks; its formal
gate is correctly `SIZE_FAILURE`. The formal deficit is 8,987,785 tokens.
The prior Wikimedia v3 release remains immutable and rejected for its
Wikipedia-host concentration.

The deficit planner uses realized tokenizer-token composition. It distinguishes
minor imbalance from dominance and directs future explicit manifests toward
broad multi-domain expansion rather than repeating single-domain cohorts.

## Development experiments

Two controlled 50.55M-parameter, 100-step, 51,200-token runs used the same
tokenizer, seed, context (512), optimizer, learning rate (3e-4), and model
architecture. Both are `DEVELOPMENT_SCALE`, not formal Phase L evidence.

| Run | Packing | Final train / validation / sealed-test loss | Final raw generation |
| --- | --- | --- | --- |
| `ultron-phase-l-development-50m-v1` | Legacy tail drop; 0 EOS-bearing sequences | 6.453 / 10.332 / 8.328 | 0% EOS; 98.44% repeated unigrams |
| `ultron-phase-l-development-50m-eos-packing-v1` | EOS-preserving terminal window; 98 EOS-bearing sequences | 6.955 / 10.364 / 8.342 | 0% EOS; 98.44% repeated unigrams |

Generation was measured directly at checkpoints 25, 50, 75, and 100 using
three fixed model-only prompts. The legacy path collapsed to repeated
backticks or commas. The EOS-corrected path briefly reached distinct-1 0.03125
at step 50, then collapsed again. Neither run reached useful native generation.

## Current conclusion and next diagnostic

The old packing implementation was a verified EOS-exposure defect, not an
inference-only explanation. Fixing it alone did not restore generation at this
token budget. The evidence points first to severe undertraining and/or the
current optimization schedule; tokenizer and architecture causes remain
unisolated. A controlled longer token-budget/schedule trajectory on the fixed
EOS release is higher value than an immediate 1B or HPC run. Formal
50M/100M/250M × 10M scaling remains blocked until an immutable 10M corpus is
ready.

Validation after the preceding iteration: 323 passed, 10 skipped, 2 warnings.

## L.1 duration study

The EOS-corrected 50.55M run was resumed from its complete checkpoint and
extended to 100,352 tokens (196 optimizer steps), rather than restarted. The
final sealed packed-test loss was 8.097, an improvement from 8.342 at 51,200
tokens. Native model-only generation across 12 fixed versioned prompts at
64K, 76.8K, 89.6K, and 100.352K tokens still had 0% EOS termination, 98.44%
repeated unigrams, and distinct-1 0.015625. The six-task model-only basics
suite also scored zero in every measured category. This is honest evidence of
some held-out-loss improvement, not functional generation or capability.

## L.1 250K duration study

The exact EOS-corrected 50.55M recipe was resumed from `checkpoint-000196`
to 250,368 tokens (489 optimizer steps); no architecture, corpus, tokenizer,
optimizer, schedule, batch, context, packing, EOS, sampling, prompt, or
capability-definition variable changed. The final checkpoint is complete and
resumable. It recorded final train loss 6.911, validation loss 11.048,
sealed packed-test loss 8.391, 4,067 tok/s, and 1.79 GB peak MLX memory.
The packed-test result regressed from 8.097 at the 100,352-token rung.

The fixed 12-prompt raw trajectory was measured at 128K, 179.2K, 230.4K,
and 250.368K tokens. At the final rung, EOS rate was 0%; repeated-unigram
rate was 98.31%; distinct-1 was 0.01693; and every completion reached the
64-token cap. Outputs remained punctuation loops (primarily commas), not
language. The fixed model-only basics suite was 0/6: language, mathematics,
coding, computer science, instruction following, and reasoning all failed.

This is a persistent-collapse result. The modest late distinctness change is
not evidence of functional generation, and the worsened held-out losses do
not support the hypothesis that duration alone is repairing the recipe. The
1B functional-foundation gate remains `NOT_STARTED`; hardware evidence is not
eligible. Scaling decision: `DO_NOT_SCALE`.

The next experiment is one controlled training-recipe diagnostic, not another
unbounded duration extension or a 1B run: measure target/input alignment,
loss masking, EOS-token frequency, and packed-window sampling against a small
deterministic batch, then isolate the first demonstrated defect in a new
versioned recipe.

## Diagnostic and recovery cycle

The diagnostic framework found two confirmed infrastructure defects. First,
historical packed-shard paths were repository-relative and could fail when a
command ran from `backend/`; artifact resolution is now cwd-independent.
Second, all 80 active EOS targets in the historical v4 packing occurred only
at absolute position 512. This made its raw short-prompt EOS measurement a
poor termination test and created a positional-EOS training defect. Tokenizer
identity, special-token separation, round trips, checkpoint/configuration
contracts, target shift, and the next-token cross-entropy/mask objective all
passed. A production-architecture synthetic overfit passed at 160 steps:
loss 8.4804 → 0.3534, teacher-forced accuracy 99.41%, EOS-target accuracy
100%.

The historical corpus and v1 evidence are unchanged. A separate
`VARIABLE_EOS_MASKED_V2` packed artifact now uses within-document,
variable-length EOS-terminated segments plus explicit loss masks. Its 3,191
rows pass source/target/mask checks; active EOS positions span 3–512 rather
than being entirely position 512.

The first v2 recovery recipe was a new random-init 50.55M run for 25,600
tokens / 50 steps. It completed stably (train 6.835, validation 9.624,
4,080 tok/s) and its own masked packed test was 8.327. That test is not
directly comparable with v1 because packing and loss masks changed. Crucially,
the unchanged 12-prompt raw suite still had 0% EOS, 98.44% repeated unigrams,
distinct-1 0.015625, punctuation loops, and model-only capability 0/6.
Therefore this small recovery branch does not earn a longer duration rung.

The next single experiment is a controlled corpus/recipe comparison with
shorter natural text segments (for example paragraph/sentence-aware segments)
against the deterministic v2 segmenter, using the same masked objective and
fixed evaluation. This tests whether artificial segment boundaries—not target
alignment or loss masking—are the remaining dominant cause.

## Matched multi-branch data/recipe campaign

Four fresh random-initialization, 50.55M-parameter, 25,600-token runs used
the same optimizer, seed, masked objective, fixed 12-prompt raw-generation
suite, six-task capability suite, and each branch's sealed packed test:
variable-EOS v2, natural-boundary segments, short natural segments, and an
unpacked within-document control. Packing audits passed for all comparable
branches. Their test losses were respectively 8.327, 8.352, 8.339, and
7.788, but loss did not decide the campaign.

The first three generated 64-token punctuation loops (0% EOS, 98.44%
repeated unigrams, distinct-1 0.015625; capability 0/6). The unpacked control
emitted EOS immediately for every prompt (mean length 1, EOS 100%, capability
0/6). The comparator now explicitly treats immediate-EOS output as collapse,
not a termination or diversity improvement. Its final immutable comparison
therefore reports `NO_BRANCH_FUNCTIONAL_IMPROVEMENT` and no recommended
extension. No 50K+ continuation was run.

The originally proposed dense instructional/text filter is retained as
rejected evidence: it selected no sealed-test documents. A corrected
prose-only filter preserved the test split but had 960,431 tokens, below the
enforced 1M-token development minimum. It was preflight-rejected rather than
trained or compared. This is a data-coverage constraint, not a favorable
result.

The appropriate next work is corpus-scale and optimization analysis: build a
versioned, split-complete 5M/10M prose-and-instruction corpus; keep the same
frozen functional suite; and run a small token-budget/learning-rate sweep only
after that corpus passes the development gate. The 1B gate remains
`NOT_STARTED` / `DO_NOT_SCALE`.

## Final Phase L disposition — FOUNDATION LIMIT IDENTIFIED

**Phase L is complete: `PHASE_L_FOUNDATION_LIMIT_IDENTIFIED`.** This is a
successful resolution, not an implementation blocker. The model can learn a
deterministic EOS-bearing corpus; the corrected real-data pipeline has valid
shifted targets, masks, document boundaries, source isolation, and sealed
tests. It does not, however, produce meaningful language behavior from the
available real corpus at the tested training scales.

### Final finite campaign

The closure budget allowed at most seven comparable branches, one promotion,
and a maximum 102.4K-token promoted exposure rung. Every branch used the
frozen tokenizer, 50.55M architecture, seed, context, deterministic
generation, six-task capability suite, and sealed test contract. No branch
was promoted because each remained emergence Level 0.

| Branch | Intended variable | Tokens | Test loss | Functional outcome |
| --- | --- | ---: | ---: | --- |
| variable EOS v2 | corrected masked representation | 25,600 | 8.327 | punctuation loop |
| natural | natural-boundary packing | 25,600 | 8.352 | punctuation loop |
| short natural | short natural units | 25,600 | 8.339 | punctuation loop |
| unpacked | non-packed objective control | 25,600 | 7.788 | immediate EOS collapse |
| low LR | AdamW 1e-4 | 25,600 | 8.310 | punctuation loop |
| high LR | AdamW 6e-4 | 25,600 | 8.714 | punctuation loop |
| v2 exposure | baseline recipe, longer rung | 102,400 | 8.470 | punctuation loop |

All capability results were 0/6. The lowest-loss unpacked control is not a
winner: it generated EOS immediately (mean length 1, EOS 100%). The ranking
rules reject this as Level 0 rather than accepting it as a diversity or
termination improvement. The complete machine-readable conclusion is
`backend/data/ultron/reports/phase-l-final-disposition-v1.json`.

The 102.4K exposure rung also had a mandatory midpoint at 51.2K tokens. It
already produced the same punctuation loop (0% EOS, 98.44% repeated unigrams,
0/6 capability), so it was not eligible for promotion beyond the campaign cap.

### Data, optimization, curriculum, and capacity conclusions

The current release contains 98 documents and 1,012,215 valid tokenizer
tokens for a 50,553,216-parameter model: 0.020 real corpus tokens per
parameter. It is 3,987,785 tokens short of the 5M development target and
8,987,785 short of the formal 10M target. Its domain mix is structurally
auditable, but no high-signal/curriculum subset can retain the development
minimum and a sealed test: one dense subset had no test documents, while the
corrected prose subset had 960,431 tokens. We did not pad, duplicate, or
reinterpret either result.

The two learning-rate controls reject the narrow claim that the baseline
learning rate alone explains collapse. The 102.4K v2 exposure rung rejects
the claim that the first 25.6K tokens were the sole cause, but does **not**
establish the true emergence threshold; 102.4K is still far too little data
to infer a capacity limit. Capacity is therefore unknown, not disproven.

| Claim | Status |
| --- | --- |
| Implementation, objective, and EOS path operate | Proven |
| Current recipe/data scale yields functional language | Disproven for tested branches/rungs |
| Corpus scale and usable composition are the primary current constraint | Supported |
| 50M is intrinsically too small | Unknown |
| A larger model is justified now | No |

### Exact Phase M handoff

The first action is to use the existing source registry to publish a licensed,
deduplicated, split-complete **5M-token** development corpus—without replaying
the current 98 documents. The formal prerequisite is a **10M-token** release
that passes source-isolation, composition, tokenizer, and packing audits.
Then rerun the 50.55M model with `VARIABLE_EOS_MASKED_V2`, AdamW (3e-4,
weight decay 0.01, 10-step warmup), at 1M, 5M, and 10M update-token rungs,
evaluating each checkpoint with the unchanged frozen generation and six-task
suite. Promote only reproducible Level 3+ behavior. A larger model is not the
next action; it becomes a justified capacity experiment only after that
corpus-qualified baseline exists.
