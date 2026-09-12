# Phase I report

## Outcome

**Level 0 — teacher-forced learning only. No promotion.** No Phase I candidate
met the minimum useful-generation gate: non-degenerate greedy output, EOS
behavior, and at least one model-only verified success.

## What the measurements showed

The Phase H training split contains 749,306 characters. The legacy 320-token
tokenizer turned those into 722,452 tokens (about 1.04 characters/token) and
used only 198 vocabulary pieces. Its raw text contains no EOS by design; the
packer adds one terminal EOS per document and never joins documents.

A train-only full-line BPE A/B study found that vocabularies 320 and 512 cannot
represent the corpus character inventory with byte fallback. The valid 8,192
candidate reached 4.18 characters/token; it was selected for a controlled
pilot. The corpus, model checkpoint, tokenizer hashes, curves, and experiment
registry are retained under `backend/data/ultron`.

The 8,192-token, fresh-random 250,112-token pilot (6 layers, hidden 384, 6
heads, FFN 1024) was still greedily repetitive: 2-gram repetition was 0.774
on the fixed matrix prompt and EOS rate was zero. Temperature/top-k sampling
lowered repetition but yielded incoherent, task-irrelevant text. That is a
decoding observation, not a capability improvement, so the sealed test was
not consumed for this rejected candidate.

An earlier 4,096-token fresh 1M-token run was evaluated after validation-only
checkpoint selection and scored 0/16 on the expanded model-only suite. It is
preserved as a failed experiment, but excluded from tokenizer selection after
we found its tokenizer fitting skipped six overlong input lines.

## Decisions

`BEST_TOKENIZER`: 8,192-token `ultron-phase-i-v2`, selected on train-split
compression only. `BEST_ARCHITECTURE`: no new architecture selected; no
controlled architecture winner exists. `BEST_OVERALL_MODEL`: none eligible for
promotion. The Phase H 8M checkpoint remains a teacher-forced baseline, not a
useful generator.

The primary Phase J bottleneck is **DATA_SCALE**: the validated training corpus
is too small for this model family and quickly repeats. Secondary work should
use a much larger, provenance-controlled, diverse corpus before treating model
size, architecture, or optimizer changes as explanations for capability.

## Limits

No emergent threshold was detected. Token losses across tokenizers are not
directly comparable. Architecture, LR, warmup, weight-decay, curriculum, and
large model/data matrices were intentionally not claimed as completed: the
tokenizer/data preparation failure was found first, and the corrected pilot
did not survive the generation gate.

Machine: Apple M4 Pro with 24 GB unified memory. Phase I valid pilot throughput
was 12,314 tokens/sec; no memory pressure was recorded. Complete machine,
lineage, telemetry, leakage, capability, and generation details are in:

- `backend/data/ultron/reports/phase-i-report.json`
- `backend/data/ultron/reports/phase-i-diagnostics.json`
- `backend/data/ultron/reports/generation-report.json`
- `backend/data/ultron/reports/scaling-report.json`
