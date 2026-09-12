# Phase J — scaled foundation pretraining and capability emergence

## Scope and boundary

Phase J continues the verified Phase I `0.5.0-phase-i-v2` 8,192-piece ULTRON tokenizer. It does not load, distil, query, or use the weights, outputs, or training data of a third-party model. The selected model is an ULTRON-owned six-layer decoder, random initialized with seed 97. Its checkpoint manifests declare `ULTRON_FROM_SCRATCH`, `RANDOM`, `NONE`, and no parent model; those fields are verified before generation or evaluation.

This phase is a reproducible scaling experiment, not a claim of human-level or generally useful capability. A checkpoint can be promoted only after independent held-out loss, generation, task-owned capability, safety, and regression evidence. Phase J makes no promotion.

## Immutable data release

The valid release is `ultron-corpus-0.5.0-phase-j-scale-v2`:

- 185 accepted documents; 6,664,882 normalized raw characters; 1,914,488 tokenizer tokens; 1,363,554 train-partition tokens.
- 84 source families; source-group, URL, and near-duplicate overlap checks all passed across train/validation/test (152/12/21 documents).
- Provenance coverage is 1.0. License-evidence coverage is only 0.664865: inherited local-authored records do not uniformly carry a separate `license_evidence` field. This is reported rather than inferred away.
- Every fetched source was explicitly requested through the local acquisition cache, with URL, retrieval record, content hash, licence metadata, parser, and acquisition version. New material includes public-domain Project Gutenberg texts, PSF-2.0 CPython tutorial material, and existing permitted technical/science/code records.
- Long documents were segmented only for sampling. Their source group stays intact, so no segment can cross a split boundary. Packed training uses `context-512`, appends EOS per document, and drops incomplete tails rather than crossing documents.

`ultron-corpus-0.5.0-phase-j-scale-v1` is retained as an invalid audit artifact. Its URL leakage check found duplicate Gutenberg URLs from two acquisition jobs in different partitions. It was never packed or trained.

The target mixture was not met: GENERAL is 80.54% of documents (and 1,696,332 tokens), CODE 5.41%, MATHEMATICS 2.70%, INSTRUCTION 2.16%, SCIENCE 2.16%, and ENGINEERING 1.08%. Gutenberg represents 53.51% of documents by source domain. Consequently this release is intentionally labelled literature-heavy and unsuitable as a flagship broad-coverage corpus.

## Training and resources

The primary measured configuration is 15,602,560 parameters (8192 vocabulary, 384 hidden units, 6 layers, 6 heads, 1024 MLP, 512 learned positions), batch 1, AdamW, gradient clip 1.0, 500-step warm-up, and cosine decay. Primary and 25M secondary resource preflights were SAFE on the 24GB Apple M4 Pro host. A 12-step benchmark of the exact primary configuration measured 22,443.5 tokens/s, 750,761,452 MLX peak bytes, and zero swap growth.

The actual run `ultron-foundation-0.6.0-phase-j-15m-10m-v1` consumed 10,000,384 tokens in 516.17 seconds (19,374.3 tokens/s). It stored full resumable checkpoints approximately every million tokens, along with optimizer, scheduler, RNG, hardware, model, tokenizer, and immutable dataset lineage. The 1M and 5M analyses use checkpoints from this one continuous 10M-token run; they are not falsely presented as independent trials.

## Validation and assessment

Development validation selected the final saved checkpoint before sealed-test evaluation. Sealed test windows were then copied immutably from the untouched test partition and evaluated at ~1M, ~5M, and 10M tokens. A freshly built random initialization of the exact architecture was also measured on the same sealed windows.

| Model state | Tokens | sealed loss | sealed PPL |
| --- | ---: | ---: | ---: |
| fresh random initialization | 0 | 9.1589 | 9498.2 |
| checkpoint-001953 | 999,936 | 5.6356 | 280.2 |
| checkpoint-009765 | 4,999,680 | 5.3870 | 218.6 |
| checkpoint-019532 | 10,000,384 | 5.3759 | 216.1 |

This is a real held-out loss reduction. It does not establish reliable language, instruction following, arithmetic, coding, scientific, or reasoning capability. Model-only greedy outputs remained degenerate: final raw probes had 49.47% repeated bigrams, zero EOS endings, and 6.85 mean token-entropy bits. The task-owned capability suite scored 0.0 in every tested category.

## Explicit non-results and next gate

No live cognitive-execution integration is run in this phase: the prerequisite general-capability gate failed, and exposing this model as a decision-making component would be misleading. The model is not promoted, registered as a general local assistant, or allowed to affect memory, planning, actions, or self-improvement.

The next experiment must first acquire and validate a substantially more balanced, license-evidenced corpus (especially code, math, science, engineering, instructional, and dialogue material), preserve strict source-family isolation, and repeat at least multiple seeds before any architecture or capability claim. The 5M/10M *corpus-size* tiers were not reached: this release is 1.91M tokenizer tokens, although the training compute budget did reach 10.0M consumed tokens through deterministic re-use of the training partition.

## Artifacts

- `backend/data/ultron/corpora/releases/ultron-corpus-0.5.0-phase-j-scale-v2`
- `backend/data/ultron/corpora/tokenized/ultron-corpus-0.5.0-phase-j-scale-v2/d37ad9d016f65792/context-512`
- `backend/data/ultron/models/base/ultron-foundation-0.6.0-phase-j-15m-10m-v1`
- `backend/data/ultron/reports/phase-j-data-report.json`
- `backend/data/ultron/reports/phase-j-scaling-report.json`
- `backend/data/ultron/reports/phase-j-generation-report.json`
- `backend/data/ultron/reports/phase-j-capability-report.json`
- `backend/data/ultron/experiments/phase-j.jsonl`
