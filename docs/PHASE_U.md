# Phase U — Proprietary Learning Campaign and Capability Acquisition

Phase U changes the evidence threshold from a short optimizer smoke test to a
minimum **20,480,000 effective, non-duplicated training-token** campaign.  It
does not provide a second trainer: a native mixed sequence dataset is supplied
to the existing Phase O job lifecycle and `FoundationTrainer`.

## Retained source corpus

The retained corpus is at `backend/data/phase-u-e2e/`.  It was acquired as
specific, hashable source trees and processed into five immutable releases:

| Domain | Source | License | Train tokens | Train sequences |
|---|---|---|---:|---:|
| Natural language | CPython documentation | PSF-2.0 | 3,787,355 | 58,267 |
| Programming | CPython source | PSF-2.0 | 14,261,325 | 219,405 |
| Mathematics | NumPy documentation | BSD-3-Clause | 1,975,675 | 30,395 |
| Computer systems | CPython runtime/source | PSF-2.0 | 7,799,740 | 119,996 |
| Science/engineering | NumPy source | BSD-3-Clause | 3,356,470 | 51,638 |

This is 31,180,565 packed train tokens and 479,701 available fixed-width
sequences. Each processed document retains source URL, license, license
evidence, timestamp, content hash, document type, and domain. CPython is
pinned to `121e27c484539faaf24e6b264846217a8f4b9738`; NumPy is pinned to
`e80572f1407205a37182e3ec9a276f4ea698b163`. The source manifest records the
release and packed-shard hashes.

Exact hash isolation and document-level structural fingerprint checks pass in
the releases. They are not semantic-contamination detection; related upstream
files may still share concepts. No teacher-generated material is included.

## Tokenizer and model

The source-derived corpus requires a **384-piece** proprietary SentencePiece
BPE tokenizer when byte fallback is enabled: its ASCII alphabet plus byte and
special pieces cannot fit the former 320-piece experimental vocabulary. This
is deliberately recorded as an experimental-variable change, not hidden. The
tokenizer is trained only from the released train partitions, frozen, and
hashed in `backend/data/phase-u-e2e/tokenizer/`.

The candidate remains a scratch-initialized float32 decoder-only causal
transformer: hidden size 160, four layers, four attention heads, FFN 640,
context 64. With the 384-piece tokenizer its constructed parameter count is
**1,363,360**, rather than Phase T's 1,342,880 (the 20,480-parameter difference
is exactly the larger embedding/output vocabulary effect). It has no
pretrained weights or parent model. A true immutable zero-step random baseline
is written before candidate training.

## Native mixture and budget

The full campaign uses a seeded, immutable, no-replacement-per-release sampler:

| Domain | Requested share |
|---|---:|
| Natural language | 15% |
| Programming | 45% |
| Mathematics | 8% |
| Systems | 22% |
| Science/engineering | 10% |

The loader writes the observed exposure and sampler state. It selects release
then a deterministic shuffled sequence, preventing accidental reuse until a
release is exhausted. At batch 2 and context 64, 160,000 optimizer steps equal
20,480,000 tokens: precisely 10,000 times Phase T's 2,048-token candidate.
The corpus capacity is sufficient for this plan without planned sequence reuse.

## Checkpoints, evaluation, and decision rules

Training config hashes include the mixture identity/seed/version. Checkpoints
contain model, Adam state, scheduler, RNG, dataset position, configuration,
release, and tokenizer identities. Curve checkpoints are requested at 1%, 5%,
10%, 25%, 50%, 75%, and 100%; the zero-step baseline supplies 0%.

Phase P remains the only core capability scorer and is local-model-only. Phase
R must attribute held-out and structural/counterfactual scores, and Phase S may
recommend a bounded next intervention after evidence exists. The decision is
one of the Phase-U verdicts; validation loss alone never produces a capability
claim.

## Measured resource preflight

The initial 100-step native-mixture Phase O benchmark processed 12,800 tokens
at **15,684.91 tokens/s**, projecting approximately **1,305.7 seconds** (21.8
minutes) for 20,480,000 tokens before checkpoint/evaluation overhead. Its
observed short-run mixture is intentionally not used as the full-run exposure
claim because 200 samples are too few for a 2% tolerance; the full plan has
320,000 samples. The measured preflight cleared the resource gate, so a full
campaign was launched rather than reporting `PHASE_U_RESOURCE_BLOCKED`.

The retained candidate completed all 160,000 steps / 20,480,000 tokens in
769.9 seconds at **26,602.10 tokens/s**, with a measured MLX peak of
61,995,364 bytes and 197,196,357 bytes of campaign artifacts. Actual mixture
proportions were 15.0056% natural language, 45.0556% programming, 7.9797%
mathematics, 21.9753% systems, and 9.9838% science/engineering; the configured
2% tolerance passed and no planned sequence was reused.

The true zero-step baseline's held-out loss/perplexity was 6.084706 / 439.09;
the completed checkpoint measured 1.019794 / 2.77. Local model-only capability
scores were 0.0 before and after on the independent test suite, while structural
and counterfactual generalization deltas were both 0.0. The final verdict is
therefore **`NO_LEARNING_EVIDENCE`**—not capability acquisition. Phase R records
the attribution and Phase S recommends structurally more diverse, validated
data before another bounded intervention.

## Limits

The retained corpus is source-heavy and is not a balanced general-language
pretraining corpus. Semantic contamination detection is unavailable, Phase R
ablations remain bounded and pending, and capability/generalization/replication
claims must wait for independent local checkpoint evaluations after the full
run. The 384-token vocabulary is still small for a production language model.
