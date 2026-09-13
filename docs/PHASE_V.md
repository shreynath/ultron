# Phase V — Capability Acquisition Laboratory

## Result

**REPLICATED NARROW DIRECT CAPABILITY ACQUISITION; NO GENERALIZATION CLAIM.**

Five real from-scratch campaigns were run locally. All use the frozen Phase U
decoder (1,363,360 float32 parameters), frozen 384-piece tokenizer, AdamW,
batch size 2, context length 64, and exactly 160,000 steps / 20,480,000
training tokens. No external model weights or teacher-generated data were
used. The first four conditions use seed 17; the replication uses independent
model seed 19 and sampling seed 20260913.

| Condition | Requested / observed target fraction | Direct held-out | Other four families | Total exact successes |
| --- | --- | --- | --- | --- |
| Zero-step baseline | 0% | 0/24 | 0/96 | 0/120 |
| V-A general-only control | 0% / 0% | 0/24 | 0/96 | 0/120 |
| V-B | 10% / 9.994% | 0/24 | 0/96 | 0/120 |
| V-C | 25% / 24.924% | 16/24 | 0/96 | 16/120 |
| V-D | 50% / 50.024% | 14/24 | 0/96 | 14/120 |
| V-C replication, seed 19 | 25% / 24.963% | 6/24 | 0/96 | 6/120 |

The direct family is 24 examples from the immutable target release's `test`
partition. The other 96 items are 24 each of surface-form, out-of-range
structural, reversed-template counterfactual, and three-operand compositional
addition. At the final checkpoint they remain zero for every condition. Thus
the data supports a learned response on the narrow withheld target template,
including a second initialization, but it does **not** support a generalized
addition capability.

The 95% Wilson interval over all 120 exact items is 8.38–20.56% for V-C,
7.08–18.63% for V-D, and 2.31–10.48% for the replication. The direct effect
is non-monotonic between 25% and 50%, so density has not established a simple
scaling law. General-only validation loss is worse than the 0% control for all
target mixtures; this is tracked as a trade-off and is not used as a
capability proxy.

## Data and evaluation limits

The target release is `phase-v-addition-v1`: 100,000 unique locally authored,
CC0-1.0 structured addition documents (7,814,705 tokenizer tokens), spanning
the Cartesian range `left=0..999`, `right=0..99`. It is not a small repeated
lookup set, but it is still one synthetic template family. Exact content-hash
separation is enforced; semantic contamination detection is explicitly
limited. The result must therefore remain narrowly scoped to this curriculum
and evaluator.

## Reproducible artifacts

All execution artifacts are local and immutable under
`backend/data/phase-v-e2e/`:

- `target-manifest.json` — target release and tokenizer hashes.
- `evaluation-suite.json` — frozen 120-item deterministic suite.
- `runs/<condition>/mixture-exposure.json` — actual token counts/proportions.
- `runs/<condition>/models/base/<condition>/` — checkpoints at 0, 1, 5, 10,
  25, 50, 75, and 100 percent of training.
- `runs/<condition>/capability-curve.json` — every local model output,
  deterministic grade, Wilson interval, and general validation result.

`backend/scripts/phase_v_evaluate.py` constructs the zero-step checkpoint and
does local greedy decoding only. `backend/scripts/phase_u_train.py` executes
the fixed-budget training campaigns and records sampler exposure.
