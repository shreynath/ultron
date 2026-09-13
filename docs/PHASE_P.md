# Phase P: model-only capability evidence

Phase P adds task-level capability evidence to the existing Phase N policy
ledger and Phase O MLX job engine. It does not treat loss, tool use, memory,
retrieval, or external-model output as proof of model capability.

## Evidence path

`CapabilitySuite` is hash-pinned and immutable. Its task-owned items carry a
domain, split/role, prompt, answer key, difficulty, provenance, contamination
state, and validation state. A changed item requires a new suite version.

`ModelOnlyCapabilityEvaluator` loads a complete ULTRON checkpoint through
`UltronGenerator` and evaluates only:

```
ULTRON checkpoint + ULTRON tokenizer + task prompt
```

It has no teacher, web, retrieval, memory, or answer-injection input. The
initial scorer is deterministic normalized exact match; a generation error is
recorded as an item error, not converted into a pass. Item results and
per-domain scores are persisted under `phase_p/`.

`PhaseOEngine.evaluate_capability_model_only` records the matched result in
the Phase N ledger. A candidate can therefore be validated only against the
same suite ID, version, and hash used by its champion baseline.

## Safety boundaries

- `validate_dataset_isolation` scans the immutable training split for direct,
  normalized prompt/key overlap and quarantines it before use. This is an
  exact-overlap guard, not a claim to detect all semantic contamination.
- External teacher artifacts are provenance/accounting records only. Their
  runtime requirement is reported explicitly and they cannot generate the
  answer consumed by the model-only evaluator.
- `TrainingConfig.maximum_wall_seconds` is enforced between optimizer steps.
  When crossed, the trainer saves a complete checkpoint and stops with a
  `WALL_CLOCK_LIMIT` reason; Phase O records that state and leaves the
  checkpoint available for evaluation/recovery.
- Promotion remains subject to the existing Phase N immutable policy and
  identified human authorization where configured.

## Development knowledge

Failures are clustered by declared domain and difficulty. Opportunities retain
the priority inputs:

```
importance × failure frequency × expected improvement
× generalization potential × confidence ÷ estimated compute tokens
```

Hypotheses, measured outcomes (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `REFUTED`,
or `INCONCLUSIVE`), traces, and next-experiment selections are immutable.
The selector skips a matching intervention/dataset/domain combination that has
already been refuted.

## Demonstration

`tests/test_phase_p_e2e.py` performs a real bounded MLX sequence:

1. Train a local from-scratch baseline for 16 tokens.
2. Evaluate a seven-domain immutable model-only suite and establish an
   operator-reviewed baseline.
3. Resume the same owned model for 176 additional tokens (192 total).
4. Evaluate the candidate against the exact same suite, record the Phase N
   decision, cluster failures, and select the next hypothesis.

The retained local run at `backend/data/phase-p-e2e/` measured zero exact-match
passes for both tiny checkpoints. Its candidate weights changed, but it was
correctly **rejected** for no measured capability improvement. The selected
next hypothesis is a proposal only; it is not automatically trained until a
new eligible, non-overlapping target dataset exists and Phase N resources
allow it.

## Current limits

The capability architecture supports all requested domains, but the retained
exercise is deliberately tiny and is not evidence that this model has learned
those domains. Initial scoring is exact-match only; rubric/judge scoring,
semantic contamination detection, disk/API budgets, targeted-data acquisition,
and an independent background service remain future work. The existing Phase O
bounded host loop remains the daemon mechanism; it does not silently launch
unapproved data construction or training.
