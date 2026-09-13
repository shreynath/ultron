# Phase N — bounded development orchestrator

Phase N adds a persistent control plane around the existing ULTRON-owned
foundation path. It does not claim that the model has learned new capabilities,
and it does not start a daemon or a training job at import/API startup.

`app.foundation.development.DevelopmentOrchestrator` stores an append-only,
hash-chained event ledger under `ULTRON_HOME/development/`. Its policy,
candidate plans, and evaluation evidence are immutable. The ledger separately
records the candidate lifecycle (`CANDIDATE`, `TRAINING`, `EVALUATING`,
`VALIDATED`, `CHAMPION`, `REJECTED`, or `FAILED`), a training-run/checkpoint
identity, exact dataset and tokenizer hashes, a curriculum reason, evaluation
suite hashes, and the reason for every decision.

The orchestrator is deliberately bounded:

- It refuses any training plan until an immutable policy exists.
- It prevents exact repeated configurations, per-run token overruns,
  total-token overruns, and more active runs than the policy permits.
- It distinguishes base-model evaluation from model-plus-memory, tools, and
  full-system evaluation. A policy must require `BASE_MODEL` evidence for an
  intrinsic capability claim.
- It validates candidates only against a champion on the exact same immutable
  evaluation suite/version/hash. A missing metric, a regression, or an
  increased hallucination/regression rate rejects the candidate.
- A completion never promotes a candidate. The default policy requires an
  identified human operator to promote an already validated candidate.

When a policy keeps automatic launches off, the operator starts the local
training process and records its local job ID through
`record_operator_training_start`; the subsequent checkpoint completion and
evaluation still follow the same audited lifecycle.

This layer does not alter immutable evaluation data, delete a checkpoint,
train from third-party weights, or infer ground truth. The Study Queue remains
an acquisition and memory-evidence component; a future intervention planner
can turn its verified failure categories into a separately approved
`CandidatePlan`.
