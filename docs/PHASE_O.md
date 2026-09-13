# Phase O — proprietary training-job engine

Phase O connects the existing owned MLX training path to the Phase N candidate
ledger. `PhaseOEngine` adds a persistent dataset registry, complete-checkpoint
registry, durable training jobs, model-only evaluation records, task-failure
analysis, Study Queue training requirements, intervention proposals, recovery,
and a development monitor.

It reuses rather than replaces the current stack:

- `UltronModelConfig` builds a randomly initialized, decoder-only proprietary
  MLX model; it rejects pretrained weights and parent-weight initialization.
- `FoundationTrainer` performs real AdamW updates and writes atomic, resumable
  checkpoints with optimizer, scheduler, token position, and RNG state.
- `CorpusPipeline` provides immutable, provenance-aware releases; Phase O
  rejects a release with missing splits, incomplete provenance, leakage, or
  uncertain licensing.
- `FoundationEvaluator` is used in `BASE_MODEL` mode only. It loads the
  checkpoint and proprietary tokenizer directly; there is no retrieval,
  memory, tool, or external answer path.

The first implemented cycle is deliberately conservative: a queued Phase N
candidate is bound to one registered dataset and one exact training config;
the engine performs actual local training, verifies and registers the resulting
checkpoint, records the job/candidate transition, and runs model-only
evaluation. Completion alone cannot promote a candidate. Task-owned capability
and regression evidence must be recorded separately in Phase N, or the
candidate is rejected for insufficient evidence.

`run_once` is the restart-safe persistent state-machine tick: it recovers a
dead job, executes one automatically permitted bounded job, or performs one
model-only evaluation at autonomy level 4. `run_continuously` is a finite
host-side daemon loop around that tick; it never bypasses the immutable Phase N
resource, launch, evaluation, or promotion policy.

`GET /api/local-intelligence/development-status` exposes the monitor without
starting a training job. It reports unconfigured policy state honestly rather
than implying that a daemon is active.

`recover_interrupted_jobs` converts a persisted `RUNNING` job to `RESUMABLE`
only when the latest on-disk checkpoint passes `CheckpointStore.inspect`; it
otherwise records `FAILED`. Study Queue events are surfaced as training
requirements, but intentionally remain ineligible until an operator creates a
license-valid, provenance-aware corpus release. This prevents memory retrieval
or an unlicensed fetched page from silently becoming model weights.
