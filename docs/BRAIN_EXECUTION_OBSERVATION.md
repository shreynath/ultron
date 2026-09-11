# Execution Observation & Result Interpretation (Step 6)

Step 6 turns one actual `ActionResult` into a bounded, immutable `ExecutionObservation`. It records what the executor reported, not whether the user's objective was achieved.

## Flow

`ActionResult -> ObservationInterpreter -> ExecutionObservation -> WorkingMemory -> STOP`

The runtime exposes `observe_action_result(...)` as an explicit internal entry point. Step 6.5's isolated single-action conversation bridge uses the same interpreter immediately after its one executor result; normal chat never exposes observations. Observation interpretation does not call the planner, action executor, a capability, or an LLM.

## Contract

Every observation has an application-generated observation ID and timestamp, plus exact action/task IDs, capability, operation, action status, outcome, source, and bounded typed facts. `ObservationOutcome` maps deterministically:

| Action status | Observation outcome |
| --- | --- |
| `completed` | `success` |
| `failed` | `failure` |
| `rejected` | `rejected` |
| `unsupported` | `unsupported` |
| `timed_out` | `timed_out` |

Facts include executor-measured duration and safe structural output metadata. Stable error codes are retained for failures; raw exception messages, parameters, prompts, and raw text output are not. Numeric, boolean, and null values from a bounded, non-sensitive top-level output mapping may be retained. Oversized action output remains explicitly marked as truncated and its details are omitted.

Observations are immutable and task-local. Re-observing the same action result in the same context returns its existing observation instead of appending a duplicate. A task mismatch, conflicting action identity, or normalization error raises `ObservationProcessingError` and writes nothing.

## Guarantees and non-goals

- The existing 32-item WorkingMemory observation FIFO is used; there is no second observation store or global state.
- No LLM interpretation is required.
- No verification occurs: successful execution is not goal completion, correctness, or verification.
- No plan advancement, retry, alternate capability, replanning, persistent memory, autonomous execution, or cognitive loop occurs.
