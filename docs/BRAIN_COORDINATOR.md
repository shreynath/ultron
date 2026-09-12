# Bounded Execution Coordinator (Step 8)

Step 8 is the runtime integration boundary for one already-assessed Ultron
task. It composes the frozen Executive, Cognitive Runtime, conversation
capability bridge, Action Executor, Observation Interpreter, and evidence
verifier. It does not replace or absorb their responsibilities.

## Input and output

`BoundedExecutionCoordinator.run(...)` accepts either a validated
`CognitiveContext` or the existing `(user_input, history, TaskAssessment)`
inputs used by `CognitiveRuntime`. It reuses `CognitiveTask.task_id` as the
only run identifier.

It returns `CoordinatorExecution`, containing:

- the existing `CognitiveExecution` user-facing result;
- a final `OrchestrationStage`;
- append-only, typed `OrchestrationTransition` events; and
- explicit degradation reasons.

The normal chat endpoint uses the coordinator and still returns its unchanged
chat schema. `/api/brain/coordinate` is an engineering diagnostic surface for
the typed lifecycle. `/api/brain/run` retains its existing legacy runtime seam.

## Lifecycle

```
RECEIVED -> ASSESSED -> DISPATCHED
                              |\
                              | \-> BLOCKED  (no registered capability)
                              |
                              +-> EXECUTING -> RESPONDING -> VERIFYING
                                                       |          |\
                                                       |          | +-> DEGRADED
                                                       |          +---> COMPLETED
                                                       +--------------> DEGRADED

Any non-terminal stage may end in FAILED or CANCELLED.
```

Direct-response tasks use `RECEIVED -> ASSESSED -> DISPATCHED -> RESPONDING ->
COMPLETED` or `FAILED`; they do not claim an action or verification occurred. A
capability-required task with no application-registered selection ends
`BLOCKED` and preserves the existing truthful limitation response.

## What Step 8 coordinates

For one selected conversational capability only, the coordinator makes these
boundaries observable:

1. `ConversationCapabilityBridge` selects a static, application-owned action.
2. The Step 5 executor invokes that action exactly once.
3. The Step 6 interpreter records its factual observation exactly once.
4. The existing response path generates or deterministically formats text.
5. Step 7 evaluates already-produced evidence.

Planning remains descriptive and unexecuted. No model text is turned into a
capability name, operation, endpoint, or callable. There are no retries,
alternate-tool loops, or autonomous execution. The coordinator retrieves only
previously verified experience as typed untrusted planning data, then commits a
redacted experience after a terminal result; neither operation grants
capability authority or changes the completed lifecycle.

## Degradation and failures

- No matching capability: `BLOCKED`; no action is reported.
- Rejected, unsupported, failed, or timed-out action: `FAILED`; the factual
  action status remains on `CognitiveExecution`.
- Model response generation failure after a successful action: the existing
  deterministic capability formatter is used and the lifecycle is `DEGRADED`.
- Evidence that is unverified, partially verified, contradicted, or failed:
  the action result is preserved but the lifecycle is `DEGRADED`.
- A contradicted generated answer is corrected once from the registered
  evidence provider; it is never silently treated as verified.
- Cancellation is recorded as `CANCELLED` in structured logs and propagated to
  the caller. It never yields a successful result.
- Unexpected coordination errors are logged with task identity and re-raised;
  they are not converted into a fake completion.

Voice remains outside this lifecycle. Text is the cognitive outcome; existing
voice delivery may fail independently without changing a completed cognitive
outcome.
