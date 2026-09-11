# Cognitive Planner (Step 4)

The Cognitive Planner creates a bounded, typed `ExecutionPlan` for a request that the Executive has already marked `requires_planning=true`. It is a planning boundary, not an executor: it never invokes tools, retrieves memory, verifies claims, executes code, retries a model response, or reports work as completed.

## Flow

`Executive -> TaskAssessment -> CognitiveContext -> CognitivePlanner -> ExecutionPlan`

The Runtime calls the planner once only on its `cognitive_execution_required` route when planning is required. It retains the existing `requires_future_capability` result and its honest user-facing response. The internal plan is not included in the normal chat response or TTS input.

## Contract

`ExecutionPlan` is application-owned and contains a generated `plan_id`, the context `task_id`, the task objective, a status, and validated steps. The model controls only `steps`; it cannot set IDs, objectives, status, or failure fields.

Each `PlanStep` has a `step_N` ID, bounded description and purpose, optional capability requirements, and explicit dependencies. Plans are limited to 12 steps. Descriptions are limited to 2,000 characters and purposes to 1,000 characters. Supported declared capabilities are `reasoning`, `tool`, `memory`, `verification`, `retrieval`, `computation`, and `coding`.

Dependencies must be unique, refer to another plan step, avoid self-dependency, and form an acyclic graph. A required plan with no steps is invalid.

## Model call and failures

No model call is made when planning is not required. Otherwise the planner makes at most one structured Ollama-compatible call with a JSON schema and existing model/host configuration. Context is serialized as untrusted task data; it cannot override planner instructions.

Timeout, transport, schema, constraint, dependency, and unexpected failures return a typed failed plan with no steps. The planner does not retry or synthesize a fallback plan.

## Configuration

Empty planner model and host values inherit the existing Brain/Ollama values.

- `BRAIN_PLANNER_MODEL`
- `BRAIN_PLANNER_MODEL_BASE_URL`
- `BRAIN_PLANNER_TIMEOUT` (default `30`, must be positive)
- `BRAIN_PLANNER_MAX_STEPS` (default `12`, from `1` to `12`)

## Tests

From `backend`:

    pytest tests/test_brain_planner.py tests/test_brain_runtime.py
