# Cognitive Runtime (Step 2)

The Cognitive Runtime turns a validated Executive assessment into a truthful execution decision. It constructs a bounded CognitiveContext for the current request, then either uses the existing direct-response generator or records that the task requires a future cognitive capability. Step 6.5 adds `handle_conversation(...)`, an explicit integration entry point that may hand a capability-required request to the isolated single-action bridge after the regular route decision.

## Data flow

USER
  |
  v
CHAT ROUTER
  |
  v
EXECUTIVE
  |
  v
TaskAssessment
  |
  v
COGNITIVE RUNTIME
  |
  v
COGNITIVE CONTEXT / WORKING MEMORY
  |---------------------- direct_response --------------------> existing LLM -> existing TTS
  |
  |-- cognitive_execution_required --> optional bounded planner --> future tools / memory / verification

When planning is required, Step 4 may create one internal, unexecuted plan. The runtime still does not claim that work on the unavailable execution branch occurred.

## Input and output

CognitiveRuntime.handle receives:

- the untrusted user input and bounded conversational history;
- a validated, normalized TaskAssessment from the Executive;
- an optional existing response system prompt.

It creates an internal CognitiveContext containing the Executive objective,
assessment, current input, and bounded recent history. It returns CognitiveExecution:

- route: direct_response or cognitive_execution_required;
- status: completed, requires_future_capability, or failed;
- the original assessment, unchanged;
- answer: an existing-model direct reply or a bounded, honest capability message;
- required_capabilities: ordered capability labels needed by the task.
- plan: an internal `ExecutionPlan` only when planning was required; it is not a completion claim.

The runtime does not accept arbitrary dictionaries as assessment state and it never parses user_goal or rationale as control instructions. Future callers may prebuild a context with runtime.create_context(...) and provide it to handle(context=...).

## Routing rules

Routing is deterministic and makes no model call:

1. Any explicit requirement for planning, tools, memory, or verification takes precedence and routes to cognitive_execution_required.
2. An assessment with no required future capability and direct_response_allowed=true routes to direct_response.
3. Any remaining non-direct assessment also routes to cognitive_execution_required.

This preserves Executive fallback behavior: its fallback assessment has no capability requirements and allows a direct response. A direct response generator failure is reported as failed, which is distinct from requires_future_capability.

## Current capability matrix

| Capability | Step 2 state |
| --- | --- |
| Executive assessment | Implemented |
| Direct model response | Implemented |
| Existing TTS / voice delivery | Existing downstream behavior |
| Planning | Internal, unexecuted plan only |
| Single-action weather lookup | Step 6.5 conversational bridge only |
| General tool execution | Future |
| Memory | Future |
| Verification | Future |

For a task requiring a future capability, the base runtime returns an honest bounded message and does not invoke the response generator. The conversation integration may replace that message only after one explicitly registered capability returns an actual result. This prevents claims that research, code execution, document retrieval, or verification happened when they did not.

## Chat integration

Both text and voice chat use this sequence:

1. Executive assesses the user input.
2. Runtime receives the original input, history, and assessment; a capability-required request may use one supported bridge action.
3. The router passes the result-grounded execution answer into the existing voice-delivery path.

Only safe route/status metadata is added to the normal chat API response; plans, actions, observations, and result payloads are not.

## Observability and failure behavior

The runtime logs route, task type, complexity, required capability flags, fallback use, status, and latency. It does not log raw user input, secrets, prompts, or hidden reasoning.

Direct-response generation failures return a safe failed execution. Requests requiring unavailable capabilities return requires_future_capability even if their optional plan was created; this is a truthful route decision, not an internal error.

## Tests

Run the focused suite from backend:

    pytest tests/test_brain_schemas.py tests/test_brain_executive.py tests/test_brain_runtime.py tests/test_brain_working_memory.py tests/test_brain_planner.py tests/test_api.py

Run all backend tests:

    pytest

Live Executive tests remain opt-in:

    ULTRON_LIVE_BRAIN=1 pytest tests/test_brain_live.py

## Learning integration

The coordinator supplies the production runtime with durable learning storage.
After assessment it retrieves only verified, provenance-bearing system
experience and places it in bounded Working Memory as untrusted reference data.
The planner receives that data in its existing task payload and cannot treat it
as policy, authorization, or commands. After any terminal lifecycle, the
coordinator records a sanitized experience and promotes only Step 7
evidence-verified claims. This post-execution step cannot change the route,
action, verification result, or lifecycle outcome. See
[BRAIN_LEARNING.md](BRAIN_LEARNING.md) for persistence and confidence rules.

## Relationship to Step 8

Step 8's `BoundedExecutionCoordinator` now uses this runtime as its routing,
response, and verification integration point for normal chat.
`handle_conversation(...)` remains the stable public entry point and delegates
to that coordinator. The runtime continues to own no lifecycle state machine:
it exposes the existing application-owned capability selection, one-action
dispatch, and result-completion seams so the coordinator can record real
transitions without duplicating those subsystems.

## Deliberately not implemented

The runtime does not execute plans, general tools, memory retrieval, learning,
agent loops, autonomous behavior, model routing, or new frontend and voice
behavior. The Step 6.5 bridge is the narrow exception: one registered, safe
action may run after routing and immediately stops after its observation. Step
8 makes that bounded path observable; it does not turn it into a multi-action
execution loop. Future execution subsystems attach at the
`cognitive_execution_required` boundary and consume the existing assessment
and optional plan.
