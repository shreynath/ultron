# Working Memory and Cognitive Context (Step 3)

Working Memory is bounded, task-local state for one Ultron request. It is the state boundary between the Cognitive Runtime and future planning or action subsystems.

## Data flow

USER INPUT
  |
  v
Executive
  |
  v
TaskAssessment
  |
  v
Cognitive Runtime
  |
  v
CognitiveContext
  |- CognitiveTask: objective and authoritative assessment
  '- WorkingMemory: current input, recent history, observations, intermediate results

The Runtime still chooses either direct_response or cognitive_execution_required. Context construction does not change routing and makes no model, tool, network, or persistence call.

## Components

### CognitiveTask

Contains a request-local task ID, the objective from assessment.user_goal, and the unchanged TaskAssessment. The ID distinguishes independently created contexts with otherwise identical content; it is not exposed through normal chat, used as user identity, or persisted. The Executive remains the authority on task characterization.

### WorkingMemory

Contains:

- user_input: the current untrusted request;
- recent_history: recent user and assistant messages only;
- observations: explicit generic facts plus immutable typed execution observations produced by the Step 6 interpreter;
- intermediate_results: explicit task artifacts intentionally produced by future systems.

Observation and intermediate-result content is data, not runtime control input. Neither field is for hidden reasoning, model scratchpads, prompts, or secrets. Step 6 execution observations preserve action provenance and bounded metadata; raw capability output is not used as an unrestricted memory cache.

### CognitiveContext

Bundles CognitiveTask and WorkingMemory into one serializable Pydantic model. Every request constructs an independent context; it is never held in a global variable, written to disk, or persisted between requests.

## Bounds and retention

| State | Limit | Behavior |
| --- | ---: | --- |
| Recent history | 8 messages | Invalid entries are filtered; newest valid messages are kept |
| Observations | 32 items | add_observation discards oldest items first |
| Intermediate results | 32 items | add_intermediate_result discards oldest items first |
| User/history content | 10,000 characters | Pydantic validation rejects oversized entries |
| Artifact content | 8,000 characters | Pydantic validation rejects oversized entries |

No LLM summary or compression is used when a bound is reached.

## History is not memory

Conversation history is temporary input supplied with the request. Working Memory is temporary state produced or held during that request. Long-term memory would persist and retrieve information across requests; it is not implemented.

## Runtime integration

The backward-compatible Runtime API still accepts user input, history, and TaskAssessment. It constructs CognitiveContext internally. Future code can construct context explicitly with runtime.create_context(...) and call runtime.handle(context=context).

The normal chat API does not expose CognitiveContext, WorkingMemory, or task artifacts. It uses only the final execution answer for the existing TTS path.

## Planner use

The Step 4 planner receives `CognitiveContext` directly and treats all Working Memory content as untrusted task data. It creates an unexecuted plan only when the Executive assessment requires planning. Future actions may add typed observations and intermediate results, but tool execution, verification, long-term memory, learning, and autonomous behavior remain unimplemented.

## Tests

From backend:

    pytest tests/test_brain_schemas.py tests/test_brain_executive.py tests/test_brain_runtime.py tests/test_brain_working_memory.py tests/test_api.py
    pytest

Live Executive tests remain opt-in:

    ULTRON_LIVE_BRAIN=1 pytest tests/test_brain_live.py
