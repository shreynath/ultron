# Executive cognitive layer (Step 1)

The Executive is Ultron’s cognitive triage. It does not answer the user, execute tools, plan, store memory, or verify results. It produces a validated `TaskAssessment` that later stages can consume.

## Why it exists

The chat model is a reasoning substrate, not the brain. Downstream components need typed signals (task type, tools, planning, verification, budget) rather than inferred prose.

## Input

- `user_input: str` — current user turn (untrusted)
- `history: list[dict]` — optional recent `{role, content}` turns

History is bounded to the last 8 `user`/`assistant` messages. It is conversational context only, not long-term memory.

## Output

`TaskAssessment`:

| Field | Meaning |
| --- | --- |
| `task_type` | `casual`, `question`, `conversation`, `computation`, `retrieval`, `research`, `coding`, `creative`, `planning`, `analysis`, `instruction`, `transformation`, `unknown` |
| `complexity` | `trivial` … `very_complex` (workload, not sentence length) |
| `requires_tools` | External capability needed (calc, code exec, files, web, DB, images) |
| `requires_memory` | Forward-compatible; prior/persistent context likely material |
| `requires_planning` | Multiple dependent actions / decomposition |
| `requires_verification` | Result should later be checked |
| `direct_response_allowed` | Later stage may answer without the full execution loop |
| `reasoning_budget` | `minimal` … `maximum` resource hint, not a probability |
| `urgency` | `normal` or `urgent` (explicit user signal) |
| `user_goal` | Concise restatement of the aim, not the answer |
| `rationale` | 1–2 operational sentences |
| `used_fallback` | Application-only; true when classification failed |

The model is not allowed to set `used_fallback`. Extra JSON fields are rejected.

## Classification

One structured Ollama `/api/chat` call with JSON Schema (`format`) and temperature 0. The application validates with Pydantic (`extra="forbid"`), then applies conservative normalization. There is no recursive Executive loop.

## Configuration

Empty brain values reuse the existing Ollama settings.

- `BRAIN_MODEL` (default: `OLLAMA_MODEL`)
- `BRAIN_MODEL_BASE_URL` (default: `OLLAMA_HOST`)
- `BRAIN_EXECUTIVE_TIMEOUT` (default: `30`)

Model names are not hard-coded in Executive logic. Local Ollama does not require `OLLAMA_API_KEY` for the structured call; chat replies still use the existing key check.

## Failures

Timeout, HTTP errors, invalid JSON, invalid enums, missing fields, or extra fields produce `fallback_assessment()`:

- `task_type=unknown`, `complexity=moderate`
- all capability flags false except `direct_response_allowed=true`
- `reasoning_budget=medium`, `urgency=normal`
- `used_fallback=true`

Chat response generation continues. Exception details are logged, not returned to the user.

## Normalization

1. `direct_response_allowed=true` ⇒ `requires_planning=false`
2. `urgency=urgent` only if the user text contains an explicit urgency marker
3. `complexity=trivial` cannot keep `high`/`maximum` budget (forced to `minimal`)
4. Casual/conversation over-escalation (tools+planning+verification+maximum budget, or trivial/simple with high/maximum budget) is de-escalated to a direct reply
5. `user_goal` and `rationale` are clipped; rationale is at most two sentences

Unusual but possible combinations are otherwise left intact.

## Chat integration

chat.py calls Executive.assess(...) and passes the resulting TaskAssessment to CognitiveRuntime.handle_conversation(...). The runtime selects a direct response, a supported bounded single-action capability, or a truthful limitation before TTS. Assessment and runtime metadata are not added to the HTTP response or spoken text. The frontend sends only a bounded recent history.

## Tests

```bash
cd backend
pytest tests/test_brain_schemas.py tests/test_brain_executive.py tests/test_api.py
ULTRON_LIVE_BRAIN=1 pytest tests/test_brain_live.py
```

Live tests are skipped unless `ULTRON_LIVE_BRAIN=1` so CI does not need a local model.

## Future consumption

```python
assessment = await executive.assess(user_input, history)
context = runtime.create_context(
    user_input=user_input,
    history=history,
    assessment=assessment,
)
if assessment.requires_planning:
    planner.create_plan(context)
elif assessment.direct_response_allowed:
    responder.respond(...)
```

The Step 4 planner consumes the resulting context and creates only a bounded, unexecuted plan. Tools and verification are not implemented in this step.
