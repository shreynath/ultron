# Action Executor & Capability Boundary (Step 5)

Step 5 adds a narrow internal boundary between planning intent and an actual application capability. It does not make Ultron an autonomous agent.

## Flow

`Executive -> TaskAssessment -> CognitiveContext -> ExecutionPlan -> ActionRequest -> ActionExecutor -> Capability -> ActionResult -> WorkingMemory`

`ExecutionPlan` and `PlanStep` remain descriptive. `ActionRequestBuilder` creates a request for one selected step with an application-generated action ID, the context task ID, and application-controlled plan metadata. The executor dispatches only that request. It never inspects a plan to execute other steps, traverses dependencies, calls the planner, or invokes an LLM.

## Guarantees

- One action is dispatched at most once per executor invocation.
- Capabilities are application-owned objects in an explicit `CapabilityRegistry`; names and operations never trigger dynamic imports or arbitrary callable lookup.
- The executor uses an explicit capability allowlist. A registered capability is not executable unless policy permits it.
- Parameter models validate untrusted parameters with strict Pydantic schemas before dispatch.
- The action task ID must match the supplied `CognitiveContext`.
- Dependencies are only checked against task-local executor result records. They are never run automatically.
- There are no retries, fallback capabilities, automatic re-planning, recursion, queues, or autonomous loops.
- Execution is bounded by `BRAIN_ACTION_TIMEOUT` (default 30 seconds, maximum 60 seconds).
- `completed` is returned only after the capability actually returns `CapabilityResult`. Rejections, unsupported operations, failures, and timeouts have separate typed statuses and stable error codes.
- The executor records only task-local action status needed for prerequisite checks. Step 6 owns conversion of an `ActionResult` into WorkingMemory observations. Action logs record only identifiers, names, status, duration, and an error code.

## Current executable capabilities

The default low-level executor remains empty. Step 6.5 creates a separate, application-owned conversational registry for the single-action bridge. Its initial `weather.current_weather` capability calls only fixed Open-Meteo geocoding and forecast endpoints with a strict location schema and a short timeout. It is not arbitrary networking: neither capability names, operations, endpoints, nor callable code are derived from a user or model response.

An unregistered request is reported truthfully as unsupported or unavailable; a registered but policy-disabled capability is rejected. Neither is ever faked as completed. Test capabilities are isolated to tests.

## Non-goals

Step 5 does not implement autonomous agents, multi-step execution loops, replanning, arbitrary tools, browser or computer control, shell execution, filesystem access, arbitrary Python execution, arbitrary networking, credential access, scheduling, queues, or long-term memory.

## Configuration

- `BRAIN_ACTION_TIMEOUT` — internal action deadline in seconds; defaults to `30`, must be greater than zero and no more than `60`.
