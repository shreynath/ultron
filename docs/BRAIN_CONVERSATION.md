# Conversation Architecture (Step 6.5)

Step 6.5 is an integration and UX hardening step, not a new cognitive stage. It makes Conversation the normal Ultron surface while preserving the Step 1–6 ownership boundaries.

## Conversation flow

`Conversation UI -> /api/chat -> Executive -> CognitiveRuntime -> CognitiveContext -> optional Planner -> optional Single-Action Capability Bridge -> response -> optional Voice delivery`

Text and microphone input use the same authoritative conversation route. Microphone audio is transcribed by the existing Voicebox/Whisper path, then its transcript enters Executive and Runtime exactly like typed text. The frontend sends only the recent bounded conversation turns; the backend applies the existing Brain history bound before assessment and runtime execution.

Executive normalization treats explicit live-information requests as capability-required even when a structured-model response is incomplete or incorrectly marks them direct. This is semantic triage at the Executive boundary, not a chat-route capability shortcut: the bridge still selects only a registered capability and may decline to act when none supports the request.

## Forensic integration safeguard

The conversation endpoint calls `CognitiveRuntime.handle_conversation(...)`, rather than a standalone text generator. That entry point routes first, then gives only capability-required requests to the single-action bridge. Executive fallback assessments are normalized with the same live-information rule, so an unavailable or malformed Executive model response cannot silently send an explicit weather request to the direct-response model path.

The normal UI receives a final answer plus safe presentation metadata (`task_type`, route, and status). It never exposes planning, actions, observations, prompts, internal state labels, or hidden reasoning. The separate Brain page remains the engineering telemetry surface.

## Voice delivery

Voice is a delivery modality, not a cognitive stage. When automatic voice is enabled and a cloned profile is selected, the Conversation UI asks `/api/chat` or `/api/chat/voice` to run the existing delivery path with the final answer. The returned audio is played only for its matching turn, preventing stale responses from interrupting newer ones. A speech or playback failure leaves the text answer intact and appears only as a subtle audio-unavailable indicator.

Voice profile, engine, and delivery controls live behind the compact Voice settings disclosure. The Advanced page retains the standalone Dictation and Text-to-Speech tools for users who need them.

## Capability-backed conversation

For a capability-required request, the optional bridge can select one explicitly registered application capability. Selection is deterministic: a capability supplies static metadata (name, supported task types, and operations) and application code may construct at most one validated action. User or model text never names a callable, authorizes an operation, or supplies a URL.

The selected `ActionRequest` goes through the Step 5 executor once, its `ActionResult` goes through the Step 6 observation interpreter once, and the capability's deterministic formatter creates the conversational text from the structured result. A formatter is intentionally available as a fallback when a conversational model is unavailable; no model is asked to invent a result.

The first production capability is `weather.current_weather`. It uses fixed Open-Meteo geocoding and forecast endpoints, accepts a strict location field, and returns normalized current conditions (location, temperature, apparent temperature, precipitation, condition, wind, time, and timezone). Provider, validation, or timeout failures produce a concise truthful failure response and never fabricate weather data.

If no registered capability recognizes the request, Ultron keeps the existing truthful limitation response. Adding another capability requires a strict parameter model, descriptor, deterministic action builder, executor implementation, formatter, and tests. Register it in the bridge's application-owned registry; do not add conversational `if` branches or dynamic tool lookup.

This is still a single-action boundary. Step 6.5 does not verify results, advance plans, retry work, replan, persist memory, create a multi-action coordinator, or create autonomous observation/decision loops. Those remain future work.
