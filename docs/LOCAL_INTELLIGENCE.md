# ULTRON Local Intelligence Architecture

## Status

The backend now defaults to local-only model orchestration. It has no active
OpenAI, Anthropic, Gemini, OpenRouter, Hugging Face Inference, or Ollama
inference path. A missing model produces `LOCAL MODEL UNAVAILABLE`; it never
routes to a hosted model.

The frozen Executive Cognitive Layer remains responsible for assessment,
planning, controlled tools, verification, and degradation. It calls the local
structured-generation client through the same typed boundary as before.

## Phase 0 audit

Before this migration, the direct response path, Executive, and Planner used
`app.services.ollama_client`; the default configuration targeted
`https://ollama.com` and `gpt-oss:120b-cloud`. That is now retired. The module
name remains only as a compatibility import for JSON parsing and its functions
delegate to the local orchestrator.

Existing durable SQLite learning (`app.brain.learning`) records verified system
experience and web provenance. The new local memory store is complementary: it
holds explicitly saved episodic, semantic, procedural, and user memory without
claiming those are training examples or model changes.

Network-capable code is confined to explicit capabilities: bounded research,
weather, registered public corpus resolution, media ingestion, and the local
Voicebox HTTP loopback service. Those are tools, not inference. The local model
runtime does not import an HTTP client.

## Runtime and storage

`ULTRON_HOME` defaults to `data/ultron` and is created with this layout:

```
models/{base,adapters,promoted,archived}
memory/{conversations,semantic,episodic,procedural,user}
learning/{raw,curated,feedback,evaluations,training,checkpoints}
registry/ benchmarks/ cache/ logs/
```

`LocalModelRuntime` is the backend contract. The supplied `MlxLocalRuntime`
loads only a path under `ULTRON_HOME/models` and uses MLX-LM lazily. Install the
optional `local` dependency on Apple Silicon, then place verified weights there
and register them. No startup download is performed.

`LocalModelRegistry` hashes artifacts at registration, appends an audit event,
maps capability classes (`fast`, `general`, `reasoning`, `coding`) to explicitly
promoted models, and preserves older artifacts. Promotion is explicit;
`rollback(capability, model_id)` promotes an already registered compatible
artifact without deleting anything.

## Memory, learning, and training semantics

- Saving memory creates a persistent local memory artifact, not a weight update.
- Feedback is raw or review-required. Only explicit correction / “learn this”
  feedback with content is eligible for human curation.
- Dataset creation writes a versioned local JSONL artifact only from curator-
  approved events. It refuses common credentials and bearer tokens.
- `LocalTrainingRunner` invokes the local MLX-LM LoRA CLI only when explicitly
  started. It records the command and result, and it cannot register a
  candidate unless the training command completed and an adapter directory
  exists.

Evaluation and promotion controls should be wired to a task-specific benchmark
suite before any candidate is promoted in production. The registry deliberately
does not fabricate benchmark scores.

## Diagnostics and setup

`GET /api/local-intelligence/status` reports local runtime availability, active
models, paths, selected network mode, `cloud_llm: DISABLED`, and
`ollama: NOT_REQUIRED`. Memory and feedback controls are exposed below the same
prefix.

On this Apple Silicon machine, the selected native backend is MLX-LM. The
first installed baseline is `mlx-community/Qwen3-8B-4bit` (Apache-2.0), stored
under `data/ultron/models/base/qwen3-8b-4bit` and registered as
`qwen3-8b-4bit-v1`. Its measured M4 Pro baseline (MLX 0.32.2 / MLX-LM 0.31.3,
512 prompt tokens / 64 generated tokens) is 383.805 prompt tokens/s, 42.685
generation tokens/s, and 5.217 GB peak MLX memory. The immutable benchmark
record lives in `ULTRON_HOME/benchmarks`; it is a starting point for comparing
the next candidate, not a model-quality score.
