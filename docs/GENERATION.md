# ULTRON-native generation

`app.foundation.generation.UltronGenerator` accepts only complete ULTRON-from-scratch checkpoints whose tokenizer hash and architecture hash match the supplied owned tokenizer and model configuration. It rejects pretrained or third-party lineage.

Use `python scripts/foundation.py generate --checkpoint … --model-config … --tokenizer … --prompt …`. Greedy evaluation is `temperature=0`; sampling requires an explicit positive temperature and records seed/top-k/top-p. Generation stops at EOS, a configured stop token, the context limit, cancellation, timeout, or an error.

The current decoder recomputes every active prefix because the MLX attention block lacks a tested incremental cache API. Phase H measured roughly 536–732 generated tokens/sec on short greedy 8M smoke prompts, but output quality was highly repetitive. Throughput is demonstrated; useful semantic generation is not.
