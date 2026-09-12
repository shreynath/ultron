# ULTRON model card

ULTRON is a from-scratch experimental foundation model. It is not a pretrained, distilled, cloud-hosted, or third-party model.

The Phase H flagship candidate has 8,426,368 parameters, six layers, hidden size 384, six heads, 128-token context, float32 training, and a 320-token ULTRON-owned SentencePiece BPE tokenizer. It was randomly initialized and trained on `ultron-corpus-0.4.0-phase-h-diverse-v1` for 1,000,192 packed tokens on an Apple M4 Pro with 24 GB unified memory.

Measured: validation-selected checkpoint loss 2.555 on the validation probe and 2.888 on a sealed unseen-source test probe. Native decoding works and is bounded. Not demonstrated: coherent language, simple instruction following, mathematics, executable code, tool use, structured model contract outputs, or useful cognitive completion. All Phase H model-only capability tasks failed; the candidate is not promoted.

Prohibited claims: that ULTRON is generally intelligent, conscious, a substitute for a third-party model, or capable beyond its recorded evaluations.
