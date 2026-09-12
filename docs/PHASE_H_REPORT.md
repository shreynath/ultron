# Phase H report

Phase H’s flagship experiment used the random-init 8.43M ULTRON model on the new 1,046,073-token source-isolated corpus. It trained for 1,000,192 tokens at 14,936 tokens/sec. Checkpoint `003500` was selected only from validation loss (2.555 at 896,000 tokens), then evaluated once on the sealed test: loss 2.888, perplexity 17.95 over 16,384 tokens. The held-out validation→test loss gap was 0.333. Release leakage checks passed, but the corpus still reports general-domain dominance and instruction starvation.

Native generation is demonstrated mechanically. It uses ULTRON’s own checkpoint and tokenizer, not Qwen, Ollama, cloud inference, or a fallback. The six greedy smoke prompts produced repeated fragments such as `The the the the the the the the`; repetition rates were 0.55–0.81 by repeated bigrams, and every prompt exhausted the max-token limit without EOS. The 15-task model-only executable suite scored 0/15; generated Python failed inside the macOS sandbox with a syntax error. These outputs are preserved unedited in the generation and capability reports.

A 25M-class candidate was also trained for 100,096 tokens on the same corpus. Its validation loss was 3.123 and its generation was similarly repetitive. That shorter-budget experiment does not establish a size comparison, but it does show that extra parameters without comparable training tokens did not help.

The real foundation-model cognitive adapter attempted assessment, plan, and tool-request stages. It generated malformed text and no allowlisted tool name; the loop correctly ended `TOOL_REQUEST_REJECTED`. No deterministic plan or answer was inserted.

The Phase H 8M candidate is rejected for promotion. The primary Phase I bottleneck is **training compute and data scale**: 1M tokens can improve teacher-forced loss without producing useful free generation. The next experiment should use a much larger, more balanced, independently held-out corpus and a substantially longer budget before treating capacity as the dominant constraint.

| Capability | Model-only | System-assisted | Generalization | Status |
| --- | --- | --- | --- | --- |
| Language | 0/1 fail | Not claimed | Teacher-forced only | INSUFFICIENT_EVIDENCE |
| Mathematics | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Coding | 0/1 runtime error | Sandbox only | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Computer Science | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Science | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Engineering | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Instruction | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Reasoning | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Research | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Tool Use | 0/1 fail | Rejected invalid request | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Planning | 0/1 fail | Validation only | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Memory | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
| Truthfulness | 0/1 fail | Not run | Not demonstrated | INSUFFICIENT_EVIDENCE |
