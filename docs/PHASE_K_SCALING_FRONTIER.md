# Phase K scaling frontier

No scaling law is claimed. There is one Phase K parameter-training pilot and no fair same-corpus/same-token multi-parameter full-run comparison.

| State | Actual parameters | Tokens | validation loss | sealed loss | native repetition | capability | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Phase J preserved baseline | 14,750,592 | 10,000,384 | 5.610 | 5.376 | 49.47% | 0.0 | not promoted |
| K1 50M pilot | 50,553,216 | 100,352 | 7.402 | 8.194 | 97.87% | 0.0 | completed, not promotable |

The two rows differ in corpus and token budget, so they cannot isolate parameter scaling. The K1 result rules out the claim that “50M parameters alone fixes degeneration”: with only 100k pilot tokens, greedy outputs collapse to repeated `the` and never emit EOS. The next fair experiment requires the same accepted corpus, tokenizer, context, optimizer, and token budget across at least two scales.
