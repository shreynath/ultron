# Phase J report

## Decision

**Completed, not promotable.** The exact 15.6M-parameter ULTRON model was trained from random initialization for 10,000,384 tokens on a clean, source-isolated Phase J release. It improves sealed held-out loss versus its own fresh random initialization, but it has no demonstrated task capability and its data composition is too narrow for a general-model claim.

## Quantitative result

| Metric | Result |
| --- | ---: |
| corpus raw characters | 6,664,882 |
| corpus tokenizer tokens | 1,914,488 |
| training-partition tokens | 1,363,554 |
| train / validation / test documents | 152 / 12 / 21 |
| source families | 84 |
| provenance coverage | 100% |
| license-evidence coverage | 66.49% |
| training budget / actual | 10,000,384 / 10,000,384 tokens |
| training duration | 516.17 s |
| mean throughput | 19,374.3 tokens/s |
| peak MLX memory | 789,128,976 bytes |
| observed swap growth | 0 bytes |
| final development validation loss / PPL | 5.6097 / 273.1 |
| sealed random loss / PPL | 9.1589 / 9498.2 |
| sealed 1M / 5M / 10M loss | 5.6356 / 5.3870 / 5.3759 |
| raw final repeated-bigram rate | 49.47% |
| raw final EOS rate | 0% |
| task-owned capability categories passing | 0 of 8 |

The release passes source-group, URL, and near-duplicate cross-split checks. The discarded v1 release did not: two duplicate Gutenberg URLs crossed splits; it is preserved but not used for training.

## Interpretation

Loss improvement is evidence that the model learned predictive regularities in this corpus. It is not evidence of generalization sufficient for deployment: raw prompts still collapse into repetitive literary fragments, neither arithmetic nor exact instruction tasks pass, and all code/science/engineering/reasoning task scores are zero. The corpus is also 80.54% GENERAL by document count and 53.51% Gutenberg by domain, against a much more balanced target.

No candidate was promoted and no cognitive-execution or continual-learning integration was enabled.
