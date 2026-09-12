# Phase L Research Log

- Corpus gate: Phase K balanced pilot v4 is provenance/licensing/leakage-valid
  but contains 246,370 tokenizer tokens, below the 10,000,000-token threshold.
  It cannot authorize serious scaling conclusions.
- Sustained-run instrumentation records optimizer-step windows, loss range,
  MLX peak, RSS peak, and swap trajectory.  Its random-input mode is hardware
  evidence only.
- 1B sustained hardware result: 100 steps completed without a reported runtime
  failure.  Swap did not grow monotonically across the ten windows, so the
  result is `SUSTAINED_EXECUTION_WITH_SWAP`; checkpoint behavior was not
  attempted by this harness and remains unmeasured.
- Cache expansion candidate v1 (strict PII rejection) and v2 (explicit PII
  redaction) are immutable rejected candidates.  V2 is integrity-valid and
  balanced at 497,507 tokens, but remains far below the 10M corpus gate.
- Checkpoint portability currently verifies the complete local MLX checkpoint
  format, architecture hash, and immutable weights hash.  Cross-backend reload
  and identical generation are not yet measured because a second backend does
  not yet exist.
