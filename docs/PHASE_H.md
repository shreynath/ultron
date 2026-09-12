# Phase H: generation and generalization validation

Phase H adds a local-only, checkpoint-validated autoregressive decoder for ULTRON’s random-init foundation models. It supports greedy, temperature, top-k, and top-p selection, with bounded output/context/time/cancellation behavior and generation telemetry. The current architecture uses correct full-prefix recomputation; no unsafe KV-cache shortcut was added.

The phase also adds declared source categories, source-family leakage checks, immutable evaluation split copies, actual generator-backed executable evaluation, a macOS-sandboxed Python evaluator, and a real-generator cognitive contract adapter. These systems preserve the distinction between a model output and system validation/tool behavior.

See [PHASE_H_REPORT.md](/Users/shreyaannath/Programming/Ultron/docs/PHASE_H_REPORT.md) for the measured outcome: native generation works mechanically but is not useful at the measured data/training scale.
