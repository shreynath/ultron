# Scaling

ScalingHarness generates valid 100K, 1M, 10M, 50M, 100M, and 300M candidate configurations but does not run them unless requested. Each short benchmark records the exact config, model estimate, token throughput, step time, MLX peak allocation, process peak RSS, swap before/after where macOS reports it, hardware, status, and any failure.

Measured on this machine on 2026-09-11: the 112,960-parameter, 2-layer, hidden-64, 4-head, context-32 configuration ran 3 measured batch-1 steps at 5,959.82 tokens/s and 0.00537 s/step. MLX peak was 6,684,600 bytes; process peak RSS 70,189,056 bytes; swap delta was 0 bytes. This is a short measurement, not an extrapolation to larger models or long training.

The report classifies practical feasibility with configurable throughput and swap-pressure thresholds. Missing platform measurements are recorded as null, never invented.
