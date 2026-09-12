# Generalization methodology

`ultron-corpus-0.4.0-phase-h-diverse-v1` holds complete source groups apart via `source_group_balanced`. Its leakage report checks source-group, URL, and retained near-duplicate fingerprint overlap across train, validation, and test; the recorded release passes those checks.

The selected 8M checkpoint was chosen at 896,000 training tokens by validation loss only. The sealed test was run afterwards. Teacher-forced probe losses were 2.401 for train-source, 2.555 for validation-source, and 2.888 for unseen test-source windows, giving a validation-to-test loss gap of 0.333. This is a measurable generalization gap, not evidence of task-level generalization; free generation remained degenerate.
