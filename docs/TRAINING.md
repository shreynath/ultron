# Training

FoundationTrainer runs a local MLX AdamW loop with a linear-warmup/constant learning-rate schedule. steps is an absolute target: a resumed run continues from its saved global step. The trainer records train/validation loss, perplexity, throughput, gradient norm, token count, and run status.

The only mutable values allowed on resume are the final target step and checkpoint/validation cadence. Model, data, optimizer, seed, and scheduler settings must match. The direct-text API exists for development; production candidates should carry a corpus release and manifest hash.

MLX exposes a seed but no readable random-generator state in the installed runtime. Python and NumPy state are restored. The training path makes no MLX random draws after deterministic model construction, which is why the deterministic regression can compare resumed and continuous runs.
