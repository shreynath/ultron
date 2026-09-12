# Checkpoints

Each new checkpoint is an immutable directory with model.safetensors, optimizer.safetensors, scheduler.json, trainer_state.json, and manifest.json. Publication writes to a sibling temporary directory, fsyncs artifacts, validates SHA-256 hashes and required files, then atomically renames it.

The manifest schema is ultron.checkpoint.v1 and records architecture/tokenizer/dataset hashes, run and parent IDs, state position, metrics, runtime/hardware, and lineage. CheckpointStore.inspect rejects missing or modified artifacts. Legacy standalone safetensors files are readable as weight-only artifacts but are explicitly not exact-resumable.
