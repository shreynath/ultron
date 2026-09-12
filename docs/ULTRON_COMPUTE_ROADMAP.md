# ULTRON compute roadmap

ULTRON checkpoint semantics are hardware-independent: architecture configuration, tensor shapes/dtype, tokenizer hash, corpus manifest hash, optimizer/scheduler state, RNG metadata, and lineage travel with each checkpoint. Current execution is local MLX only.

Planned execution targets are represented in configuration but are **UNIMPLEMENTED** until tested:

1. Apple M4 Pro — current measured platform.
2. Larger local Apple Silicon — same local checkpoint contract.
3. Local high-memory GPU — future backend, not silently substituted.
4. Multi-GPU local workstation — future data/tensor/pipeline parallel implementation.
5. University HPC — explicit operator-selected target only; no data/model is sent automatically.
6. Larger dedicated training infrastructure — future sharded checkpoint and distributed execution.

The future configuration supports node count, GPU count, distributed mode, data/tensor/pipeline parallel factors, sharding, and checkpoint sharding. All non-local backends remain `UNIMPLEMENTED`; no distributed capability is claimed.
