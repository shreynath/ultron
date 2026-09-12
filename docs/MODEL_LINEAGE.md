# Model lineage

The registry distinguishes ULTRON_FROM_SCRATCH, THIRD_PARTY_REFERENCE_MODEL, and DERIVED_FROM_THIRD_PARTY (with compatibility aliases for older local registry values).

Promotion of an ULTRON from-scratch record rejects missing tokenizer provenance, dataset-manifest provenance, or a lineage other than initialization=RANDOM, pretrained_weights=NONE, and parent_model=NONE. Completing a training run does not promote it. Evaluation and regression evidence remain required before a promotion decision.
