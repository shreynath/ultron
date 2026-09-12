# Data pipeline

CorpusPipeline accepts only explicitly supplied CorpusDocument objects carrying provenance. It has no reader for user memory, private conversations, credentials, or private tool results. Curated experience is a separately labeled source type and must cross an explicit release boundary.

The pipeline normalizes NFC/whitespace, identifies a conservative language label, applies auditable PII/secret checks, quality filters, deterministic exact and normalized-text deduplication, classification, and deterministic hash-based train/validation splitting. Rejections are retained with reasons. Releases are immutable and include provenance, filtering, category, source, language, and duplicate statistics.

TokenPacker makes a separate tokenizer-keyed artifact, preserves document boundaries, drops incomplete tails, and writes checksummed ordered JSONL shards.
