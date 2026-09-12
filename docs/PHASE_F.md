# Phase F

Phase F addresses two measured Phase E blockers: direct public-web acquisition and natural packed sequences. The acquisition engine accepts only explicit curated URLs, applies the existing safe public fetch policy, retries within a bounded budget, stores resumable job state, and caches content by SHA-256. Retrieved text is untrusted external data until it crosses the corpus provenance, privacy, quality, and release boundary.

The first web-curated development release is ultron-corpus-0.2.0-curated-web-v4-source-split. It has 63 documents and 76,583 tokenizer tokens, including three public sources with recorded license identifiers. The development split is intentionally 50 percent because only three long external source documents are available; it is not a production train/validation ratio.

The packed-shard run trained the 8.43M random-init model for 10,240 tokens. Held-out loss improved from 5.634 at 1,280 tokens to 4.985 at 8,960 tokens, then degraded to 5.261 by 10,240 tokens. The run is preserved but not promoted.
