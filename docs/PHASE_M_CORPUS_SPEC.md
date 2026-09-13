# Phase M corpus specification — v1

Phase M uses `ultron.phase-m.corpus-spec.v1` and the frozen tokenizer at
`backend/data/ultron/tokenizers/ultron-phase-i-v2`. The tokenizer count,
including one terminal EOS per document, is the authoritative size metric.

Only a source already present in ULTRON's acquisition registry may enter a
release. It must have a source ID, canonical URL, cache content hash,
retrieval time, source group, licence, and licence evidence. Permitted licences
are public-domain, PSF-2.0, MIT-or-Apache-2.0, CC-BY-4.0/SA, and the recorded
IETF/RFC terms. A missing cache, hash mismatch, unclear licence, or duplicate
cache hash is a recorded exclusion.

The builder applies NFC/whitespace normalization, privacy rejection, a 500
character minimum, exact raw/normalized SHA-256 deduplication, and conservative
5-token-shingle near-duplicate detection. It uses source-group-balanced 80/10/10
train/validation/sealed-test allocation; source groups, URLs, and fingerprints
must not cross partitions. Evaluation prompts and answers are never corpus
inputs.

New releases use the validated `VARIABLE_EOS_MASKED_V2` policy at context 512:
each row is single-document, has one terminal EOS, and carries an explicit
loss mask. The release builder emits immutable source, split, quality,
deduplication, and licence reports alongside checksummed corpus and packed
manifests.

The required exposure ladder is deliberately separate from corpus size: a valid
5M release is trained at 1M then 5M update tokens, then a valid 10M release at
10M update tokens. Architecture, tokenizer, objective, packing, optimizer,
context, seed, and the six-task frozen suite remain controls.
