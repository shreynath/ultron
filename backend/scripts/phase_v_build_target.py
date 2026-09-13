"""Create an explicitly synthetic-but-structured arithmetic curriculum release."""
from __future__ import annotations
import hashlib
from pathlib import Path
from app.foundation.data import CorpusConfig, CorpusDocument, CorpusPipeline, DocumentCategory, Provenance, SourceType, TokenPacker
from app.foundation.tokenizer import UltronTokenizer

root = Path("data/phase-v-e2e").resolve()
tokenizer = UltronTokenizer(Path("data/phase-u-e2e/tokenizer"))
docs = []
for left in range(1000):
    for right in range(100):
        total = left + right
        text = f"Arithmetic lesson. ADD {left} {right} = {total}. The sum of {left} and {right} is {total}. Check: {total} - {right} = {left}."
        docs.append(CorpusDocument(content=text, provenance=Provenance(source_type=SourceType.LOCAL_AUTHORED, source_identifier=f"phase-v-addition-{left}-{right}", source_group=f"addition-{left}-{right}", license="CC0-1.0", license_evidence="ULTRON Phase V locally authored deterministic arithmetic curriculum", declared_category=DocumentCategory.MATHEMATICS)))
release_id = "phase-v-addition-v1"
manifest = root / "corpora" / "releases" / release_id / "manifest.json"
release = __import__("app.foundation.data", fromlist=["DatasetRelease"]).DatasetRelease.model_validate_json(manifest.read_text()) if manifest.exists() else CorpusPipeline(root, CorpusConfig(validation_percent=2, test_percent=2, minimum_characters=30)).release(release_id, docs)
packed = TokenPacker(root).pack(root / "corpora" / "releases" / release.release_id, tokenizer, context_length=64, max_sequences_per_shard=4096)
(root / "target-manifest.json").write_text('{"release_id":"' + release.release_id + '","release_sha256":"' + (release.manifest_sha256 or '') + '","tokenizer_sha256":"' + tokenizer.sha256 + '","unique_target_examples":100000,"synthetic_tokens":' + str(packed.token_count) + '}\n')
