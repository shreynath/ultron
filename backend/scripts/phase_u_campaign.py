"""Build the retained Phase U source-derived corpus from acquired repositories.

This script intentionally does not download sources.  Acquisition is a separate,
reviewable step; pass only repositories whose checked license evidence is stored
next to the acquired revision.  It produces immutable CorpusPipeline releases,
trains a tokenizer on their train partitions, and packs each partition for the
native Phase-U sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from app.foundation.data import (
    CorpusConfig,
    CorpusDocument,
    CorpusPipeline,
    DatasetRelease,
    DocumentCategory,
    Provenance,
    SourceType,
    TokenPacker,
)
from app.foundation.tokenizer import TokenizerConfig, UltronTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _files(
    root: Path, prefixes: tuple[str, ...], suffixes: tuple[str, ...]
) -> Iterable[Path]:
    for prefix in prefixes:
        directory = root / prefix
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in suffixes
                and path.stat().st_size >= 200
            ):
                yield path


def _documents(
    *,
    root: Path,
    source_id: str,
    source_url: str,
    license_name: str,
    license_evidence: str,
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
    domain: str,
    category: DocumentCategory,
) -> tuple[CorpusDocument, ...]:
    records: list[CorpusDocument] = []
    for path in _files(root, prefixes, suffixes):
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        records.append(
            CorpusDocument(
                content=content,
                provenance=Provenance(
                    source_type=SourceType.PUBLIC_DATASET,
                    source_identifier=f"{source_id}:{relative}",
                    source_url=f"{source_url}/blob/acquired/{relative}",
                    source_group=f"{source_id}:{relative}",
                    collection_timestamp=datetime.now(UTC),
                    license=license_name,
                    license_evidence=license_evidence,
                    acquisition_version="phase-u-acquired-v1",
                    parser_version="phase-u-source-tree-v1",
                    declared_category=category,
                ),
            )
        )
    if not records:
        raise ValueError(f"No usable documents selected for {source_id}/{domain}.")
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/phase-u-e2e"))
    args = parser.parse_args()
    root = args.root.resolve()
    cpython, numpy = root / "raw" / "cpython", root / "raw" / "numpy"
    if not (cpython / "LICENSE").is_file() or not (numpy / "LICENSE.txt").is_file():
        raise FileNotFoundError(
            "Acquire CPython and NumPy plus their license files first."
        )
    sources = {
        "natural-language": _documents(
            root=cpython,
            source_id="cpython-docs",
            source_url="https://github.com/python/cpython",
            license_name="PSF-2.0",
            license_evidence=f"LICENSE sha256:{_sha256(cpython / 'LICENSE')}",
            prefixes=("Doc",),
            suffixes=(".rst", ".txt"),
            domain="natural language",
            category=DocumentCategory.GENERAL,
        ),
        "programming": _documents(
            root=cpython,
            source_id="cpython-code",
            source_url="https://github.com/python/cpython",
            license_name="PSF-2.0",
            license_evidence=f"LICENSE sha256:{_sha256(cpython / 'LICENSE')}",
            prefixes=("Lib", "Parser", "Python"),
            suffixes=(".py", ".c", ".h"),
            domain="programming",
            category=DocumentCategory.CODE,
        ),
        "mathematics": _documents(
            root=numpy,
            source_id="numpy-docs",
            source_url="https://github.com/numpy/numpy",
            license_name="BSD-3-Clause",
            license_evidence=f"LICENSE.txt sha256:{_sha256(numpy / 'LICENSE.txt')}",
            prefixes=("doc", "numpy"),
            suffixes=(".rst", ".md"),
            domain="mathematics",
            category=DocumentCategory.MATHEMATICS,
        ),
        "systems": _documents(
            root=cpython,
            source_id="cpython-systems",
            source_url="https://github.com/python/cpython",
            license_name="PSF-2.0",
            license_evidence=f"LICENSE sha256:{_sha256(cpython / 'LICENSE')}",
            prefixes=("Modules", "Objects", "Include"),
            suffixes=(".c", ".h", ".py"),
            domain="computer systems",
            category=DocumentCategory.TECHNICAL,
        ),
        "science-engineering": _documents(
            root=numpy,
            source_id="numpy-source",
            source_url="https://github.com/numpy/numpy",
            license_name="BSD-3-Clause",
            license_evidence=f"LICENSE.txt sha256:{_sha256(numpy / 'LICENSE.txt')}",
            prefixes=("numpy",),
            suffixes=(".py", ".c", ".h"),
            domain="science engineering",
            category=DocumentCategory.ENGINEERING,
        ),
    }
    pipeline = CorpusPipeline(
        root,
        CorpusConfig(validation_percent=2, test_percent=2, minimum_characters=200),
    )
    releases = {}
    for domain, documents in sources.items():
        release_id = f"phase-u-{domain}-v1"
        manifest = root / "corpora" / "releases" / release_id / "manifest.json"
        releases[domain] = (
            DatasetRelease.model_validate_json(manifest.read_text(encoding="utf-8"))
            if manifest.exists()
            else pipeline.release(release_id, documents)
        )
    tokenizer_corpus = root / "phase-u-tokenizer-train.txt"
    tokenizer_corpus.write_text(
        "\n".join(
            json.loads(line)["content"].encode("ascii", "ignore").decode("ascii")
            for release in releases.values()
            for line in (
                root / "corpora" / "releases" / release.release_id / "train.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        ),
        encoding="utf-8",
    )
    tokenizer = (
        UltronTokenizer(root / "tokenizer")
        if (root / "tokenizer" / "ultron_tokenizer.model").exists()
        else UltronTokenizer.train(
            tokenizer_corpus,
            root / "tokenizer",
            TokenizerConfig(version="phase-u-source-derived-v1", vocab_size=384),
        )
    )
    packed = {
        domain: TokenPacker(root).pack(
            root / "corpora" / "releases" / release.release_id,
            tokenizer,
            context_length=64,
            max_sequences_per_shard=4096,
            packing_strategy="FIXED_TERMINAL_EOS_OVERLAP",
        )
        for domain, release in releases.items()
    }
    (root / "source-manifest.json").write_text(
        json.dumps(
            {
                "releases": {
                    key: value.model_dump(mode="json")
                    for key, value in releases.items()
                },
                "packed": {
                    key: value.model_dump(mode="json") for key, value in packed.items()
                },
                "tokenizer_sha256": tokenizer.sha256,
                "acquisition": {
                    "cpython_commit": subprocess.check_output(
                        ["git", "-C", str(cpython), "rev-parse", "HEAD"], text=True
                    ).strip(),
                    "numpy_commit": subprocess.check_output(
                        ["git", "-C", str(numpy), "rev-parse", "HEAD"], text=True
                    ).strip(),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
