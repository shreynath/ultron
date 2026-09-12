"""ULTRON-owned SentencePiece BPE tokenizer training and serialization."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "0.0.1"
    vocab_size: int = Field(default=256, ge=64)
    model_type: str = "bpe"
    max_sentence_length: int = Field(default=1_048_576, ge=256)
    special_tokens: tuple[str, ...] = ("<ULTRON_SYSTEM>", "<ULTRON_USER>", "<ULTRON_ASSISTANT>", "<ULTRON_TOOL>", "<ULTRON_DOCUMENT>")


class UltronTokenizer:
    def __init__(self, model_path: Path) -> None:
        import sentencepiece as spm

        # CLI callers naturally provide a tokenizer release directory; the
        # persisted artifacts and older callers provide the model directly.
        # Resolve both forms deterministically.
        if model_path.is_dir():
            model_path = model_path / "ultron_tokenizer.model"
        if not model_path.is_file():
            raise FileNotFoundError(f"ULTRON tokenizer model not found: {model_path}")
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    @property
    def version(self) -> str:
        config = self.model_path.parent / "tokenizer_config.json"
        if not config.exists():
            return "UNKNOWN"
        return TokenizerConfig.model_validate_json(config.read_text(encoding="utf-8")).version

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    @staticmethod
    def train(corpus: Path, output_dir: Path, config: TokenizerConfig) -> UltronTokenizer:
        import sentencepiece as spm

        output_dir.mkdir(parents=True, exist_ok=False)
        prefix = output_dir / "ultron_tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(prefix), vocab_size=config.vocab_size,
            model_type=config.model_type, character_coverage=1.0, hard_vocab_limit=False,
            user_defined_symbols=list(config.special_tokens), byte_fallback=True,
            max_sentence_length=config.max_sentence_length,
        )
        (output_dir / "tokenizer_config.json").write_text(config.model_dump_json(indent=2) + "\n")
        (output_dir / "tokenizer.sha256").write_text(hashlib.sha256(prefix.with_suffix(".model").read_bytes()).hexdigest() + "\n")
        return UltronTokenizer(prefix.with_suffix(".model"))
