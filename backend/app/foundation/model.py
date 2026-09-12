"""A compact, ULTRON-owned decoder-only Transformer for from-scratch training."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UltronModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = "ultron-foundation"
    version: str = "0.0.1"
    vocab_size: int = Field(ge=64)
    hidden_size: int = Field(default=64, ge=16)
    num_layers: int = Field(default=2, ge=1)
    num_heads: int = Field(default=4, ge=1)
    num_kv_heads: int | None = Field(default=None, ge=1)
    intermediate_size: int = Field(default=256, ge=16)
    max_sequence_length: int = Field(default=64, ge=8)
    normalization: str = "rmsnorm"
    activation: str = "gelu"
    positional_encoding: str = "learned"
    dtype: str = "float32"
    random_seed: int = 17
    initialization: str = "RANDOM"
    pretrained_weights: str = "NONE"
    parent_model: str = "NONE"

    @model_validator(mode="after")
    def validate_architecture(self) -> UltronModelConfig:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if self.num_kv_heads not in (None, self.num_heads):
            raise ValueError("The current ULTRON decoder supports only equal query and KV heads.")
        if self.initialization != "RANDOM" or self.pretrained_weights != "NONE" or self.parent_model != "NONE":
            raise ValueError("ULTRON foundation configurations must declare random initialization and no parent weights.")
        return self

    def parameter_estimate(self) -> int:
        """Return the exact trainable-count formula for the current decoder.

        This is deliberately kept in lockstep with :func:`build_model`.  The
        historical Phase J helper treated position embeddings as though they
        had vocabulary rows and omitted the untied output projection; that
        happened to produce a plausible label but not the model's real count.
        Phase K records instantiated counts as the authoritative value and
        uses this formula only for safe, allocation-free preflight planning.
        """
        embeddings = self.vocab_size * self.hidden_size + self.max_sequence_length * self.hidden_size
        attention = self.num_layers * 4 * self.hidden_size * self.hidden_size
        mlp = self.num_layers * 2 * self.hidden_size * self.intermediate_size
        norms = self.num_layers * 2 * self.hidden_size + self.hidden_size
        output = self.vocab_size * self.hidden_size
        return embeddings + attention + mlp + norms + output


def build_model(config: UltronModelConfig):
    """Construct fresh MLX parameters; this function never loads foreign weights."""
    import mlx.core as mx
    from mlx import nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_norm = nn.RMSNorm(config.hidden_size)
            self.attn = nn.MultiHeadAttention(config.hidden_size, config.num_heads, bias=False)
            self.mlp_norm = nn.RMSNorm(config.hidden_size)
            self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        def __call__(self, x):
            y = self.attn_norm(x)
            mask = nn.MultiHeadAttention.create_additive_causal_mask(y.shape[1])
            x = x + self.attn(y, y, y, mask=mask)
            return x + self.down(nn.gelu(self.up(self.mlp_norm(x))))

    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mx.random.seed(config.random_seed)
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
            self.layers = [Block() for _ in range(config.num_layers)]
            self.final_norm = nn.RMSNorm(config.hidden_size)
            self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def __call__(self, tokens):
            positions = mx.arange(tokens.shape[1])
            x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
            for layer in self.layers:
                x = layer(x)
            return self.output(self.final_norm(x))

    return Decoder()


UltronDecoder = build_model
