"""A compact decoder-only Transformer with interleaved RoPE and NoPE layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tokens import EOS, PAD, VALUE_OFFSET


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    symbol_count: int = 10
    representation: str = "numbers"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    position_pattern: str = "alternating"
    rotary_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("representation must be 'alphabet' or 'numbers'")
        if self.symbol_count < 2:
            raise ValueError("symbol_count must be at least two")
        if self.vocab_size < VALUE_OFFSET + self.symbol_count:
            raise ValueError("vocab_size is too small for symbol_count")
        if self.d_model < 1 or self.n_layers < 1 or self.n_heads < 1:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_pattern not in {"alternating", "rotary", "none"}:
            raise ValueError(
                "position_pattern must be 'alternating', 'rotary', or 'none'"
            )
        if self.rotary_base <= 1.0:
            raise ValueError("rotary_base must be greater than one")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def uses_rotary(self, layer_index: int) -> bool:
        if self.position_pattern == "rotary":
            return True
        if self.position_pattern == "none":
            return False
        return layer_index % 2 == 0


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, tensor: Tensor) -> Tensor:
        """Rotate adjacent feature pairs in ``[batch, heads, time, dim]``."""

        positions = torch.arange(
            tensor.shape[-2],
            device=tensor.device,
            dtype=self.inverse_frequency.dtype,
        )
        angles = torch.outer(positions, self.inverse_frequency)
        cosine = angles.cos().to(dtype=tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=tensor.dtype)[None, None, :, :]
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, *, use_rotary: bool) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.use_rotary = use_rotary
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = (
            RotaryEmbedding(self.head_dim, config.rotary_base)
            if use_rotary
            else None
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, sequence_length, model_dim = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        if self.rotary is not None:
            query = self.rotary(query)
            key = self.rotary(key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            model_dim,
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.ffn_multiplier)
        self.input = nn.Linear(config.d_model, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.input(hidden).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, *, use_rotary: bool) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config, use_rotary=use_rotary)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden)))
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


class DecoderTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.number_projection = (
            nn.Linear(1, config.d_model, bias=False)
            if config.representation == "numbers"
            else None
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, use_rotary=config.uses_rotary(layer_index))
            for layer_index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def layer_position_modes(self) -> tuple[str, ...]:
        return tuple(
            "rotary" if block.attention.use_rotary else "none"
            for block in self.blocks
        )

    def embed(self, token_ids: Tensor) -> Tensor:
        hidden = self.token_embedding(token_ids)
        if self.number_projection is None:
            return hidden
        is_value = (token_ids >= VALUE_OFFSET) & (
            token_ids < VALUE_OFFSET + self.config.symbol_count
        )
        values = token_ids.to(dtype=hidden.dtype) - VALUE_OFFSET
        values = 2.0 * values / (self.config.symbol_count - 1) - 1.0
        values = torch.where(is_value, values, torch.zeros_like(values))
        return hidden + self.number_projection(values.unsqueeze(-1))

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.embed(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
    ) -> Tensor:
        """Greedily decode and return only tokens generated after the prompt."""

        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, time]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        full_sequence = prompt_ids
        finished = torch.zeros(
            prompt_ids.shape[0],
            dtype=torch.bool,
            device=prompt_ids.device,
        )
        for _ in range(max_new_tokens):
            next_token = self(full_sequence)[:, -1].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD),
                next_token,
            )
            full_sequence = torch.cat((full_sequence, next_token[:, None]), dim=1)
            finished = finished | next_token.eq(EOS)
            if bool(finished.all()):
                break
        return full_sequence[:, prompt_ids.shape[1] :]
