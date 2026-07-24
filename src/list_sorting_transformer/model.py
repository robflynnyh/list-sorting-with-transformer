"""A compact decoder-only Transformer with interleaved RoPE and NoPE layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tokens import EOS, PAD, VALUE_OFFSET


KeyValueCache = tuple[Tensor, Tensor]


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
    rotate_values_with_rope: bool = False

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

    def forward(self, tensor: Tensor, *, position_offset: int = 0) -> Tensor:
        """Rotate adjacent feature pairs in ``[batch, heads, time, dim]``."""

        positions = torch.arange(
            position_offset,
            position_offset + tensor.shape[-2],
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
        self.rotate_values_with_rope = config.rotate_values_with_rope
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = (
            RotaryEmbedding(self.head_dim, config.rotary_base)
            if use_rotary
            else None
        )

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, model_dim = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.n_heads,
            model_dim // self.n_heads,
        ).transpose(1, 2)

    def query_key_logits(
        self,
        hidden: Tensor,
        *,
        query_index: int,
    ) -> Tensor:
        """Return causal pre-softmax scores for one query in every head."""

        sequence_length = hidden.shape[1]
        if not -sequence_length <= query_index < sequence_length:
            raise IndexError("query_index is outside the sequence")
        resolved_query_index = query_index % sequence_length
        query, key, _ = self.qkv(hidden).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        if self.rotary is not None:
            query = self.rotary(query)
            key = self.rotary(key)
        logits = (
            query[:, :, resolved_query_index].unsqueeze(-2)
            @ key.transpose(-2, -1)
            / self.head_dim**0.5
        ).squeeze(-2)
        if resolved_query_index + 1 < sequence_length:
            logits = logits.masked_fill(
                torch.arange(sequence_length, device=hidden.device)
                > resolved_query_index,
                float("-inf"),
            )
        return logits

    def forward_with_cache(
        self,
        hidden: Tensor,
        *,
        cache: KeyValueCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, KeyValueCache]:
        batch_size, sequence_length, model_dim = hidden.shape
        if cache is not None and attention_mask is not None:
            raise ValueError("custom attention masks are not supported with a cache")
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        position_offset = 0 if cache is None else cache[0].shape[-2]
        if self.rotary is not None:
            query = self.rotary(query, position_offset=position_offset)
            key = self.rotary(key, position_offset=position_offset)
            if self.rotate_values_with_rope:
                value = self.rotary(value, position_offset=position_offset)
        if cache is not None:
            if sequence_length != 1:
                raise ValueError(
                    "cached attention accepts one new token at a time"
                )
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        combined_mask = None
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean")
            if attention_mask.shape == (sequence_length, sequence_length):
                combined_mask = attention_mask
            elif attention_mask.shape == (
                batch_size,
                sequence_length,
                sequence_length,
            ):
                combined_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError(
                    "attention_mask must have shape [time, time] or "
                    "[batch, time, time]"
                )
            causal_mask = torch.ones(
                sequence_length,
                sequence_length,
                device=hidden.device,
                dtype=torch.bool,
            ).tril()
            combined_mask = combined_mask.to(device=hidden.device) & causal_mask
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=cache is None and combined_mask is None,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            model_dim,
        )
        return self.output(attended), (key, value)

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attended, _ = self.forward_with_cache(
            hidden,
            attention_mask=attention_mask,
        )
        return attended


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

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = hidden + self.dropout(
            self.attention(
                self.attention_norm(hidden),
                attention_mask=attention_mask,
            )
        )
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden

    def forward_with_cache(
        self,
        hidden: Tensor,
        *,
        cache: KeyValueCache | None = None,
    ) -> tuple[Tensor, KeyValueCache]:
        attended, new_cache = self.attention.forward_with_cache(
            self.attention_norm(hidden),
            cache=cache,
        )
        hidden = hidden + self.dropout(attended)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden, new_cache


class DecoderTransformer(nn.Module):
    architecture = "transformer"

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
            (
                "rotary+value"
                if block.attention.use_rotary
                and block.attention.rotate_values_with_rope
                else "rotary"
                if block.attention.use_rotary
                else "none"
            )
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

    def hidden_states(
        self,
        token_ids: Tensor,
        *,
        extra_input_embeddings: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.embed(token_ids)
        if extra_input_embeddings is not None:
            if extra_input_embeddings.shape == hidden.shape[-2:]:
                extra_input_embeddings = extra_input_embeddings.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            elif extra_input_embeddings.shape == hidden.shape:
                extra_input_embeddings = extra_input_embeddings.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            else:
                raise ValueError(
                    "extra_input_embeddings must have shape [time, d_model] "
                    "or [batch, time, d_model]"
                )
            hidden = hidden + extra_input_embeddings
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.final_norm(hidden)

    def forward(
        self,
        token_ids: Tensor,
        *,
        extra_input_embeddings: Tensor | None = None,
    ) -> Tensor:
        hidden = self.hidden_states(
            token_ids,
            extra_input_embeddings=extra_input_embeddings,
        )
        return F.linear(hidden, self.token_embedding.weight)

    def forward_with_cache(
        self,
        token_ids: Tensor,
        *,
        caches: tuple[KeyValueCache, ...] | None = None,
        extra_input_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, tuple[KeyValueCache, ...]]:
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("one key/value cache is required per layer")
        hidden = self.embed(token_ids)
        if extra_input_embeddings is not None:
            if extra_input_embeddings.shape == hidden.shape[-2:]:
                extra_input_embeddings = extra_input_embeddings.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            elif extra_input_embeddings.shape == hidden.shape:
                extra_input_embeddings = extra_input_embeddings.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            else:
                raise ValueError(
                    "extra_input_embeddings must have shape [time, d_model] "
                    "or [batch, time, d_model]"
                )
            hidden = hidden + extra_input_embeddings
        new_caches = []
        for layer_index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[layer_index]
            hidden, new_cache = block.forward_with_cache(hidden, cache=cache)
            new_caches.append(new_cache)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        return logits, tuple(new_caches)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        stop_token: int = EOS,
        extra_input_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Greedily decode and return only tokens generated after the prompt."""

        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, time]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        prompt_extra = None
        if extra_input_embeddings is not None:
            prompt_extra = extra_input_embeddings[:, : prompt_ids.shape[1]]
        next_logits, caches = self.forward_with_cache(
            prompt_ids,
            extra_input_embeddings=prompt_extra,
        )
        generated = []
        finished = torch.zeros(
            prompt_ids.shape[0],
            dtype=torch.bool,
            device=prompt_ids.device,
        )
        for _ in range(max_new_tokens):
            next_token = next_logits[:, -1].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, PAD),
                next_token,
            )
            generated.append(next_token)
            finished = finished | next_token.eq(stop_token)
            if bool(finished.all()):
                break
            next_extra = None
            if extra_input_embeddings is not None:
                position_index = prompt_ids.shape[1] + len(generated) - 1
                next_extra = extra_input_embeddings[
                    :,
                    position_index : position_index + 1,
                ]
            next_logits, caches = self.forward_with_cache(
                next_token[:, None],
                caches=caches,
                extra_input_embeddings=next_extra,
            )
        return torch.stack(generated, dim=1)


class SplitInputDecoderTransformer(nn.Module):
    """Decoder body with separate content and position input subspaces."""

    architecture = "split_input_transformer"

    def __init__(self, config: ModelConfig, *, content_dim: int) -> None:
        super().__init__()
        if not 1 <= content_dim < config.d_model:
            raise ValueError("content_dim must be inside the model dimension")
        self.config = config
        self.content_dim = content_dim
        self.position_dim = config.d_model - content_dim
        self.token_embedding = nn.Embedding(config.vocab_size, content_dim)
        self.number_projection = (
            nn.Linear(1, content_dim, bias=False)
            if config.representation == "numbers"
            else None
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config, use_rotary=config.uses_rotary(layer_index))
            for layer_index in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(DecoderTransformer._initialize)

    @property
    def layer_position_modes(self) -> tuple[str, ...]:
        return tuple(
            (
                "rotary+value"
                if block.attention.use_rotary
                and block.attention.rotate_values_with_rope
                else "rotary"
                if block.attention.use_rotary
                else "none"
            )
            for block in self.blocks
        )

    def embed(self, token_ids: Tensor) -> Tensor:
        content = self.token_embedding(token_ids)
        if self.number_projection is None:
            return content
        is_value = (token_ids >= VALUE_OFFSET) & (
            token_ids < VALUE_OFFSET + self.config.symbol_count
        )
        values = token_ids.to(dtype=content.dtype) - VALUE_OFFSET
        values = 2.0 * values / (self.config.symbol_count - 1) - 1.0
        values = torch.where(is_value, values, torch.zeros_like(values))
        return content + self.number_projection(values.unsqueeze(-1))

    def hidden_states(
        self,
        token_ids: Tensor,
        *,
        extra_input_embeddings: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if extra_input_embeddings is None:
            raise ValueError("split inputs require position embeddings")
        content = self.embed(token_ids)
        if extra_input_embeddings.shape == (
            token_ids.shape[-1],
            self.position_dim,
        ):
            positions = extra_input_embeddings.to(
                device=content.device,
                dtype=content.dtype,
            ).unsqueeze(0).expand(content.shape[0], -1, -1)
        elif extra_input_embeddings.shape == (
            *token_ids.shape,
            self.position_dim,
        ):
            positions = extra_input_embeddings.to(
                device=content.device,
                dtype=content.dtype,
            )
        else:
            raise ValueError(
                "split position embeddings must have shape [time, position_dim] "
                "or [batch, time, position_dim]"
            )
        return self.hidden_states_from_embeddings(
            content,
            positions,
            attention_mask=attention_mask,
        )

    def hidden_states_from_embeddings(
        self,
        content_embeddings: Tensor,
        position_embeddings: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if content_embeddings.shape[:-1] != position_embeddings.shape[:-1]:
            raise ValueError("content and position sequence shapes must match")
        if content_embeddings.shape[-1] != self.content_dim:
            raise ValueError("content embedding dimension does not match")
        if position_embeddings.shape[-1] != self.position_dim:
            raise ValueError("position embedding dimension does not match")
        hidden = torch.cat((content_embeddings, position_embeddings), dim=-1)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.final_norm(hidden)
