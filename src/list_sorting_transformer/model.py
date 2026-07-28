"""A compact decoder-only Transformer with interleaved RoPE and NoPE layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tokens import EOS, PAD, VALUE_OFFSET


KeyValueCache = tuple[Tensor, Tensor]


class _RoutedAttentionBackward(torch.autograd.Function):
    """Keep the normal forward output but use routed attention in backward."""

    @staticmethod
    def forward(
        ctx: object,
        attended: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        routed_weights: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(query, key, value, routed_weights)
        return attended

    @staticmethod
    def backward(
        ctx: object,
        output_gradient: Tensor,
    ) -> tuple[
        None,
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
    ]:
        query, key, value, routed_weights = ctx.saved_tensors
        scale = query.shape[-1] ** -0.5

        value_gradient = routed_weights.transpose(-2, -1) @ output_gradient
        weight_gradient = output_gradient @ value.transpose(-2, -1)
        score_gradient = routed_weights * (
            weight_gradient
            - (weight_gradient * routed_weights).sum(dim=-1, keepdim=True)
        )
        query_gradient = (score_gradient @ key) * scale
        key_gradient = (
            score_gradient.transpose(-2, -1) @ query
        ) * scale
        return (
            None,
            query_gradient,
            key_gradient,
            value_gradient,
            None,
        )


class _SourceReversedAttentionBackward(torch.autograd.Function):
    """Keep attention forward exact while reversing selected source credit."""

    @staticmethod
    def forward(
        ctx: object,
        attended: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        weights: Tensor,
        source_multipliers: Tensor,
        reverse_score_credit: bool,
        reverse_value_credit: bool,
        attention_penalty_strength: float,
    ) -> Tensor:
        ctx.reverse_score_credit = reverse_score_credit
        ctx.reverse_value_credit = reverse_value_credit
        ctx.attention_penalty_strength = attention_penalty_strength
        ctx.save_for_backward(
            query,
            key,
            value,
            weights,
            source_multipliers,
        )
        return attended

    @staticmethod
    def backward(
        ctx: object,
        output_gradient: Tensor,
    ) -> tuple[
        None,
        Tensor,
        Tensor,
        Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            query,
            key,
            value,
            weights,
            source_multipliers,
        ) = ctx.saved_tensors
        scale = query.shape[-1] ** -0.5
        edge_multipliers = source_multipliers[:, None, None, :]

        value_gradient = (
            (weights * edge_multipliers)
            if ctx.reverse_value_credit
            else weights
        ).transpose(-2, -1) @ output_gradient
        weight_gradient = output_gradient @ value.transpose(-2, -1)
        if ctx.reverse_score_credit:
            weight_gradient = weight_gradient * edge_multipliers
        score_gradient = weights * (
            weight_gradient
            - (weight_gradient * weights).sum(dim=-1, keepdim=True)
        )
        if ctx.attention_penalty_strength:
            selected_sources = source_multipliers.lt(0).to(weights.dtype)
            penalty_weight_gradient = (
                selected_sources[:, None, None, :]
                * ctx.attention_penalty_strength
                / (
                    weights.shape[0]
                    * weights.shape[1]
                    * weights.shape[2]
                )
            )
            score_gradient = score_gradient + weights * (
                penalty_weight_gradient
                - (
                    penalty_weight_gradient * weights
                ).sum(dim=-1, keepdim=True)
            )
        query_gradient = (score_gradient @ key) * scale
        key_gradient = (
            score_gradient.transpose(-2, -1) @ query
        ) * scale
        return (
            None,
            query_gradient,
            key_gradient,
            value_gradient,
            None,
            None,
            None,
            None,
            None,
        )


class _RoutedLinearBackward(torch.autograd.Function):
    """Keep a linear forward exact while changing its parameter credit."""

    @staticmethod
    def forward(
        ctx: object,
        input: Tensor,
        routed_input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        ctx.has_bias = bias is not None
        ctx.save_for_backward(routed_input, weight)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(
        ctx: object,
        output_gradient: Tensor,
    ) -> tuple[Tensor, None, Tensor, Tensor | None]:
        routed_input, weight = ctx.saved_tensors
        input_gradient = output_gradient @ weight
        flattened_output = output_gradient.reshape(
            -1, output_gradient.shape[-1]
        )
        flattened_input = routed_input.reshape(-1, routed_input.shape[-1])
        weight_gradient = flattened_output.T @ flattened_input
        bias_gradient = (
            flattened_output.sum(dim=0) if ctx.has_bias else None
        )
        return input_gradient, None, weight_gradient, bias_gradient


def _attention_weights(
    query: Tensor,
    key: Tensor,
    *,
    attention_mask: Tensor | None,
    is_causal: bool,
) -> Tensor:
    scores = query @ key.transpose(-2, -1) / query.shape[-1] ** 0.5
    if is_causal:
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        causal_mask = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=query.device,
        ).tril(diagonal=key_length - query_length)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float("-inf"))

    return scores.softmax(dim=-1)


def _routed_attention_weights(
    query: Tensor,
    key: Tensor,
    *,
    backward_gate: Tensor,
    attention_mask: Tensor | None,
    is_causal: bool,
) -> Tensor:
    weights = _attention_weights(
        query,
        key,
        attention_mask=attention_mask,
        is_causal=is_causal,
    )
    routed_weights = weights * backward_gate.to(
        device=weights.device,
        dtype=weights.dtype,
    )
    return routed_weights / routed_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(routed_weights.dtype).tiny)


def routed_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    backward_gate: Tensor,
    attention_mask: Tensor | None = None,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run normal attention forward with a suppress-only backward routing map."""

    if backward_gate.shape != (
        query.shape[0],
        query.shape[1],
        query.shape[-2],
        key.shape[-2],
    ):
        raise ValueError("backward gate must match the attention map")
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    routed_weights = _routed_attention_weights(
        query,
        key,
        backward_gate=backward_gate,
        attention_mask=attention_mask,
        is_causal=is_causal,
    )
    routed_surrogate = routed_weights @ value
    routed = (
        attended.detach()
        + (routed_surrogate - routed_surrogate.detach())
    )
    manually_routed_attended = (
        routed_weights.detach() @ value.detach()
    )
    routed_attended = torch.where(
        backward_gate.eq(1).all(),
        attended.detach(),
        manually_routed_attended,
    )
    return routed, routed_attended


def signed_routed_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    backward_multipliers: Tensor,
    attention_mask: Tensor | None = None,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run exact attention forward with signed per-edge backward credit."""

    if backward_multipliers.shape != (
        query.shape[0],
        query.shape[1],
        query.shape[-2],
        key.shape[-2],
    ):
        raise ValueError("backward multipliers must match the attention map")
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    weights = _attention_weights(
        query,
        key,
        attention_mask=attention_mask,
        is_causal=is_causal,
    )
    multipliers = backward_multipliers.detach().to(
        device=weights.device,
        dtype=weights.dtype,
    )
    score_surrogate = (weights * multipliers) @ value.detach()
    value_surrogate = (weights.detach() * multipliers) @ value
    backward_surrogate = score_surrogate + value_surrogate
    routed = (
        attended.detach()
        + (backward_surrogate - backward_surrogate.detach())
    )
    routed_attended = (
        weights.detach() * multipliers
    ) @ value.detach()
    return routed, routed_attended


def source_reversed_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    source_multipliers: Tensor,
    reverse_score_credit: bool = True,
    reverse_value_credit: bool = True,
    attention_penalty_strength: float = 0.0,
    attention_mask: Tensor | None = None,
    is_causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run exact attention forward with per-source backward multipliers."""

    expected_shape = (query.shape[0], key.shape[-2])
    if source_multipliers.shape != expected_shape:
        raise ValueError(
            "source multipliers must have shape [batch, key_time]"
        )
    if attention_penalty_strength < 0:
        raise ValueError("attention penalty strength must be nonnegative")

    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    scores = query.detach() @ key.detach().transpose(
        -2, -1
    ) / query.shape[-1] ** 0.5
    if is_causal:
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        causal_mask = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=query.device,
        ).tril(diagonal=key_length - query_length)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    multipliers = source_multipliers.detach().to(
        device=weights.device,
        dtype=weights.dtype,
    )
    if (
        reverse_score_credit
        and not reverse_value_credit
        and attention_penalty_strength == 0
    ):
        live_scores = query @ key.transpose(
            -2,
            -1,
        ) / query.shape[-1] ** 0.5
        if is_causal:
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            causal_mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=query.device,
            ).tril(diagonal=key_length - query_length)
            live_scores = live_scores.masked_fill(
                ~causal_mask,
                float("-inf"),
            )
        if attention_mask is not None:
            live_scores = live_scores.masked_fill(
                ~attention_mask,
                float("-inf"),
            )
        live_weights = live_scores.softmax(dim=-1)
        edge_multipliers = multipliers[:, None, None, :]
        routed_weights = (
            live_weights * edge_multipliers
            + live_weights.detach() * (1 - edge_multipliers)
        )
        routed_attention = routed_weights @ value
        reversed_attended = routed_attention + (
            attended - routed_attention
        ).detach()
        return reversed_attended, live_weights.detach() @ value.detach()
    backward_attended = (
        (
            weights * multipliers[:, None, None, :]
            if reverse_value_credit
            else weights
        )
        @ value.detach()
    )
    reversed_attended = _SourceReversedAttentionBackward.apply(
        attended.detach(),
        query,
        key,
        value,
        weights,
        multipliers,
        reverse_score_credit,
        reverse_value_credit,
        attention_penalty_strength,
    )
    return reversed_attended, backward_attended


def routed_linear(
    input: Tensor,
    routed_input: Tensor,
    layer: nn.Linear,
) -> Tensor:
    """Use routed activations only for a linear layer's parameter gradient."""

    return _RoutedLinearBackward.apply(
        input,
        routed_input.detach(),
        layer.weight,
        layer.bias,
    )


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
        self.top_k: int | None = None
        self.top_k_straight_through = False

    def configure_top_k(
        self,
        top_k: int | None,
        *,
        straight_through: bool = False,
    ) -> None:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if straight_through and top_k is None:
            raise ValueError("straight-through attention requires top_k")
        self.top_k = top_k
        self.top_k_straight_through = straight_through

    def _top_k_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if is_causal:
            causal_mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=query.device,
            ).tril(diagonal=key_length - query_length)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        soft_weights = scores.softmax(dim=-1)
        selected = scores.topk(
            min(self.top_k or key_length, key_length),
            dim=-1,
        ).indices
        hard_scores = torch.full_like(scores, float("-inf"))
        hard_scores.scatter_(-1, selected, scores.gather(-1, selected))
        hard_weights = hard_scores.softmax(dim=-1)
        if self.top_k_straight_through and self.training:
            weights = soft_weights + (hard_weights - soft_weights).detach()
        else:
            weights = hard_weights
        weights = F.dropout(
            weights,
            p=self.dropout,
            training=self.training,
        )
        return weights @ value

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
        backward_attention_gate: Tensor | None = None,
        signed_backward_attention: bool = False,
        backward_source_multipliers: Tensor | None = None,
        reverse_source_score_credit: bool = True,
        reverse_source_value_credit: bool = True,
        source_attention_penalty_strength: float = 0.0,
        route_source_output_projection: bool = True,
        route_output_projection: bool = False,
    ) -> tuple[Tensor, KeyValueCache]:
        batch_size, sequence_length, model_dim = hidden.shape
        if cache is not None and attention_mask is not None:
            raise ValueError("custom attention masks are not supported with a cache")
        if cache is not None and backward_attention_gate is not None:
            raise ValueError("backward attention routing is not cacheable")
        if cache is not None and backward_source_multipliers is not None:
            raise ValueError("source-gradient reversal is not cacheable")
        if (
            backward_attention_gate is not None
            and backward_source_multipliers is not None
        ):
            raise ValueError(
                "attention routing and source-gradient reversal are exclusive"
            )
        if signed_backward_attention and backward_attention_gate is None:
            raise ValueError(
                "signed backward attention requires an attention gate"
            )
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
        if backward_attention_gate is not None:
            if self.top_k is not None:
                raise ValueError(
                    "backward attention routing does not support top-k attention"
                )
            if self.dropout and self.training:
                raise ValueError(
                    "backward attention routing requires zero attention dropout"
                )
            if signed_backward_attention:
                attended, routed_attended = (
                    signed_routed_scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        backward_multipliers=backward_attention_gate,
                        attention_mask=combined_mask,
                        is_causal=combined_mask is None,
                    )
                )
            else:
                attended, routed_attended = (
                    routed_scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        backward_gate=backward_attention_gate,
                        attention_mask=combined_mask,
                        is_causal=combined_mask is None,
                    )
                )
        elif backward_source_multipliers is not None:
            if self.top_k is not None:
                raise ValueError(
                    "source-gradient reversal does not support top-k attention"
                )
            if self.dropout and self.training:
                raise ValueError(
                    "source-gradient reversal requires zero attention dropout"
                )
            attended, routed_attended = (
                source_reversed_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    source_multipliers=backward_source_multipliers,
                    reverse_score_credit=reverse_source_score_credit,
                    reverse_value_credit=reverse_source_value_credit,
                    attention_penalty_strength=(
                        source_attention_penalty_strength
                    ),
                    attention_mask=combined_mask,
                    is_causal=combined_mask is None,
                )
            )
        elif self.top_k is None:
            routed_attended = None
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=combined_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=cache is None and combined_mask is None,
            )
        else:
            routed_attended = None
            attended = self._top_k_attention(
                query,
                key,
                value,
                attention_mask=combined_mask,
                is_causal=cache is None and combined_mask is None,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            model_dim,
        )
        if (
            route_output_projection
            or (
                backward_source_multipliers is not None
                and route_source_output_projection
            )
        ):
            if routed_attended is None:
                raise ValueError(
                    "output-projection routing requires backward routing"
                )
            routed_attended = routed_attended.transpose(
                1, 2
            ).contiguous().view(
                batch_size,
                sequence_length,
                model_dim,
            )
            projected = routed_linear(
                attended,
                routed_attended,
                self.output,
            )
        else:
            projected = self.output(attended)
        return projected, (key, value)

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
        backward_attention_gate: Tensor | None = None,
        signed_backward_attention: bool = False,
        backward_source_multipliers: Tensor | None = None,
        reverse_source_score_credit: bool = True,
        reverse_source_value_credit: bool = True,
        source_attention_penalty_strength: float = 0.0,
        route_source_output_projection: bool = True,
        route_output_projection: bool = False,
    ) -> Tensor:
        attended, _ = self.forward_with_cache(
            hidden,
            attention_mask=attention_mask,
            backward_attention_gate=backward_attention_gate,
            signed_backward_attention=signed_backward_attention,
            backward_source_multipliers=backward_source_multipliers,
            reverse_source_score_credit=reverse_source_score_credit,
            reverse_source_value_credit=reverse_source_value_credit,
            source_attention_penalty_strength=(
                source_attention_penalty_strength
            ),
            route_source_output_projection=route_source_output_projection,
            route_output_projection=route_output_projection,
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
        backward_attention_gate: Tensor | None = None,
        signed_backward_attention: bool = False,
        backward_source_multipliers: Tensor | None = None,
        reverse_source_score_credit: bool = True,
        reverse_source_value_credit: bool = True,
        source_attention_penalty_strength: float = 0.0,
        route_source_output_projection: bool = True,
        route_output_projection: bool = False,
    ) -> Tensor:
        hidden = hidden + self.dropout(
            self.attention(
                self.attention_norm(hidden),
                attention_mask=attention_mask,
                backward_attention_gate=backward_attention_gate,
                signed_backward_attention=signed_backward_attention,
                backward_source_multipliers=backward_source_multipliers,
                reverse_source_score_credit=reverse_source_score_credit,
                reverse_source_value_credit=reverse_source_value_credit,
                source_attention_penalty_strength=(
                    source_attention_penalty_strength
                ),
                route_source_output_projection=route_source_output_projection,
                route_output_projection=route_output_projection,
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
