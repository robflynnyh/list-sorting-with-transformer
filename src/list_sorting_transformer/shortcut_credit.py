"""Learned backward credit assignment for a controlled shortcut task."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .model import DecoderTransformer, ModelConfig
from .tokens import BOS, COMMA, PAD, SEP, VALUE_OFFSET, PointerNextVocabulary


LeakMode = Literal["correct", "masked", "incorrect"]


@dataclass(frozen=True)
class ShortcutPointerVocabulary(PointerNextVocabulary):
    """Pointer-next vocabulary with explicit leak, mask, and query markers."""

    @property
    def leak_token(self) -> int:
        return super().size

    @property
    def mask_token(self) -> int:
        return super().size + 1

    @property
    def query_token(self) -> int:
        return super().size + 2

    @property
    def size(self) -> int:
        return super().size + 3

    def encode_shortcut_prompt(
        self,
        values: Sequence[int],
        pointer_index: int,
        *,
        leak_mode: LeakMode,
        incorrect_value: int | None = None,
    ) -> tuple[list[int], int]:
        target = int(values[pointer_index + 1])
        if leak_mode == "correct":
            hint_token = self.value_token(target)
        elif leak_mode == "masked":
            hint_token = self.mask_token
        elif leak_mode == "incorrect":
            if incorrect_value is None:
                raise ValueError("incorrect leak mode requires an incorrect value")
            if incorrect_value == target:
                raise ValueError("incorrect leak must differ from the target")
            hint_token = self.value_token(incorrect_value)
        else:
            raise ValueError(f"unknown leak mode: {leak_mode}")
        prompt = [
            *self.encode_prompt_with_pointer(values, pointer_index),
            self.leak_token,
            hint_token,
            self.query_token,
        ]
        return prompt, self.value_token(target)

    def render_tokens(self, tokens: Sequence[int]) -> str:
        rendered = []
        for token_value in tokens:
            token = int(token_value)
            if token == self.leak_token:
                rendered.append("<LEAK>")
            elif token == self.mask_token:
                rendered.append("<MASK>")
            elif token == self.query_token:
                rendered.append("<QUERY>")
            else:
                rendered.append(super().render_tokens([token]))
        return " ".join(rendered)


@dataclass(frozen=True)
class ShortcutBatch:
    input_ids: Tensor
    targets: Tensor
    length: int
    leak_mode: LeakMode

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def to(self, device: torch.device | str) -> "ShortcutBatch":
        return ShortcutBatch(
            input_ids=self.input_ids.to(device),
            targets=self.targets.to(device),
            length=self.length,
            leak_mode=self.leak_mode,
        )


def make_shortcut_batch(
    batch_size: int,
    length: int,
    *,
    leak_mode: LeakMode,
    generator: torch.Generator,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device | str | None = None,
) -> ShortcutBatch:
    """Generate a same-length pointer batch with a controlled answer leak."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if length < 2:
        raise ValueError("length must be at least two")
    values = torch.randint(
        vocabulary.symbol_count,
        (batch_size, length),
        generator=generator,
    )
    pointers = torch.randint(
        length - 1,
        (batch_size,),
        generator=generator,
    )
    target_values = values[
        torch.arange(batch_size),
        pointers + 1,
    ]
    incorrect_values = None
    if leak_mode == "incorrect":
        nonzero_offsets = torch.randint(
            1,
            vocabulary.symbol_count,
            (batch_size,),
            generator=generator,
        )
        incorrect_values = (
            target_values + nonzero_offsets
        ) % vocabulary.symbol_count

    prompts = []
    targets = []
    for row_index in range(batch_size):
        incorrect_value = (
            None
            if incorrect_values is None
            else int(incorrect_values[row_index])
        )
        prompt, target = vocabulary.encode_shortcut_prompt(
            values[row_index].tolist(),
            int(pointers[row_index]),
            leak_mode=leak_mode,
            incorrect_value=incorrect_value,
        )
        prompts.append(prompt)
        targets.append(target)
    input_ids = torch.tensor(prompts, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    if device is not None:
        input_ids = input_ids.to(device)
        target_tensor = target_tensor.to(device)
    return ShortcutBatch(
        input_ids=input_ids,
        targets=target_tensor,
        length=length,
        leak_mode=leak_mode,
    )


def make_fitness_batches(
    example_count: int,
    *,
    min_length: int,
    max_length: int,
    batch_size: int,
    generator: torch.Generator,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device | str | None = None,
) -> tuple[ShortcutBatch, ...]:
    """Create a fixed, balanced masked/incorrect fitness dataset."""

    if example_count < 2 or example_count % 2:
        raise ValueError("fitness example count must be positive and even")
    if not 2 <= min_length <= max_length:
        raise ValueError("invalid length range")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    grouped_counts: dict[tuple[LeakMode, int], int] = defaultdict(int)
    examples_per_mode = example_count // 2
    for leak_mode in ("masked", "incorrect"):
        sampled_lengths = torch.randint(
            min_length,
            max_length + 1,
            (examples_per_mode,),
            generator=generator,
        )
        for length in sampled_lengths.tolist():
            grouped_counts[(leak_mode, int(length))] += 1

    batches = []
    for (leak_mode, length), count in sorted(grouped_counts.items()):
        remaining = count
        while remaining:
            current_batch_size = min(batch_size, remaining)
            batches.append(
                make_shortcut_batch(
                    current_batch_size,
                    length,
                    leak_mode=leak_mode,
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                )
            )
            remaining -= current_batch_size
    return tuple(batches)


class BackwardSwiGLU(nn.Module):
    def __init__(self, d_model: int, multiplier: float) -> None:
        super().__init__()
        hidden_dim = int(d_model * multiplier)
        self.input = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.input(hidden).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class ReverseCausalBackwardBlock(nn.Module):
    """Transformer block whose token dependencies follow backward causality."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = BackwardSwiGLU(d_model, ffn_multiplier)

    def forward(self, hidden: Tensor) -> Tensor:
        sequence_length = hidden.shape[1]
        reverse_causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        ).tril(diagonal=-1)
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=reverse_causal_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclass(frozen=True)
class BackwardRuleConfig:
    d_model: int = 128
    forward_d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    forward_layers: int = 3
    ffn_multiplier: float = 4.0
    rms_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if min(
            self.d_model,
            self.forward_d_model,
            self.n_layers,
            self.n_heads,
            self.forward_layers,
        ) < 1:
            raise ValueError("backward-rule dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("backward d_model must be divisible by n_heads")
        if self.ffn_multiplier <= 0 or self.rms_epsilon <= 0:
            raise ValueError("backward-rule scaling must be positive")


class LearnedBackwardRule(nn.Module):
    """Identity-anchored shared Transformer that edits upstream gradients."""

    def __init__(self, config: BackwardRuleConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(
            2 * config.forward_d_model,
            config.d_model,
            bias=False,
        )
        self.layer_embedding = nn.Embedding(
            config.forward_layers,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            ReverseCausalBackwardBlock(
                config.d_model,
                config.n_heads,
                config.ffn_multiplier,
            )
            for _ in range(config.n_layers)
        )
        self.output_projection = nn.Linear(
            config.d_model,
            config.forward_d_model,
            bias=False,
        )
        self.gates = nn.Parameter(torch.zeros(config.forward_layers))
        self.capture_statistics = False
        self.statistics: list[dict[str, float]] = []
        self.apply(DecoderTransformer._initialize)
        nn.init.zeros_(self.gates)
        self.requires_grad_(False)

    def clear_statistics(self) -> None:
        self.statistics.clear()

    def transform(
        self,
        gradient: Tensor,
        activation: Tensor,
        layer_index: int,
    ) -> Tensor:
        """Return the gradient passed into one ordinary block backward."""

        if not 0 <= layer_index < self.config.forward_layers:
            raise IndexError("forward layer index is outside the backward rule")
        gate = self.gates[layer_index]
        if gate.item() == 0.0:
            if self.capture_statistics:
                self.statistics.append(
                    {
                        "layer": float(layer_index),
                        "cosine": 1.0,
                        "correction_rms_ratio": 0.0,
                    }
                )
            return gradient

        gradient_scale = gradient.square().mean(
            dim=(-2, -1),
            keepdim=True,
        ).add(self.config.rms_epsilon).sqrt()
        normalized_gradient = gradient / gradient_scale
        normalized_activation = F.layer_norm(
            activation,
            (activation.shape[-1],),
        )
        hidden = self.input_projection(
            torch.cat((normalized_gradient, normalized_activation), dim=-1)
        )
        layer_ids = torch.full(
            (gradient.shape[0], gradient.shape[1]),
            layer_index,
            dtype=torch.long,
            device=gradient.device,
        )
        hidden = hidden + self.layer_embedding(layer_ids)
        for block in self.blocks:
            hidden = block(hidden)
        correction = self.output_projection(hidden)
        unscaled = gradient + gate * correction * gradient_scale
        unscaled_rms = unscaled.square().mean(
            dim=(-2, -1),
            keepdim=True,
        ).add(self.config.rms_epsilon).sqrt()
        modified = unscaled * (gradient_scale / unscaled_rms)

        if self.capture_statistics:
            flattened_gradient = gradient.flatten(start_dim=1)
            flattened_modified = modified.flatten(start_dim=1)
            cosine = F.cosine_similarity(
                flattened_gradient,
                flattened_modified,
                dim=-1,
            ).mean()
            correction_rms = (gate * correction * gradient_scale).square().mean(
                dim=(-2, -1)
            ).sqrt()
            original_rms = gradient.square().mean(
                dim=(-2, -1)
            ).add(self.config.rms_epsilon).sqrt()
            self.statistics.append(
                {
                    "layer": float(layer_index),
                    "cosine": float(cosine),
                    "correction_rms_ratio": float(
                        (correction_rms / original_rms).mean()
                    ),
                }
            )
        return modified


class BidirectionalRoutingBlock(nn.Module):
    """A bidirectional context layer for input-dependent credit routing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = BackwardSwiGLU(d_model, ffn_multiplier)

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclass(frozen=True)
class AttentionRoutingRuleConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    forward_layers: int = 3
    ffn_multiplier: float = 4.0
    routing_temperature: float = 0.1
    max_log_suppression: float = 8.0
    route_output_projection: bool = False

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.d_model,
            self.n_heads,
            self.forward_layers,
        ) < 1:
            raise ValueError("routing-rule dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("routing d_model must be divisible by n_heads")
        if min(
            self.ffn_multiplier,
            self.routing_temperature,
            self.max_log_suppression,
        ) <= 0:
            raise ValueError("routing-rule scaling must be positive")


class AttentionRoutingRule(nn.Module):
    """Input-conditioned, suppress-only routing for attention backward maps."""

    def __init__(self, config: AttentionRoutingRuleConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.context = BidirectionalRoutingBlock(
            config.d_model,
            config.n_heads,
            config.ffn_multiplier,
        )
        self.routing_norm = nn.LayerNorm(config.d_model)
        routing_width = config.forward_layers * config.d_model
        self.routing_query = nn.Linear(
            config.d_model,
            routing_width,
            bias=False,
        )
        self.routing_key = nn.Linear(
            config.d_model,
            routing_width,
            bias=False,
        )
        self.gates = nn.Parameter(
            torch.zeros(config.forward_layers, config.n_heads)
        )
        self.capture_statistics = False
        self.statistics: list[dict[str, float]] = []
        self.apply(DecoderTransformer._initialize)
        nn.init.zeros_(self.gates)
        self.requires_grad_(False)

    def clear_statistics(self) -> None:
        self.statistics.clear()

    def _position_encoding(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        half_width = (self.config.d_model + 1) // 2
        inverse_frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(
                half_width,
                device=device,
                dtype=torch.float32,
            )
            / max(half_width - 1, 1)
        )
        angles = torch.outer(
            torch.arange(
                sequence_length,
                device=device,
                dtype=torch.float32,
            ),
            inverse_frequency,
        )
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(
            start_dim=-2
        )
        return encoding[:, : self.config.d_model].to(dtype=dtype)

    def attention_gates(self, token_ids: Tensor) -> tuple[Tensor, ...]:
        """Return one existing-edge suppression map per forward layer."""

        batch_size, sequence_length = token_ids.shape
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self._position_encoding(
            sequence_length,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = self.context(hidden)
        hidden = self.routing_norm(hidden)

        head_dim = self.config.d_model // self.config.n_heads
        routing_shape = (
            batch_size,
            sequence_length,
            self.config.forward_layers,
            self.config.n_heads,
            head_dim,
        )
        query = self.routing_query(hidden).view(routing_shape).permute(
            0, 2, 3, 1, 4
        )
        key = self.routing_key(hidden).view(routing_shape).permute(
            0, 2, 3, 1, 4
        )
        scores = (
            query @ key.transpose(-2, -1)
            / head_dim**0.5
            / self.config.routing_temperature
        )
        reverse_causal = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden.device,
        ).triu()
        reverse_weights = scores.masked_fill(
            ~reverse_causal,
            float("-inf"),
        ).softmax(dim=-1)
        valid_destinations = torch.arange(
            sequence_length,
            0,
            -1,
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(1, 1, 1, sequence_length, 1)
        routing_priority = reverse_weights * valid_destinations
        active_strength = (
            self.config.max_log_suppression * F.relu(self.gates)
        ).view(
            1,
            self.config.forward_layers,
            self.config.n_heads,
            1,
            1,
        )
        log_suppression = (
            active_strength * routing_priority
        ).clamp(max=self.config.max_log_suppression)
        forward_gates = (-log_suppression).exp().transpose(-2, -1)

        if self.capture_statistics:
            causal = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden.device,
            ).tril()
            valid_gates = forward_gates[..., causal]
            leak_gates = forward_gates[..., -1, -2]
            other_query_gates = torch.cat(
                (
                    forward_gates[..., -1, :-2],
                    forward_gates[..., -1, -1:],
                ),
                dim=-1,
            )
            self.statistics.append(
                {
                    "routing_gate": float(valid_gates.mean()),
                    "routing_min_gate": float(valid_gates.min()),
                    "routing_suppressed_fraction": float(
                        (valid_gates < 0.99).float().mean()
                    ),
                    "routing_strength": float(active_strength.mean()),
                    "routing_leak_gate": float(leak_gates.mean()),
                    "routing_query_other_gate": float(
                        other_query_gates.mean()
                    ),
                    "routing_leak_relative_gate": float(
                        leak_gates.mean()
                        / other_query_gates.mean().clamp_min(
                            torch.finfo(hidden.dtype).tiny
                        )
                    ),
                }
            )
        return tuple(forward_gates.unbind(dim=1))


BackwardRule = Union[LearnedBackwardRule, AttentionRoutingRule]


class ShortcutDecoderTransformer(DecoderTransformer):
    """Decoder Transformer that can install training-only backward hooks."""

    def forward_with_backward_rule(
        self,
        token_ids: Tensor,
        backward_rule: BackwardRule | None,
    ) -> Tensor:
        hidden = self.embed(token_ids)
        attention_gates = (
            backward_rule.attention_gates(token_ids)
            if isinstance(backward_rule, AttentionRoutingRule)
            else None
        )
        for layer_index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                backward_attention_gate=(
                    None
                    if attention_gates is None
                    else attention_gates[layer_index]
                ),
                route_output_projection=(
                    isinstance(backward_rule, AttentionRoutingRule)
                    and backward_rule.config.route_output_projection
                ),
            )
            if (
                isinstance(backward_rule, LearnedBackwardRule)
                and hidden.requires_grad
            ):
                activation = hidden.detach()

                def transform_gradient(
                    gradient: Tensor,
                    *,
                    saved_activation: Tensor = activation,
                    saved_layer_index: int = layer_index,
                ) -> Tensor:
                    return backward_rule.transform(
                        gradient,
                        saved_activation,
                        saved_layer_index,
                    )

                hidden.register_hook(transform_gradient)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)


@dataclass(frozen=True)
class EggrollDirection:
    tensors: dict[str, Tensor]


def move_eggroll_direction(
    direction: EggrollDirection,
    device: torch.device | str,
) -> EggrollDirection:
    return EggrollDirection(
        {
            name: tensor.to(device)
            for name, tensor in direction.tensors.items()
        }
    )


def sample_eggroll_direction(
    module: nn.Module,
    *,
    generator: torch.Generator,
) -> EggrollDirection:
    """Sample rank-one matrix noise and Gaussian vector noise."""

    tensors = {}
    for name, parameter in module.named_parameters():
        if parameter.ndim == 2:
            left = torch.randn(
                parameter.shape[0],
                1,
                generator=generator,
                dtype=parameter.dtype,
            )
            right = torch.randn(
                parameter.shape[1],
                1,
                generator=generator,
                dtype=parameter.dtype,
            )
            noise = left @ right.T
        else:
            noise = torch.randn(
                parameter.shape,
                generator=generator,
                dtype=parameter.dtype,
            )
        tensors[name] = noise.to(parameter.device)
    return EggrollDirection(tensors)


@torch.no_grad()
def apply_eggroll_direction(
    module: nn.Module,
    center_parameters: dict[str, Tensor],
    direction: EggrollDirection,
    *,
    sigma: float,
    sign: int,
) -> None:
    if sign not in {-1, 1}:
        raise ValueError("antithetic sign must be -1 or +1")
    for name, parameter in module.named_parameters():
        parameter.copy_(
            center_parameters[name]
            + sign * sigma * direction.tensors[name]
        )


def clone_center_parameters(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


@torch.no_grad()
def paper_eggroll_update(
    module: nn.Module,
    directions: Sequence[EggrollDirection],
    fitnesses: Tensor,
    *,
    sigma: float,
    learning_rate: float,
) -> Tensor:
    """Apply the official standardized-fitness EGGROLL SGD update."""

    if fitnesses.ndim != 1 or fitnesses.numel() != 2 * len(directions):
        raise ValueError("fitnesses must contain one positive/negative pair")
    standardized = (
        fitnesses - fitnesses.mean()
    ) / torch.sqrt(fitnesses.var(unbiased=False) + 1e-5)
    population_size = fitnesses.numel()
    scale = learning_rate * math.sqrt(population_size) / population_size
    for name, parameter in module.named_parameters():
        update = torch.zeros_like(parameter)
        for direction_index, direction in enumerate(directions):
            positive_weight = standardized[2 * direction_index]
            negative_weight = standardized[2 * direction_index + 1]
            update.add_(
                sigma
                * (positive_weight - negative_weight)
                * direction.tensors[name]
            )
        parameter.add_(scale * update)
    return standardized


def shortcut_loss(
    model: ShortcutDecoderTransformer,
    batch: ShortcutBatch,
    backward_rule: BackwardRule | None = None,
) -> Tensor:
    logits = model.forward_with_backward_rule(batch.input_ids, backward_rule)
    return F.cross_entropy(logits[:, -1], batch.targets)


@dataclass(frozen=True)
class ShortcutMetrics:
    loss: float
    accuracy: float
    mode_accuracy: dict[str, float]
    mode_loss: dict[str, float]
    unique_prediction_count: int
    unique_value_prediction_count: int
    prediction_mode_fraction: float


@torch.inference_mode()
def evaluate_shortcut_batches(
    model: ShortcutDecoderTransformer,
    batches: Iterable[ShortcutBatch],
    *,
    evaluation_batch_size: int = 64,
) -> ShortcutMetrics:
    if evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    mode_correct: dict[str, int] = defaultdict(int)
    mode_loss: dict[str, float] = defaultdict(float)
    mode_examples: dict[str, int] = defaultdict(int)
    prediction_counts = torch.zeros(
        model.config.vocab_size,
        dtype=torch.long,
    )
    examples: list[tuple[Tensor, Tensor, LeakMode]] = []
    for batch in batches:
        examples.extend(
            (input_ids, target, batch.leak_mode)
            for input_ids, target in zip(batch.input_ids, batch.targets)
        )
    for start in range(0, len(examples), evaluation_batch_size):
        chunk = examples[start : start + evaluation_batch_size]
        input_rows = [example[0] for example in chunk]
        query_positions = torch.tensor(
            [row.shape[0] - 1 for row in input_rows],
            device=input_rows[0].device,
        )
        input_ids = pad_sequence(
            input_rows,
            batch_first=True,
            padding_value=PAD,
        )
        targets = torch.stack([example[1] for example in chunk])
        all_logits = model(input_ids)
        logits = all_logits[
            torch.arange(len(chunk), device=input_ids.device),
            query_positions,
        ]
        losses = F.cross_entropy(logits, targets, reduction="none")
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(targets)
        total_loss += float(losses.sum())
        total_correct += int(correct.sum())
        total_examples += len(chunk)
        chunk_modes = [example[2] for example in chunk]
        for leak_mode in set(chunk_modes):
            indices = [
                index
                for index, mode in enumerate(chunk_modes)
                if mode == leak_mode
            ]
            mode_correct[leak_mode] += int(correct[indices].sum())
            mode_loss[leak_mode] += float(losses[indices].sum())
            mode_examples[leak_mode] += len(indices)
        prediction_counts += torch.bincount(
            predictions.detach().cpu(),
            minlength=model.config.vocab_size,
        )
    if total_examples == 0:
        raise ValueError("evaluation requires at least one example")
    return ShortcutMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        mode_accuracy={
            mode: mode_correct[mode] / count
            for mode, count in mode_examples.items()
        },
        mode_loss={
            mode: mode_loss[mode] / count
            for mode, count in mode_examples.items()
        },
        unique_prediction_count=int(prediction_counts.count_nonzero()),
        unique_value_prediction_count=int(
            prediction_counts[
                VALUE_OFFSET : VALUE_OFFSET + model.config.symbol_count
            ].count_nonzero()
        ),
        prediction_mode_fraction=float(prediction_counts.max()) / total_examples,
    )


def make_forward_model_config(
    vocabulary: ShortcutPointerVocabulary,
    *,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocabulary.size,
        symbol_count=vocabulary.symbol_count,
        representation=vocabulary.representation,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_multiplier=4.0,
        dropout=0.0,
        position_pattern="rotary",
    )


def clone_module(module: nn.Module) -> nn.Module:
    """Deep-copy helper kept explicit for rollout reset tests."""

    return deepcopy(module)
