"""Forward-only EGGROLL training for a hard-attention pointer Transformer."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .compiled_pointer_compare import (
    DEFAULT_POSITION_MODULI,
    _set_modular_fourier_codebooks,
)
from .data import PointerNextBatch, make_pointer_next_batch
from .evaluate import resolve_device
from .model import (
    ModelConfig,
    SplitInputDecoderTransformer,
    sample_top_k_indices,
)
from .positions import ModularPositionEmbedding, sample_position_offsets
from .tokens import VALUE_OFFSET, PointerNextVocabulary


@dataclass(frozen=True)
class HardAttentionEggrollConfig:
    run_name: str = "clean-pointer-top1-eggroll-seed7"
    output_dir: str = "artifacts/hard_attention_eggroll"
    generations: int = 20_000
    population_size: int = 64
    population_chunk_size: int = 16
    batch_size: int = 256
    population_data_mode: str = "cartesian"
    population_precision: str = "float32"
    train_min_length: int = 2
    train_max_length: int = 20
    eval_lengths: tuple[int, ...] = (2, 5, 10, 20, 40, 100)
    eval_examples: int = 1_024
    long_eval_examples: int = 64
    long_eval_min_length: int = 1_000
    eval_batch_size: int = 128
    eval_attention_element_budget: int = 32_000_000
    symbol_count: int = 10
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn_multiplier: float = 4.0
    attention_mode: str = "top1"
    sample_sparse_attention: bool = False
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    sigma: float = 0.005
    learning_rate: float = 0.3
    momentum: float = 0.0
    weight_decay: float = 0.0
    fitness_shaping: str = "zscore"
    update_rule: str = "paper_standardized"
    elite_count: int = 8
    curriculum: bool = False
    curriculum_progress_mode: str = "probe"
    curriculum_accuracy_threshold: float = 0.70
    curriculum_success_checks: int = 3
    curriculum_check_interval: int = 500
    curriculum_examples: int = 1_024
    curriculum_initial_top_k: int = 20
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 1_000
    seed: int = 7
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "list-sorting-hard-attention-eggroll"
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        positive_integers = (
            self.generations,
            self.population_size,
            self.population_chunk_size,
            self.batch_size,
            self.train_min_length,
            self.train_max_length,
            self.eval_examples,
            self.long_eval_examples,
            self.long_eval_min_length,
            self.eval_batch_size,
            self.eval_attention_element_budget,
            self.curriculum_success_checks,
            self.curriculum_check_interval,
            self.curriculum_examples,
            self.curriculum_initial_top_k,
            self.symbol_count,
            self.d_model,
            self.layers,
            self.heads,
            self.log_interval,
            self.eval_interval,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("integer configuration values must be positive")
        if self.population_size % 2:
            raise ValueError("population size must be even")
        if self.population_chunk_size % 2:
            raise ValueError("population chunk size must be even")
        if self.population_chunk_size > self.population_size:
            raise ValueError("population chunk cannot exceed population size")
        if self.population_data_mode not in {"cartesian", "grouped"}:
            raise ValueError(
                "population data mode must be cartesian or grouped"
            )
        if self.population_precision not in {"float32", "bfloat16"}:
            raise ValueError(
                "population precision must be float32 or bfloat16"
            )
        if (
            self.population_data_mode == "grouped"
            and self.population_size // 2 % self.batch_size
        ):
            raise ValueError(
                "grouped mode requires antithetic pairs to divide evenly "
                "across the unique examples"
            )
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        if not self.eval_lengths or any(length < 2 for length in self.eval_lengths):
            raise ValueError("evaluation lengths must be at least two")
        if self.d_model % 2:
            raise ValueError("d_model must split evenly into content and position")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        position_dim = self.d_model // 2
        if position_dim != 8 * len(self.position_moduli):
            raise ValueError(
                "fixed modular Fourier positions require eight dimensions "
                "per modulus"
            )
        required_span = (
            self.position_offset_max
            - self.position_offset_min
            + 2 * max(self.eval_lengths)
            + 4
        )
        if math.prod(self.position_moduli) < required_span:
            raise ValueError("modular position period is too short")
        if self.position_offset_min > self.position_offset_max:
            raise ValueError("position offset bounds are reversed")
        if self.attention_mode not in {"top1", "dense"}:
            raise ValueError("attention mode must be top1 or dense")
        if self.fitness_shaping not in {"zscore", "centered_rank"}:
            raise ValueError("unknown fitness shaping")
        if self.update_rule not in {
            "paper_standardized",
            "elite_centroid",
        }:
            raise ValueError("unknown update rule")
        if (
            self.update_rule == "elite_centroid"
            and not 1 <= self.elite_count <= self.population_size // 2
        ):
            raise ValueError(
                "elite count must fit the unique antithetic directions"
            )
        if self.sigma <= 0 or self.learning_rate <= 0:
            raise ValueError("sigma and learning rate must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative")
        if not 0 < self.curriculum_accuracy_threshold <= 1:
            raise ValueError(
                "curriculum accuracy threshold must be in (0, 1]"
            )
        if self.curriculum_progress_mode not in {
            "probe",
            "training_streak",
        }:
            raise ValueError(
                "curriculum progress mode must be probe or training_streak"
            )


@dataclass
class CurriculumState:
    current_max_length: int
    attention_top_k: int | None
    active_heads: int
    success_streak: int = 0
    promotion_count: int = 0


def initialize_curriculum_state(
    config: HardAttentionEggrollConfig,
) -> CurriculumState:
    return CurriculumState(
        current_max_length=(
            config.train_min_length
            if config.curriculum
            else config.train_max_length
        ),
        attention_top_k=(
            None
            if config.curriculum or config.attention_mode == "dense"
            else 1
        ),
        active_heads=config.heads,
    )


def restore_curriculum_state(
    values: dict[str, Any],
    config: HardAttentionEggrollConfig,
) -> CurriculumState:
    """Load curriculum state, including checkpoints from before head pruning."""

    return CurriculumState(
        current_max_length=int(values["current_max_length"]),
        attention_top_k=values["attention_top_k"],
        active_heads=int(values.get("active_heads", config.heads)),
        success_streak=int(values.get("success_streak", 0)),
        promotion_count=int(values.get("promotion_count", 0)),
    )


def curriculum_is_complete(
    state: CurriculumState,
    config: HardAttentionEggrollConfig,
) -> bool:
    return (
        config.curriculum
        and state.current_max_length == config.train_max_length
        and state.attention_top_k == 1
        and state.active_heads == 1
    )


def update_curriculum(
    state: CurriculumState,
    config: HardAttentionEggrollConfig,
    *,
    criterion_accuracy: float,
) -> str | None:
    """Advance at most one length, attention-width, or head-count stage."""

    if not config.curriculum or curriculum_is_complete(state, config):
        return None
    if criterion_accuracy < config.curriculum_accuracy_threshold:
        state.success_streak = 0
        return None
    state.success_streak += 1
    if state.success_streak < config.curriculum_success_checks:
        return None

    state.success_streak = 0
    state.promotion_count += 1
    if state.current_max_length < config.train_max_length:
        state.current_max_length += 1
        return "length"
    if state.attention_top_k is None:
        state.attention_top_k = config.curriculum_initial_top_k
        return "start_sparsity"
    if state.attention_top_k > 1:
        state.attention_top_k -= 1
        return "increase_sparsity"
    if state.active_heads > 1:
        state.active_heads -= 1
        return "prune_head"
    return None


def curriculum_check_due(
    config: HardAttentionEggrollConfig,
    state: CurriculumState,
    *,
    generation: int,
    batch_length: int,
) -> bool:
    if not config.curriculum or curriculum_is_complete(state, config):
        return False
    if config.curriculum_progress_mode == "probe":
        return generation % config.curriculum_check_interval == 0
    return batch_length == state.current_max_length


class HardAttentionPointerTransformer(nn.Module):
    """Clean pointer-next model with fixed modular absolute positions."""

    def __init__(self, config: HardAttentionEggrollConfig) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = PointerNextVocabulary(
            "numbers",
            config.symbol_count,
        )
        model_config = ModelConfig(
            vocab_size=self.vocabulary.size,
            symbol_count=config.symbol_count,
            representation="numbers",
            d_model=config.d_model,
            n_layers=config.layers,
            n_heads=config.heads,
            ffn_multiplier=config.ffn_multiplier,
            dropout=0.0,
            position_pattern="none",
        )
        self.encoder = SplitInputDecoderTransformer(
            model_config,
            content_dim=config.d_model // 2,
        )
        self.position_embedding = ModularPositionEmbedding(
            self.encoder.position_dim,
            config.position_moduli,
        )
        with torch.no_grad():
            _set_modular_fourier_codebooks(self.position_embedding)
        self.position_embedding.requires_grad_(False)
        self.current_top_k: int | None = None
        self.current_active_heads = config.heads
        self.set_attention_top_k(
            1 if config.attention_mode == "top1" else None
        )
        self.set_active_heads(config.heads)
        self.output = nn.Linear(config.d_model, config.symbol_count)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def set_attention_top_k(self, top_k: int | None) -> None:
        if top_k is not None and top_k < 1:
            raise ValueError("attention top-k must be positive")
        self.current_top_k = top_k
        for block in self.encoder.blocks:
            block.attention.configure_top_k(
                top_k,
                sample=self.config.sample_sparse_attention
                and top_k is not None,
            )

    def set_active_heads(self, active_heads: int) -> None:
        if not 1 <= active_heads <= self.config.heads:
            raise ValueError(
                f"active heads must be in [1, {self.config.heads}]"
            )
        self.current_active_heads = active_heads
        for block in self.encoder.blocks:
            block.attention.configure_active_heads(active_heads)

    def position_embeddings(
        self,
        prompt_ids: Tensor,
        offsets: Tensor,
    ) -> Tensor:
        token_offsets = torch.arange(
            prompt_ids.shape[1],
            device=prompt_ids.device,
        )
        return self.position_embedding(
            offsets[:, None] + token_offsets[None, :]
        )

    def forward(self, prompt_ids: Tensor, *, offsets: Tensor) -> Tensor:
        hidden = self.encoder.hidden_states(
            prompt_ids,
            extra_input_embeddings=self.position_embeddings(
                prompt_ids,
                offsets,
            ),
        )
        return self.output(hidden[:, -1])


@dataclass(frozen=True)
class RankOneFactors:
    left: Tensor
    right: Tensor


@dataclass(frozen=True)
class SignedPopulationNoise:
    matrices: dict[str, RankOneFactors]
    vectors: dict[str, Tensor]

    @property
    def population_size(self) -> int:
        values = [
            *(factors.left for factors in self.matrices.values()),
            *self.vectors.values(),
        ]
        if not values:
            raise ValueError("population noise contains no parameters")
        return values[0].shape[0]


@dataclass(frozen=True)
class AntitheticRankOneNoise:
    matrices: dict[str, RankOneFactors]
    vectors: dict[str, Tensor]

    @property
    def pair_count(self) -> int:
        values = [
            *(factors.left for factors in self.matrices.values()),
            *self.vectors.values(),
        ]
        if not values:
            raise ValueError("antithetic noise contains no parameters")
        return values[0].shape[0]

    @property
    def population_size(self) -> int:
        return 2 * self.pair_count

    def pair_chunk(self, start: int, end: int) -> SignedPopulationNoise:
        if not 0 <= start < end <= self.pair_count:
            raise ValueError("invalid antithetic pair slice")
        matrices = {}
        for name, factors in self.matrices.items():
            left = factors.left[start:end]
            right = factors.right[start:end]
            matrices[name] = RankOneFactors(
                left=torch.cat((left, -left), dim=0),
                right=torch.cat((right, right), dim=0),
            )
        vectors = {
            name: torch.cat(
                (values[start:end], -values[start:end]),
                dim=0,
            )
            for name, values in self.vectors.items()
        }
        return SignedPopulationNoise(matrices=matrices, vectors=vectors)


def sample_antithetic_rank_one_noise(
    model: nn.Module,
    population_size: int,
    *,
    generator: torch.Generator,
) -> AntitheticRankOneNoise:
    if population_size < 2 or population_size % 2:
        raise ValueError("population size must be positive and even")
    pair_count = population_size // 2
    matrices = {}
    vectors = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2:
            matrices[name] = RankOneFactors(
                left=torch.randn(
                    pair_count,
                    parameter.shape[0],
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                ),
                right=torch.randn(
                    pair_count,
                    parameter.shape[1],
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                ),
            )
        else:
            vectors[name] = torch.randn(
                pair_count,
                *parameter.shape,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )
    return AntitheticRankOneNoise(matrices=matrices, vectors=vectors)


def _population_linear(
    inputs: Tensor,
    layer: nn.Linear,
    name: str,
    noise: SignedPopulationNoise,
    sigma: float,
    *,
    population_input: bool,
) -> Tensor:
    factors = noise.matrices[f"{name}.weight"]
    base = F.linear(inputs, layer.weight, layer.bias)
    if population_input:
        if inputs.shape[0] != noise.population_size:
            raise ValueError("population input has the wrong leading dimension")
        projected = torch.einsum(
            "p...i,pi->p...",
            inputs,
            factors.right,
        )
    else:
        base = base.unsqueeze(0).expand(
            noise.population_size,
            *base.shape,
        )
        projected = torch.einsum(
            "...i,pi->p...",
            inputs,
            factors.right,
        )
    left_shape = (
        noise.population_size,
        *((1,) * (projected.ndim - 1)),
        factors.left.shape[-1],
    )
    perturbation = projected.unsqueeze(-1) * factors.left.view(left_shape)
    output = base + sigma * perturbation
    if layer.bias is not None:
        bias_noise = noise.vectors[f"{name}.bias"]
        bias_shape = (
            noise.population_size,
            *((1,) * (output.ndim - 2)),
            bias_noise.shape[-1],
        )
        output = output + sigma * bias_noise.view(bias_shape)
    return output


def _population_embedding(
    token_ids: Tensor,
    embedding: nn.Embedding,
    name: str,
    noise: SignedPopulationNoise,
    sigma: float,
    *,
    candidate_inputs: bool,
) -> Tensor:
    factors = noise.matrices[f"{name}.weight"]
    if candidate_inputs:
        if token_ids.shape[0] != noise.population_size:
            raise ValueError(
                "candidate token inputs have the wrong leading dimension"
            )
        base = embedding(token_ids)
        selected_left = factors.left.gather(1, token_ids).unsqueeze(-1)
        return base + sigma * selected_left * factors.right[:, None, :]
    base = embedding(token_ids).unsqueeze(0).expand(
        noise.population_size,
        -1,
        -1,
        -1,
    )
    selected_left = factors.left[:, token_ids].unsqueeze(-1)
    return base + sigma * selected_left * factors.right[:, None, None, :]


def _population_layer_norm(
    inputs: Tensor,
    layer: nn.LayerNorm,
    name: str,
    noise: SignedPopulationNoise,
    sigma: float,
) -> Tensor:
    normalized = F.layer_norm(
        inputs,
        layer.normalized_shape,
        weight=None,
        bias=None,
        eps=layer.eps,
    )
    weight = layer.weight.unsqueeze(0) + sigma * noise.vectors[
        f"{name}.weight"
    ]
    bias = layer.bias.unsqueeze(0) + sigma * noise.vectors[f"{name}.bias"]
    broadcast_shape = (
        noise.population_size,
        *((1,) * (normalized.ndim - 2)),
        weight.shape[-1],
    )
    return (
        normalized * weight.view(broadcast_shape)
        + bias.view(broadcast_shape)
    )


def _population_attention(
    hidden: Tensor,
    model: HardAttentionPointerTransformer,
    layer_index: int,
    noise: SignedPopulationNoise,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    block = model.encoder.blocks[layer_index]
    prefix = f"encoder.blocks.{layer_index}"
    normalized = _population_layer_norm(
        hidden,
        block.attention_norm,
        f"{prefix}.attention_norm",
        noise,
        sigma,
    )
    query, key, value = _population_linear(
        normalized,
        block.attention.qkv,
        f"{prefix}.attention.qkv",
        noise,
        sigma,
        population_input=True,
    ).chunk(3, dim=-1)
    population_size = query.shape[0]
    sequence_length, model_dim = query.shape[-2:]
    sample_shape = query.shape[1:-2]
    head_count = block.attention.n_heads
    head_dim = model_dim // head_count

    def split_heads(tensor: Tensor) -> Tensor:
        return tensor.view(
            population_size,
            *sample_shape,
            sequence_length,
            head_count,
            head_dim,
        ).transpose(-3, -2)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    causal_mask = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=hidden.device,
    ).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    selected = scores.argmax(dim=-1)
    if model.current_top_k == 1:
        if model.config.sample_sparse_attention:
            selected = sample_top_k_indices(
                scores,
                1,
                share_antithetic_pairs=True,
            ).squeeze(-1)
        attended = torch.gather(
            value,
            dim=-2,
            index=selected.unsqueeze(-1).expand(
                *selected.shape,
                head_dim,
            ),
        )
    elif model.current_top_k is None:
        attended = scores.softmax(dim=-1) @ value
    else:
        selected_count = min(model.current_top_k, sequence_length)
        if model.config.sample_sparse_attention:
            selected_top_k = sample_top_k_indices(
                scores,
                selected_count,
                share_antithetic_pairs=True,
            )
        else:
            selected_top_k = scores.topk(
                selected_count,
                dim=-1,
            ).indices
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(
            -1,
            selected_top_k,
            scores.gather(-1, selected_top_k),
        )
        attended = sparse_scores.softmax(dim=-1) @ value
    if model.current_active_heads < head_count:
        head_mask = torch.arange(
            head_count,
            device=attended.device,
        ) < model.current_active_heads
        attended = attended * head_mask.view(
            *((1,) * (attended.ndim - 3)),
            head_count,
            1,
            1,
        )
    attended = attended.transpose(-3, -2).reshape(
        population_size,
        *sample_shape,
        sequence_length,
        model_dim,
    )
    return (
        _population_linear(
            attended,
            block.attention.output,
            f"{prefix}.attention.output",
            noise,
            sigma,
            population_input=True,
        ),
        selected,
    )


@torch.no_grad()
def population_forward(
    model: HardAttentionPointerTransformer,
    prompt_ids: Tensor,
    offsets: Tensor,
    noise: SignedPopulationNoise,
    sigma: float,
    *,
    candidate_inputs: bool = False,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Evaluate rank-one candidates without materializing candidate weights."""

    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    content = _population_embedding(
        prompt_ids,
        model.encoder.token_embedding,
        "encoder.token_embedding",
        noise,
        sigma,
        candidate_inputs=candidate_inputs,
    )
    if model.encoder.number_projection is not None:
        is_value = (prompt_ids >= VALUE_OFFSET) & (
            prompt_ids < VALUE_OFFSET + model.config.symbol_count
        )
        values = prompt_ids.to(content.dtype) - VALUE_OFFSET
        values = (
            2.0 * values / (model.config.symbol_count - 1) - 1.0
        )
        values = torch.where(is_value, values, torch.zeros_like(values))
        content = content + _population_linear(
            values.unsqueeze(-1),
            model.encoder.number_projection,
            "encoder.number_projection",
            noise,
            sigma,
            population_input=candidate_inputs,
        )
    positions = model.position_embeddings(prompt_ids, offsets)
    if not candidate_inputs:
        positions = positions.unsqueeze(0).expand(
            noise.population_size,
            -1,
            -1,
            -1,
        )
    hidden = torch.cat(
        (
            content,
            positions,
        ),
        dim=-1,
    )
    routes = []
    for layer_index, block in enumerate(model.encoder.blocks):
        prefix = f"encoder.blocks.{layer_index}"
        attended, selected = _population_attention(
            hidden,
            model,
            layer_index,
            noise,
            sigma,
        )
        hidden = hidden + attended
        normalized = _population_layer_norm(
            hidden,
            block.ffn_norm,
            f"{prefix}.ffn_norm",
            noise,
            sigma,
        )
        gate, value = _population_linear(
            normalized,
            block.ffn.input,
            f"{prefix}.ffn.input",
            noise,
            sigma,
            population_input=True,
        ).chunk(2, dim=-1)
        hidden = hidden + _population_linear(
            F.silu(gate) * value,
            block.ffn.output,
            f"{prefix}.ffn.output",
            noise,
            sigma,
            population_input=True,
        )
        routes.append(selected)
    hidden = _population_layer_norm(
        hidden,
        model.encoder.final_norm,
        "encoder.final_norm",
        noise,
        sigma,
    )
    logits = _population_linear(
        hidden[..., -1, :],
        model.output,
        "output",
        noise,
        sigma,
        population_input=True,
    )
    return logits, tuple(routes)


def pointer_targets(batch: PointerNextBatch) -> Tensor:
    row_indices = torch.arange(
        batch.values.shape[0],
        device=batch.values.device,
    )
    return batch.values[row_indices, batch.pointers + 1]


def shape_fitness(losses: Tensor, mode: str) -> Tensor:
    if losses.ndim != 1 or losses.numel() < 2:
        raise ValueError("losses must contain a candidate population")
    rewards = -losses.float()
    if mode == "zscore":
        centered = rewards - rewards.mean()
        return centered / torch.sqrt(
            rewards.var(unbiased=False) + 1e-8
        )
    if mode == "centered_rank":
        ranks = rewards.argsort().argsort().to(rewards.dtype)
        return ranks / (len(rewards) - 1) - 0.5
    raise ValueError("unknown fitness shaping")


def estimate_reward_gradients(
    noise: AntitheticRankOneNoise,
    fitness: Tensor,
) -> dict[str, Tensor]:
    if fitness.shape != (noise.population_size,):
        raise ValueError("fitness must have one value per candidate")
    pair_fitness = fitness[: noise.pair_count] - fitness[noise.pair_count :]
    gradients = {}
    for name, factors in noise.matrices.items():
        gradients[name] = torch.einsum(
            "p,po,pi->oi",
            pair_fitness.to(factors.left.dtype),
            factors.left,
            factors.right,
        ) / noise.population_size
    for name, values in noise.vectors.items():
        broadcast = (noise.pair_count,) + (1,) * (values.ndim - 1)
        gradients[name] = (
            pair_fitness.to(values.dtype).reshape(broadcast) * values
        ).sum(dim=0) / noise.population_size
    return gradients


def estimate_elite_centroid_directions(
    noise: AntitheticRankOneNoise,
    losses: Tensor,
    *,
    elite_count: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Average the best signs from the top unique antithetic directions."""

    if losses.shape != (noise.population_size,):
        raise ValueError("losses must have one value per candidate")
    if not 1 <= elite_count <= noise.pair_count:
        raise ValueError("elite count must fit the antithetic directions")
    positive_losses = losses[: noise.pair_count]
    negative_losses = losses[noise.pair_count :]
    prefer_positive = positive_losses <= negative_losses
    preferred_losses = torch.minimum(positive_losses, negative_losses)
    elite_pairs = preferred_losses.topk(
        elite_count,
        largest=False,
    ).indices
    elite_signs = torch.where(
        prefer_positive[elite_pairs],
        torch.ones_like(preferred_losses[elite_pairs]),
        -torch.ones_like(preferred_losses[elite_pairs]),
    )
    directions = {}
    for name, factors in noise.matrices.items():
        directions[name] = torch.einsum(
            "p,po,pi->oi",
            elite_signs.to(factors.left.dtype),
            factors.left[elite_pairs],
            factors.right[elite_pairs],
        ) / elite_count
    for name, values in noise.vectors.items():
        broadcast = (elite_count,) + (1,) * (values.ndim - 1)
        directions[name] = (
            elite_signs.to(values.dtype).reshape(broadcast)
            * values[elite_pairs]
        ).mean(dim=0)
    selected_candidates = torch.where(
        prefer_positive[elite_pairs],
        elite_pairs,
        elite_pairs + noise.pair_count,
    )
    return directions, selected_candidates


def assign_maximization_gradients(
    model: nn.Module,
    reward_gradients: dict[str, Tensor],
) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(reward_gradients):
        raise ValueError("gradient estimate does not match model parameters")
    for name, parameter in parameters.items():
        parameter.grad = -reward_gradients[name]


def tensor_collection_rms(tensors: list[Tensor]) -> float:
    squared_sum = sum(
        tensor.detach().float().square().sum() for tensor in tensors
    )
    element_count = sum(tensor.numel() for tensor in tensors)
    return float(torch.sqrt(squared_sum / element_count))


@dataclass(frozen=True)
class PopulationMetrics:
    losses: Tensor
    accuracies: Tensor
    route_disagreement_fraction: float
    antithetic_loss_gap_abs_mean: float


def distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def distributed_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_primary_process() -> bool:
    return distributed_rank() == 0


def gather_population(values: Tensor) -> Tensor:
    if not dist.is_initialized():
        return values
    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    return torch.cat(gathered)


def average_gradients_across_workers(
    gradients: dict[str, Tensor],
) -> None:
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for name, value in gradients.items():
        reduced = value.contiguous()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        gradients[name] = reduced.div_(world_size)


@torch.no_grad()
def evaluate_population(
    model: HardAttentionPointerTransformer,
    batch: PointerNextBatch,
    offsets: Tensor,
    noise: AntitheticRankOneNoise,
    *,
    sigma: float,
    population_chunk_size: int,
    data_mode: str = "cartesian",
    precision: str = "float32",
) -> PopulationMetrics:
    if population_chunk_size % 2:
        raise ValueError("population chunk size must be even")
    if data_mode not in {"cartesian", "grouped"}:
        raise ValueError("population data mode must be cartesian or grouped")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("population precision must be float32 or bfloat16")
    targets = pointer_targets(batch)
    if data_mode == "grouped" and noise.pair_count % len(targets):
        raise ValueError(
            "antithetic pairs must divide evenly across grouped examples"
        )
    device_type = next(model.parameters()).device.type
    if (
        precision == "bfloat16"
        and device_type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("CUDA device does not support bfloat16")
    pair_chunk_size = population_chunk_size // 2
    positive_losses = []
    negative_losses = []
    positive_accuracies = []
    negative_accuracies = []
    disagreement_count = 0
    route_count = 0
    for start in range(0, noise.pair_count, pair_chunk_size):
        end = min(start + pair_chunk_size, noise.pair_count)
        signed_noise = noise.pair_chunk(start, end)
        local_pair_count = end - start
        if data_mode == "grouped":
            pair_examples = (
                torch.arange(start, end, device=targets.device)
                % targets.shape[0]
            )
            signed_examples = torch.cat(
                (pair_examples, pair_examples),
                dim=0,
            )
            prompt_ids = batch.prompt_ids[signed_examples]
            chunk_offsets = offsets[signed_examples]
            chunk_targets = targets[signed_examples]
            candidate_inputs = True
        else:
            prompt_ids = batch.prompt_ids
            chunk_offsets = offsets
            chunk_targets = targets.unsqueeze(0).expand(
                2 * local_pair_count,
                -1,
            )
            candidate_inputs = False
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            logits, routes = population_forward(
                model,
                prompt_ids,
                chunk_offsets,
                signed_noise,
                sigma,
                candidate_inputs=candidate_inputs,
            )
        if data_mode == "grouped":
            losses = F.cross_entropy(
                logits.float(),
                chunk_targets,
                reduction="none",
            )
            accuracies = logits.argmax(dim=-1).eq(chunk_targets).float()
        else:
            losses = F.cross_entropy(
                logits.float().transpose(1, 2),
                chunk_targets,
                reduction="none",
            ).mean(dim=1)
            accuracies = logits.argmax(dim=-1).eq(
                chunk_targets
            ).float().mean(dim=1)
        positive_losses.append(losses[:local_pair_count])
        negative_losses.append(losses[local_pair_count:])
        positive_accuracies.append(accuracies[:local_pair_count])
        negative_accuracies.append(accuracies[local_pair_count:])
        for route in routes:
            positive = route[:local_pair_count]
            negative = route[local_pair_count:]
            disagreement_count += int(positive.ne(negative).sum())
            route_count += positive.numel()
    return PopulationMetrics(
        losses=torch.cat((*positive_losses, *negative_losses)),
        accuracies=torch.cat(
            (*positive_accuracies, *negative_accuracies)
        ),
        route_disagreement_fraction=disagreement_count / route_count,
        antithetic_loss_gap_abs_mean=float(
            (
                torch.cat(positive_losses)
                - torch.cat(negative_losses)
            ).abs().mean()
        ),
    )


def make_model(
    config: HardAttentionEggrollConfig,
    *,
    device: torch.device,
) -> HardAttentionPointerTransformer:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    return HardAttentionPointerTransformer(config).to(device)


def make_optimizer(
    model: HardAttentionPointerTransformer,
    config: HardAttentionEggrollConfig,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def make_training_batch(
    config: HardAttentionEggrollConfig,
    *,
    vocabulary: PointerNextVocabulary,
    generator: torch.Generator,
    device: torch.device,
    max_length: int | None = None,
) -> tuple[PointerNextBatch, Tensor]:
    resolved_max_length = (
        config.train_max_length if max_length is None else max_length
    )
    if not config.train_min_length <= resolved_max_length <= (
        config.train_max_length
    ):
        raise ValueError("training maximum is outside the configured range")
    length = int(
        torch.randint(
            config.train_min_length,
            resolved_max_length + 1,
            (),
            generator=generator,
        )
    )
    batch = make_pointer_next_batch(
        config.batch_size,
        length,
        generator=generator,
        vocabulary=vocabulary,
        device=device,
    )
    offsets = sample_position_offsets(
        config.batch_size,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=device,
    )
    return batch, offsets


def make_evaluation_data(
    config: HardAttentionEggrollConfig,
    *,
    vocabulary: PointerNextVocabulary,
    device: torch.device,
) -> dict[int, tuple[PointerNextBatch, Tensor]]:
    generator = torch.Generator().manual_seed(config.seed + 20_000)
    return {
        length: (
            make_pointer_next_batch(
                (
                    config.long_eval_examples
                    if length >= config.long_eval_min_length
                    else config.eval_examples
                ),
                length,
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            ),
            sample_position_offsets(
                (
                    config.long_eval_examples
                    if length >= config.long_eval_min_length
                    else config.eval_examples
                ),
                minimum=config.position_offset_min,
                maximum=config.position_offset_max,
                generator=generator,
                device=device,
            ),
        )
        for length in config.eval_lengths
    }


def evaluation_batch_size(
    *,
    configured_batch_size: int,
    attention_element_budget: int,
    heads: int,
    prompt_length: int,
) -> int:
    """Bound the number of materialized attention-score elements per batch."""

    attention_limited = attention_element_budget // (
        heads * prompt_length * prompt_length
    )
    return min(configured_batch_size, max(1, attention_limited))


@torch.inference_mode()
def evaluate_pointer_batch(
    model: HardAttentionPointerTransformer,
    batch: PointerNextBatch,
    offsets: Tensor,
    *,
    eval_batch_size: int,
    eval_attention_element_budget: int,
    heads: int,
) -> tuple[float, float, int]:
    targets = pointer_targets(batch)
    resolved_batch_size = evaluation_batch_size(
        configured_batch_size=eval_batch_size,
        attention_element_budget=eval_attention_element_budget,
        heads=heads,
        prompt_length=batch.prompt_length,
    )
    loss_sum = 0.0
    correct = 0
    for start in range(0, targets.shape[0], resolved_batch_size):
        end = min(start + resolved_batch_size, targets.shape[0])
        logits = model(
            batch.prompt_ids[start:end],
            offsets=offsets[start:end],
        )
        loss_sum += float(
            F.cross_entropy(
                logits,
                targets[start:end],
                reduction="sum",
            )
        )
        correct += int(
            logits.argmax(dim=-1).eq(targets[start:end]).sum()
        )
    return (
        loss_sum / targets.shape[0],
        correct / targets.shape[0],
        resolved_batch_size,
    )


@torch.inference_mode()
def evaluate_model(
    model: HardAttentionPointerTransformer,
    evaluation_data: dict[int, tuple[PointerNextBatch, Tensor]],
    *,
    train_max_length: int,
    eval_batch_size: int,
    eval_attention_element_budget: int,
    heads: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        summary = {}
        in_domain_accuracies = []
        out_of_domain_accuracies = []
        for length, (batch, offsets) in evaluation_data.items():
            loss, accuracy, resolved_batch_size = evaluate_pointer_batch(
                model,
                batch,
                offsets,
                eval_batch_size=eval_batch_size,
                eval_attention_element_budget=eval_attention_element_budget,
                heads=heads,
            )
            summary[f"eval/length_{length}/loss"] = loss
            summary[f"eval/length_{length}/accuracy"] = accuracy
            summary[f"eval/length_{length}/examples"] = float(
                batch.values.shape[0]
            )
            summary[f"eval/length_{length}/batch_size"] = float(
                resolved_batch_size
            )
            (
                in_domain_accuracies
                if length <= train_max_length
                else out_of_domain_accuracies
            ).append(accuracy)
        summary["eval/in_domain_accuracy_mean"] = sum(
            in_domain_accuracies
        ) / len(in_domain_accuracies)
        if out_of_domain_accuracies:
            summary["eval/out_of_domain_accuracy_mean"] = sum(
                out_of_domain_accuracies
            ) / len(out_of_domain_accuracies)
        return summary
    finally:
        model.train(was_training)


def evaluate_curriculum_probe(
    model: HardAttentionPointerTransformer,
    config: HardAttentionEggrollConfig,
    state: CurriculumState,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[float, float]:
    batch = make_pointer_next_batch(
        config.curriculum_examples,
        state.current_max_length,
        generator=generator,
        vocabulary=model.vocabulary,
        device=device,
    )
    offsets = sample_position_offsets(
        config.curriculum_examples,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=device,
    )
    was_training = model.training
    model.eval()
    try:
        loss, accuracy, _ = evaluate_pointer_batch(
            model,
            batch,
            offsets,
            eval_batch_size=config.eval_batch_size,
            eval_attention_element_budget=config.eval_attention_element_budget,
            heads=config.heads,
        )
        return loss, accuracy
    finally:
        model.train(was_training)


def initialize_wandb(
    config: HardAttentionEggrollConfig,
    *,
    run_id: str | None,
) -> Any | None:
    if not config.wandb:
        return None
    import wandb

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=asdict(config),
        id=run_id,
        resume="allow" if run_id is not None else None,
    )


def save_checkpoint(
    path: Path,
    *,
    config: HardAttentionEggrollConfig,
    model: HardAttentionPointerTransformer,
    optimizer: torch.optim.Optimizer,
    generation: int,
    data_generator: torch.Generator,
    noise_generator: torch.Generator,
    curriculum_generator: torch.Generator,
    curriculum_state: CurriculumState,
    noise_generator_states: list[Tensor],
    attention_sampling_rng_states: list[Tensor],
    wandb_run_id: str | None,
) -> None:
    torch.save(
        {
            "experiment": "hard_attention_forward_eggroll",
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generation": generation,
            "data_generator_state": data_generator.get_state(),
            "noise_generator_state": noise_generator.get_state(),
            "noise_generator_states": noise_generator_states,
            "attention_sampling_rng_states": (
                attention_sampling_rng_states
            ),
            "curriculum_generator_state": (
                curriculum_generator.get_state()
            ),
            "curriculum_state": asdict(curriculum_state),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def run(config: HardAttentionEggrollConfig) -> Path:
    device = resolve_device(config.device)
    world_size = distributed_world_size()
    rank = distributed_rank()
    if config.population_size % world_size:
        raise ValueError(
            "global population must divide evenly across distributed workers"
        )
    local_population_size = config.population_size // world_size
    if local_population_size < 2 or local_population_size % 2:
        raise ValueError(
            "each distributed worker needs a positive even local population"
        )
    if config.population_chunk_size > local_population_size:
        raise ValueError(
            "population chunk cannot exceed the per-worker population"
        )
    if world_size > 1 and config.update_rule != "paper_standardized":
        raise ValueError(
            "distributed execution currently supports paper_standardized only"
        )
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_primary_process():
        (output_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )
    metrics_path = output_dir / "metrics.jsonl"
    model = make_model(config, device=device)
    optimizer = make_optimizer(model, config)
    attention_sampling_seed = config.seed + 4_000 + rank
    if device.type == "cuda":
        torch.cuda.manual_seed_all(attention_sampling_seed)
    else:
        torch.manual_seed(attention_sampling_seed)
    data_generator = torch.Generator().manual_seed(config.seed + 1_000)
    noise_generator = torch.Generator(device=device).manual_seed(
        config.seed + 2_000 + rank
    )
    curriculum_generator = torch.Generator().manual_seed(
        config.seed + 3_000
    )
    curriculum_state = initialize_curriculum_state(config)
    model.set_attention_top_k(curriculum_state.attention_top_k)
    model.set_active_heads(curriculum_state.active_heads)
    start_generation = 1
    checkpoint = None
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        if checkpoint.get("experiment") != "hard_attention_forward_eggroll":
            raise ValueError("resume checkpoint belongs to another experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
        noise_generator_states = checkpoint.get("noise_generator_states")
        if noise_generator_states is not None:
            if len(noise_generator_states) != world_size:
                raise ValueError(
                    "checkpoint noise RNG states do not match world size"
                )
            noise_generator.set_state(noise_generator_states[rank].cpu())
        elif world_size == 1:
            noise_generator.set_state(
                checkpoint["noise_generator_state"].cpu()
            )
        else:
            raise ValueError(
                "distributed resume requires per-worker noise RNG states"
            )
        if "curriculum_generator_state" in checkpoint:
            curriculum_generator.set_state(
                checkpoint["curriculum_generator_state"].cpu()
            )
        if "curriculum_state" in checkpoint:
            curriculum_state = restore_curriculum_state(
                checkpoint["curriculum_state"],
                config,
            )
        attention_sampling_rng_states = checkpoint.get(
            "attention_sampling_rng_states"
        )
        if attention_sampling_rng_states is not None:
            if len(attention_sampling_rng_states) != world_size:
                raise ValueError(
                    "checkpoint attention RNG states do not match world size"
                )
            if device.type == "cuda":
                torch.cuda.set_rng_state(
                    attention_sampling_rng_states[rank].cpu(),
                    device=device,
                )
            else:
                torch.set_rng_state(
                    attention_sampling_rng_states[rank].cpu()
                )
        model.set_attention_top_k(curriculum_state.attention_top_k)
        model.set_active_heads(curriculum_state.active_heads)
        start_generation = int(checkpoint["generation"]) + 1

    evaluation_data = make_evaluation_data(
        config,
        vocabulary=model.vocabulary,
        device=device,
    )
    resumed_wandb_run_id = (
        config.wandb_run_id
        or (
            checkpoint.get("wandb_run_id")
            if checkpoint is not None
            else None
        )
    )
    wandb_run = (
        initialize_wandb(config, run_id=resumed_wandb_run_id)
        if is_primary_process()
        else None
    )
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}", flush=True)
    if start_generation == 1 and is_primary_process():
        initial = {
            "generation": 0.0,
            "curriculum/enabled": float(config.curriculum),
            "attention/sampled_top_k": float(
                config.sample_sparse_attention
            ),
            "attention/eval_argmax": 1.0,
            "curriculum/current_max_length": float(
                curriculum_state.current_max_length
            ),
            "curriculum/attention_top_k": float(
                curriculum_state.attention_top_k or 0
            ),
            "curriculum/active_heads": float(
                curriculum_state.active_heads
            ),
            "curriculum/dense_attention": float(
                curriculum_state.attention_top_k is None
            ),
            "curriculum/complete": float(
                curriculum_is_complete(curriculum_state, config)
            ),
        }
        initial.update(
            evaluate_model(
                model,
                evaluation_data,
                train_max_length=config.train_max_length,
                eval_batch_size=config.eval_batch_size,
                eval_attention_element_budget=(
                    config.eval_attention_element_budget
                ),
                heads=config.heads,
            )
        )
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(initial) + "\n")
        if wandb_run is not None:
            wandb_run.log(initial, step=0)
    if dist.is_initialized():
        dist.barrier()

    started_at = time.monotonic()
    for generation in range(start_generation, config.generations + 1):
        batch, offsets = make_training_batch(
            config,
            vocabulary=model.vocabulary,
            generator=data_generator,
            device=device,
            max_length=curriculum_state.current_max_length,
        )
        noise = sample_antithetic_rank_one_noise(
            model,
            local_population_size,
            generator=noise_generator,
        )
        local_population = evaluate_population(
            model,
            batch,
            offsets,
            noise,
            sigma=config.sigma,
            population_chunk_size=config.population_chunk_size,
            data_mode=config.population_data_mode,
            precision=config.population_precision,
        )
        global_losses = gather_population(local_population.losses)
        global_accuracies = gather_population(local_population.accuracies)
        fitness = shape_fitness(
            global_losses,
            config.fitness_shaping,
        )
        fitness_start = rank * local_population_size
        local_fitness = fitness[
            fitness_start : fitness_start + local_population_size
        ]
        route_disagreement = torch.tensor(
            local_population.route_disagreement_fraction,
            device=device,
        )
        antithetic_loss_gap = torch.tensor(
            local_population.antithetic_loss_gap_abs_mean,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(route_disagreement, op=dist.ReduceOp.SUM)
            route_disagreement.div_(world_size)
            dist.all_reduce(antithetic_loss_gap, op=dist.ReduceOp.SUM)
            antithetic_loss_gap.div_(world_size)
        population = PopulationMetrics(
            losses=global_losses,
            accuracies=global_accuracies,
            route_disagreement_fraction=float(route_disagreement),
            antithetic_loss_gap_abs_mean=float(antithetic_loss_gap),
        )
        selected_elites = None
        if config.update_rule == "paper_standardized":
            reward_gradients = estimate_reward_gradients(
                noise,
                local_fitness,
            )
            average_gradients_across_workers(reward_gradients)
            gradient_scale = (
                config.sigma * math.sqrt(config.population_size)
            )
        else:
            reward_gradients, selected_elites = (
                estimate_elite_centroid_directions(
                    noise,
                    population.losses,
                    elite_count=config.elite_count,
                )
            )
            gradient_scale = config.sigma
        reward_gradients = {
            name: gradient * gradient_scale
            for name, gradient in reward_gradients.items()
        }
        curriculum_check = curriculum_check_due(
            config,
            curriculum_state,
            generation=generation,
            batch_length=batch.length,
        )
        report_generation = (
            generation % config.log_interval == 0
            or generation % config.eval_interval == 0
            or curriculum_check
            or generation == config.generations
        )
        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        parameters_before_update = (
            [
                parameter.detach().clone()
                for parameter in trainable_parameters
            ]
            if report_generation
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        assign_maximization_gradients(model, reward_gradients)
        optimizer.step()

        if report_generation:
            assert parameters_before_update is not None
            parameter_rms = tensor_collection_rms(trainable_parameters)
            update_rms = tensor_collection_rms(
                [
                    parameter.detach() - previous
                    for parameter, previous in zip(
                        trainable_parameters,
                        parameters_before_update,
                    )
                ]
            )
            with torch.inference_mode():
                center_logits = model(batch.prompt_ids, offsets=offsets)
                targets = pointer_targets(batch)
                predictions = center_logits.argmax(dim=-1)
                counts = torch.bincount(
                    predictions,
                    minlength=config.symbol_count,
                )
                summary = {
                    "generation": float(generation),
                    "train/length": float(batch.length),
                    "train/center_loss_after_update": float(
                        F.cross_entropy(center_logits, targets)
                    ),
                    "train/center_accuracy_after_update": float(
                        predictions.eq(targets).float().mean()
                    ),
                    "train/unique_predictions": float(
                        counts.count_nonzero()
                    ),
                    "train/prediction_mode_fraction": float(
                        counts.max() / counts.sum()
                    ),
                    "population/loss_mean": float(
                        population.losses.mean()
                    ),
                    "population/loss_std": float(
                        population.losses.std(unbiased=False)
                    ),
                    "population/accuracy_mean": float(
                        population.accuracies.mean()
                    ),
                    "population/best_accuracy": float(
                        population.accuracies.max()
                    ),
                    "population/grouped_data": float(
                        config.population_data_mode == "grouped"
                    ),
                    "population/unique_examples": float(config.batch_size),
                    "population/candidates_per_example": float(
                        config.population_size / config.batch_size
                        if config.population_data_mode == "grouped"
                        else config.population_size
                    ),
                    "population/candidate_example_evaluations": float(
                        config.population_size
                        if config.population_data_mode == "grouped"
                        else config.population_size * config.batch_size
                    ),
                    "population/distributed_workers": float(world_size),
                    "population/local_population_size": float(
                        local_population_size
                    ),
                    "population/bfloat16_forward": float(
                        config.population_precision == "bfloat16"
                    ),
                    "population/antithetic_loss_gap_abs_mean": float(
                        population.antithetic_loss_gap_abs_mean
                    ),
                    "routing/antithetic_disagreement_fraction": (
                        population.route_disagreement_fraction
                    ),
                    "optimization/fitness_std": float(
                        fitness.std(unbiased=False)
                    ),
                    "optimization/update_rule_paper_standardized": float(
                        config.update_rule == "paper_standardized"
                    ),
                    "optimization/update_rule_elite_centroid": float(
                        config.update_rule == "elite_centroid"
                    ),
                    "optimization/elite_count": float(
                        config.elite_count
                        if selected_elites is not None
                        else 0
                    ),
                    "optimization/elite_positive_fraction": float(
                        (
                            selected_elites.lt(noise.pair_count)
                            .float()
                            .mean()
                        )
                        if selected_elites is not None
                        else 0
                    ),
                    "optimization/gradient_scale": gradient_scale,
                    "optimization/reward_gradient_rms": (
                        tensor_collection_rms(
                            list(reward_gradients.values())
                        )
                    ),
                    "optimization/parameter_update_rms": update_rms,
                    "optimization/parameter_rms": parameter_rms,
                    "optimization/update_to_parameter_rms_ratio": (
                        update_rms / max(parameter_rms, 1e-12)
                    ),
                    "optimization/sigma": config.sigma,
                    "optimization/learning_rate": (
                        optimizer.param_groups[0]["lr"]
                    ),
                    "curriculum/enabled": float(config.curriculum),
                    "attention/sampled_top_k": float(
                        config.sample_sparse_attention
                    ),
                    "attention/eval_argmax": 1.0,
                    "curriculum/current_max_length": float(
                        curriculum_state.current_max_length
                    ),
                    "curriculum/attention_top_k": float(
                        curriculum_state.attention_top_k or 0
                    ),
                    "curriculum/active_heads": float(
                        curriculum_state.active_heads
                    ),
                    "curriculum/dense_attention": float(
                        curriculum_state.attention_top_k is None
                    ),
                    "curriculum/success_streak": float(
                        curriculum_state.success_streak
                    ),
                    "curriculum/promotion_count": float(
                        curriculum_state.promotion_count
                    ),
                    "curriculum/complete": float(
                        curriculum_is_complete(curriculum_state, config)
                    ),
                    "timing/generations_per_second": generation
                    / max(time.monotonic() - started_at, 1e-9),
                }
            if (
                is_primary_process()
                and (
                    generation % config.eval_interval == 0
                    or generation == config.generations
                )
            ):
                summary.update(
                    evaluate_model(
                        model,
                        evaluation_data,
                        train_max_length=config.train_max_length,
                        eval_batch_size=config.eval_batch_size,
                        eval_attention_element_budget=(
                            config.eval_attention_element_budget
                        ),
                        heads=config.heads,
                    )
                )
                print(
                    f"generation={generation} "
                    f"train={summary['train/center_accuracy_after_update']:.3f} "
                    f"in_domain={summary['eval/in_domain_accuracy_mean']:.3f} "
                    f"ood={summary.get('eval/out_of_domain_accuracy_mean', 0):.3f}",
                    flush=True,
                )
            if curriculum_check:
                uses_training_batch = (
                    config.curriculum_progress_mode == "training_streak"
                )
                if uses_training_batch:
                    criterion_loss = summary[
                        "train/center_loss_after_update"
                    ]
                    criterion_accuracy = summary[
                        "train/center_accuracy_after_update"
                    ]
                else:
                    criterion_loss, criterion_accuracy = (
                        evaluate_curriculum_probe(
                            model,
                            config,
                            curriculum_state,
                            generator=curriculum_generator,
                            device=device,
                        )
                    )
                promotion = update_curriculum(
                    curriculum_state,
                    config,
                    criterion_accuracy=criterion_accuracy,
                )
                model.set_attention_top_k(
                    curriculum_state.attention_top_k
                )
                model.set_active_heads(curriculum_state.active_heads)
                summary.update(
                    {
                        "curriculum/criterion_loss": criterion_loss,
                        "curriculum/criterion_accuracy": criterion_accuracy,
                        "curriculum/criterion_is_training_batch": float(
                            uses_training_batch
                        ),
                        "curriculum/promoted": float(
                            promotion is not None
                        ),
                        "curriculum/promoted_length": float(
                            promotion == "length"
                        ),
                        "curriculum/started_sparsity": float(
                            promotion == "start_sparsity"
                        ),
                        "curriculum/increased_sparsity": float(
                            promotion == "increase_sparsity"
                        ),
                        "curriculum/pruned_head": float(
                            promotion == "prune_head"
                        ),
                        "curriculum/current_max_length": float(
                            curriculum_state.current_max_length
                        ),
                        "curriculum/attention_top_k": float(
                            curriculum_state.attention_top_k or 0
                        ),
                        "curriculum/active_heads": float(
                            curriculum_state.active_heads
                        ),
                        "curriculum/dense_attention": float(
                            curriculum_state.attention_top_k is None
                        ),
                        "curriculum/success_streak": float(
                            curriculum_state.success_streak
                        ),
                        "curriculum/promotion_count": float(
                            curriculum_state.promotion_count
                        ),
                        "curriculum/complete": float(
                            curriculum_is_complete(
                                curriculum_state,
                                config,
                            )
                        ),
                    }
                )
                if not uses_training_batch:
                    summary.update(
                        {
                            "curriculum/probe_loss": criterion_loss,
                            "curriculum/probe_accuracy": criterion_accuracy,
                        }
                    )
            if is_primary_process():
                with metrics_path.open("a") as metrics_file:
                    metrics_file.write(json.dumps(summary) + "\n")
                if wandb_run is not None:
                    wandb_run.log(summary, step=generation)
            if dist.is_initialized():
                dist.barrier()

        if (
            generation % config.checkpoint_interval == 0
            or generation == config.generations
        ):
            local_noise_state = noise_generator.get_state().to(device)
            local_attention_rng_state = (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else torch.get_rng_state()
            ).to(device)
            if dist.is_initialized():
                gathered_noise_states = [
                    torch.empty_like(local_noise_state)
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_noise_states, local_noise_state)
                noise_generator_states = [
                    state.cpu() for state in gathered_noise_states
                ]
                gathered_attention_rng_states = [
                    torch.empty_like(local_attention_rng_state)
                    for _ in range(world_size)
                ]
                dist.all_gather(
                    gathered_attention_rng_states,
                    local_attention_rng_state,
                )
                attention_sampling_rng_states = [
                    state.cpu() for state in gathered_attention_rng_states
                ]
            else:
                noise_generator_states = [local_noise_state.cpu()]
                attention_sampling_rng_states = [
                    local_attention_rng_state.cpu()
                ]
            if is_primary_process():
                checkpoint_arguments = {
                    "config": config,
                    "model": model,
                    "optimizer": optimizer,
                    "generation": generation,
                    "data_generator": data_generator,
                    "noise_generator": noise_generator,
                    "curriculum_generator": curriculum_generator,
                    "curriculum_state": curriculum_state,
                    "noise_generator_states": noise_generator_states,
                    "attention_sampling_rng_states": (
                        attention_sampling_rng_states
                    ),
                    "wandb_run_id": (
                        wandb_run.id if wandb_run is not None else None
                    ),
                }
                save_checkpoint(
                    output_dir / f"checkpoint_{generation:06d}.pt",
                    **checkpoint_arguments,
                )
                save_checkpoint(
                    output_dir / "latest.pt",
                    **checkpoint_arguments,
                )
            if dist.is_initialized():
                dist.barrier()
    if wandb_run is not None and is_primary_process():
        wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
    return output_dir


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=HardAttentionEggrollConfig.run_name)
    parser.add_argument("--output-dir", default=HardAttentionEggrollConfig.output_dir)
    parser.add_argument(
        "--generations",
        type=int,
        default=HardAttentionEggrollConfig.generations,
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=HardAttentionEggrollConfig.population_size,
    )
    parser.add_argument(
        "--population-chunk-size",
        type=int,
        default=HardAttentionEggrollConfig.population_chunk_size,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=HardAttentionEggrollConfig.batch_size,
    )
    parser.add_argument(
        "--population-data-mode",
        choices=("cartesian", "grouped"),
        default=HardAttentionEggrollConfig.population_data_mode,
    )
    parser.add_argument(
        "--population-precision",
        choices=("float32", "bfloat16"),
        default=HardAttentionEggrollConfig.population_precision,
    )
    parser.add_argument(
        "--train-min-length",
        type=int,
        default=HardAttentionEggrollConfig.train_min_length,
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=HardAttentionEggrollConfig.train_max_length,
    )
    parser.add_argument(
        "--eval-lengths",
        type=parse_integer_tuple,
        default=HardAttentionEggrollConfig.eval_lengths,
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=HardAttentionEggrollConfig.eval_examples,
    )
    parser.add_argument(
        "--long-eval-examples",
        type=int,
        default=HardAttentionEggrollConfig.long_eval_examples,
    )
    parser.add_argument(
        "--long-eval-min-length",
        type=int,
        default=HardAttentionEggrollConfig.long_eval_min_length,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=HardAttentionEggrollConfig.eval_batch_size,
    )
    parser.add_argument(
        "--eval-attention-element-budget",
        type=int,
        default=HardAttentionEggrollConfig.eval_attention_element_budget,
    )
    parser.add_argument("--symbol-count", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--attention-mode",
        choices=("top1", "dense"),
        default="top1",
    )
    parser.add_argument(
        "--sample-sparse-attention",
        action="store_true",
        help="sample top-k source positions instead of taking the largest scores",
    )
    parser.add_argument(
        "--position-moduli",
        type=parse_integer_tuple,
        default=DEFAULT_POSITION_MODULI,
    )
    parser.add_argument("--position-offset-min", type=int, default=-1_000_000)
    parser.add_argument("--position-offset-max", type=int, default=1_000_000)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--fitness-shaping",
        choices=("zscore", "centered_rank"),
        default="zscore",
    )
    parser.add_argument(
        "--update-rule",
        choices=("paper_standardized", "elite_centroid"),
        default="paper_standardized",
    )
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--curriculum-progress-mode",
        choices=("probe", "training_streak"),
        default=HardAttentionEggrollConfig.curriculum_progress_mode,
    )
    parser.add_argument(
        "--curriculum-accuracy-threshold",
        type=float,
        default=HardAttentionEggrollConfig.curriculum_accuracy_threshold,
    )
    parser.add_argument(
        "--curriculum-success-checks",
        type=int,
        default=HardAttentionEggrollConfig.curriculum_success_checks,
    )
    parser.add_argument(
        "--curriculum-check-interval",
        type=int,
        default=HardAttentionEggrollConfig.curriculum_check_interval,
    )
    parser.add_argument(
        "--curriculum-examples",
        type=int,
        default=HardAttentionEggrollConfig.curriculum_examples,
    )
    parser.add_argument(
        "--curriculum-initial-top-k",
        type=int,
        default=HardAttentionEggrollConfig.curriculum_initial_top_k,
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=HardAttentionEggrollConfig.wandb_project,
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--resume")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
    try:
        config = HardAttentionEggrollConfig(**vars(args))
        output_dir = run(config)
        if is_primary_process():
            print(f"Artifacts: {output_dir}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
