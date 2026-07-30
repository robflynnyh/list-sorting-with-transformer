"""Adam training for the pointer-next task with ASEntmax sparse attention."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .compiled_pointer_compare import (
    DEFAULT_POSITION_MODULI,
    _set_modular_fourier_codebooks,
)
from ..core.data import PointerNextBatch, make_pointer_next_batch
from ..core.evaluate import resolve_device
from ..core.model import DecoderTransformer, ModelConfig, SplitInputDecoderTransformer
from ..core.positions import ModularPositionEmbedding, sample_position_offsets
from ..core.tokens import PointerNextVocabulary


@dataclass(frozen=True)
class SparseAttentionAdamConfig:
    run_name: str = "pointer-next-asentmax-adam-seed7"
    output_dir: str = "artifacts/sparse_attention_adam"
    steps: int = 20_000
    batch_size: int = 256
    train_min_length: int = 2
    train_max_length: int = 20
    eval_lengths: tuple[int, ...] = (2, 5, 10, 20, 40, 100, 400, 1_000, 2_000)
    final_eval_lengths: tuple[int, ...] = (5_000,)
    eval_examples: int = 512
    long_eval_examples: int = 64
    long_eval_min_length: int = 1_000
    final_eval_examples: int = 64
    eval_batch_size: int = 128
    eval_attention_element_budget: int = 128_000_000
    symbol_count: int = 10
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn_multiplier: float = 4.0
    alibi_heads: int = 2
    attention_normalizer: str = "entmax15"
    entmax_alpha: float = 1.5
    scale_delta: float = 1.0
    scale_gamma_range: float = 2.0
    scaling_mode: str = "adaptive"
    architecture: str = "standard"
    input_position_mode: str = "modular"
    value_input_mode: str = "embedding_plus_scalar"
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    learning_rate: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    minimum_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    optimizer_name: str = "adam"
    log_interval: int = 50
    eval_interval: int = 500
    checkpoint_interval: int = 1_000
    seed: int = 7
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "list-sorting-sparse-attention-adam"
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        positive_integers = (
            self.steps,
            self.batch_size,
            self.train_min_length,
            self.train_max_length,
            self.eval_examples,
            self.long_eval_examples,
            self.long_eval_min_length,
            self.final_eval_examples,
            self.eval_batch_size,
            self.eval_attention_element_budget,
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
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        all_eval_lengths = self.eval_lengths + self.final_eval_lengths
        if not self.eval_lengths or any(length < 2 for length in all_eval_lengths):
            raise ValueError("evaluation lengths must be at least two")
        if len(set(all_eval_lengths)) != len(all_eval_lengths):
            raise ValueError("evaluation lengths must be unique")
        if self.d_model % self.heads:
            raise ValueError("d_model must divide evenly across attention heads")
        if self.architecture not in {"standard", "paper_gemma2"}:
            raise ValueError("architecture must be standard or paper_gemma2")
        if self.input_position_mode not in {"modular", "nape_only"}:
            raise ValueError("input_position_mode must be modular or nape_only")
        if self.value_input_mode not in {"embedding", "embedding_plus_scalar"}:
            raise ValueError(
                "value_input_mode must be embedding or embedding_plus_scalar"
            )
        if self.input_position_mode == "modular":
            if self.d_model % 2:
                raise ValueError("modular split inputs require an even d_model")
            position_dim = self.d_model // 2
            if position_dim != 8 * len(self.position_moduli):
                raise ValueError(
                    "fixed modular Fourier positions require eight dimensions "
                    "per modulus"
                )
            required_span = (
                self.position_offset_max
                - self.position_offset_min
                + 2 * max(all_eval_lengths)
                + 4
            )
            if math.prod(self.position_moduli) < required_span:
                raise ValueError("modular position period is too short")
        if self.architecture == "paper_gemma2" and (
            self.input_position_mode != "nape_only"
            or self.value_input_mode != "embedding"
        ):
            raise ValueError(
                "paper_gemma2 requires NAPE-only positions and embedding-only "
                "value inputs"
            )
        if self.position_offset_min > self.position_offset_max:
            raise ValueError("position offset bounds are reversed")
        if not 0 <= self.alibi_heads <= self.heads:
            raise ValueError("alibi_heads must be between zero and heads")
        if self.attention_normalizer not in {"entmax15", "softmax"}:
            raise ValueError(
                "attention_normalizer must be entmax15 or softmax"
            )
        if self.entmax_alpha != 1.5:
            raise ValueError("this implementation supports entmax alpha=1.5")
        if self.scaling_mode not in {"adaptive", "none"}:
            raise ValueError("scaling_mode must be adaptive or none")
        if self.scaling_mode == "adaptive" and self.alibi_heads == self.heads:
            raise ValueError(
                "adaptive scaling requires at least one NoPE head"
            )
        if self.scale_delta < 0 or self.scale_gamma_range <= 0:
            raise ValueError("invalid adaptive-scaling configuration")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be in [0, 1]")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.precision not in {"float32", "bfloat16", "bfloat16-true"}:
            raise ValueError(
                "precision must be float32, bfloat16, or bfloat16-true"
            )
        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError("optimizer_name must be adam or adamw")


def _entmax15_probabilities(scores: Tensor) -> Tensor:
    """Compute exact 1.5-entmax probabilities along the final dimension."""

    scores = scores / 2
    scores = scores - scores.max(dim=-1, keepdim=True).values
    sorted_scores = scores.sort(dim=-1, descending=True).values
    rho = torch.arange(
        1,
        scores.shape[-1] + 1,
        device=scores.device,
        dtype=scores.dtype,
    )
    view_shape = (1,) * (scores.ndim - 1) + (scores.shape[-1],)
    rho = rho.view(view_shape)
    mean = sorted_scores.cumsum(dim=-1) / rho
    mean_square = sorted_scores.square().cumsum(dim=-1) / rho
    variance_sum = rho * (mean_square - mean.square())
    delta = (1 - variance_sum) / rho
    thresholds = mean - delta.clamp_min(0).sqrt()
    support_size = (thresholds <= sorted_scores).sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1)
    threshold = thresholds.gather(-1, support_size - 1)
    probabilities = (scores - threshold).clamp_min(0).square()
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )


class _Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: Tensor) -> Tensor:
        probabilities = _entmax15_probabilities(scores)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor]:
        (probabilities,) = ctx.saved_tensors
        density = probabilities.sqrt()
        density_sum = density.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(density.dtype).tiny
        )
        mean_gradient = (gradient * density).sum(
            dim=-1,
            keepdim=True,
        ) / density_sum
        return (density * (gradient - mean_gradient),)


def entmax15(scores: Tensor, *, dim: int = -1) -> Tensor:
    """Compute exact 1.5-entmax with its finite analytic backward pass."""

    if dim not in {-1, scores.ndim - 1}:
        raise ValueError("entmax15 currently supports the final dimension only")
    return _Entmax15Function.apply(scores)


def nape_slopes(
    heads: int,
    alibi_heads: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return the paper's linear ALiBi/NoPE head split."""

    alibi = 1 / torch.arange(
        1,
        alibi_heads + 1,
        dtype=torch.float32,
        device=device,
    )
    nope = torch.zeros(heads - alibi_heads, device=device)
    return torch.cat((alibi, nope))


class AdaptiveEntmaxSelfAttention(nn.Module):
    """Causal ASEntmax-1.5 attention with NAPE positional biases."""

    def __init__(self, config: SparseAttentionAdamConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.alibi_heads = config.alibi_heads
        self.nope_heads = config.heads - config.alibi_heads
        self.head_dim = config.d_model // config.heads
        self.attention_normalizer = config.attention_normalizer
        self.scaling_mode = config.scaling_mode
        self.scale_delta = config.scale_delta
        self.scale_gamma_range = config.scale_gamma_range
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.beta_projection = (
            nn.Linear(config.d_model, self.nope_heads, bias=True)
            if self.scaling_mode == "adaptive"
            else None
        )
        self.gamma_projection = (
            nn.Linear(config.d_model, self.nope_heads, bias=True)
            if self.scaling_mode == "adaptive"
            else None
        )
        self.register_buffer(
            "slopes",
            nape_slopes(config.heads, config.alibi_heads),
            persistent=False,
        )
        self.last_metrics: dict[str, Tensor] = {}
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, model_dim = tensor.shape
        return tensor.view(
            batch,
            length,
            self.heads,
            model_dim // self.heads,
        ).transpose(1, 2)

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
        del (
            reverse_source_score_credit,
            reverse_source_value_credit,
            route_source_output_projection,
        )
        if attention_mask is not None:
            raise ValueError("ASEntmax baseline only supports its causal mask")
        if (
            backward_attention_gate is not None
            or signed_backward_attention
            or backward_source_multipliers is not None
            or source_attention_penalty_strength
            or route_output_projection
        ):
            raise ValueError("backward-routing controls are not supported")

        batch, length, model_dim = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.scaling_mode == "adaptive":
            assert self.beta_projection is not None
            assert self.gamma_projection is not None
            beta = F.softplus(self.beta_projection(hidden)).transpose(1, 2)
            gamma = self.scale_gamma_range * torch.tanh(
                self.gamma_projection(hidden)
            ).transpose(1, 2)
            log_position = torch.arange(
                2,
                length + 2,
                device=hidden.device,
                dtype=torch.float32,
            ).log()[None, None, :]
            scaler = self.scale_delta + beta.float() * log_position.pow(
                gamma.float()
            )
            query = torch.cat(
                (
                    query[:, : self.alibi_heads],
                    query[:, self.alibi_heads :]
                    * scaler.to(query.dtype).unsqueeze(-1),
                ),
                dim=1,
            )
        else:
            beta = hidden.new_zeros(())
            gamma = hidden.new_zeros(())
            scaler = hidden.new_ones(())

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        positions = torch.arange(length, device=hidden.device)
        relative_distance = positions[None, :] - positions[:, None]
        scores = scores + (
            self.slopes.to(scores.dtype)[None, :, None, None]
            * relative_distance[None, None].to(scores.dtype)
        )
        causal_mask = positions[None, :] <= positions[:, None]
        scores = scores.float().masked_fill(
            ~causal_mask[None, None],
            -1e9,
        )
        if self.attention_normalizer == "entmax15":
            weights = entmax15(scores)
        else:
            weights = scores.softmax(dim=-1)
        weights = weights.masked_fill(~causal_mask[None, None], 0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        attended = weights.to(value.dtype) @ value

        with torch.no_grad():
            support = weights.gt(0) & causal_mask[None, None]
            allowed = causal_mask.sum() * batch
            per_head_support = support.sum(dim=(0, 2, 3)) / (
                batch * length
            )
            self.last_metrics = {
                "support_size": support.sum() / (batch * self.heads * length),
                "support_fraction": support.sum() / (allowed * self.heads),
                "alibi_support_size": (
                    per_head_support[: self.alibi_heads].mean()
                    if self.alibi_heads
                    else per_head_support.new_zeros(())
                ),
                "nope_support_size": (
                    per_head_support[self.alibi_heads :].mean()
                    if self.nope_heads
                    else per_head_support.new_zeros(())
                ),
                "beta_mean": beta.mean(),
                "gamma_mean": gamma.mean(),
                "scale_mean": scaler.mean(),
            }

        attended = attended.transpose(1, 2).contiguous().view(
            batch,
            length,
            model_dim,
        )
        return self.output(attended)


class PaperMatchedRMSNorm(nn.Module):
    """Gemma-2 RMSNorm, including its zero-initialized additive scale."""

    def __init__(self, dimension: int, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dimension))
        self.epsilon = epsilon

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * (1.0 + self.weight.float())).to(hidden.dtype)


class PaperMatchedGatedFeedForward(nn.Module):
    """Gemma-2 gated GELU feed-forward layer."""

    def __init__(self, config: SparseAttentionAdamConfig) -> None:
        super().__init__()
        intermediate_size = int(config.d_model * config.ffn_multiplier)
        self.gate = nn.Linear(config.d_model, intermediate_size, bias=False)
        self.up = nn.Linear(config.d_model, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, config.d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.gelu(self.gate(hidden), approximate="tanh") * self.up(hidden))


class PaperMatchedDecoderBlock(nn.Module):
    """Gemma-2 residual/norm layout with ASEntmax self-attention."""

    def __init__(self, config: SparseAttentionAdamConfig) -> None:
        super().__init__()
        self.input_norm = PaperMatchedRMSNorm(config.d_model)
        self.attention = AdaptiveEntmaxSelfAttention(config)
        self.post_attention_norm = PaperMatchedRMSNorm(config.d_model)
        self.pre_feedforward_norm = PaperMatchedRMSNorm(config.d_model)
        self.feedforward = PaperMatchedGatedFeedForward(config)
        self.post_feedforward_norm = PaperMatchedRMSNorm(config.d_model)

    def forward(self, hidden: Tensor) -> Tensor:
        attention_output = self.attention(self.input_norm(hidden))
        hidden = hidden + self.post_attention_norm(attention_output)
        feedforward_output = self.feedforward(self.pre_feedforward_norm(hidden))
        return hidden + self.post_feedforward_norm(feedforward_output)


class PaperMatchedDecoder(nn.Module):
    """Small Gemma-2-style decoder body for the paper-matched task variant."""

    def __init__(
        self,
        config: SparseAttentionAdamConfig,
        *,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, config.d_model)
        self.blocks = nn.ModuleList(
            PaperMatchedDecoderBlock(config) for _ in range(config.layers)
        )
        self.final_norm = PaperMatchedRMSNorm(config.d_model)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(self, token_ids: Tensor) -> Tensor:
        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class SparseAttentionPointerTransformer(nn.Module):
    """Matched pointer-next Transformer using ASEntmax instead of top-k."""

    def __init__(self, config: SparseAttentionAdamConfig) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = PointerNextVocabulary("numbers", config.symbol_count)
        model_config = ModelConfig(
            vocab_size=self.vocabulary.size,
            symbol_count=config.symbol_count,
            representation=(
                "numbers"
                if config.value_input_mode == "embedding_plus_scalar"
                else "alphabet"
            ),
            d_model=config.d_model,
            n_layers=config.layers,
            n_heads=config.heads,
            ffn_multiplier=config.ffn_multiplier,
            dropout=0.0,
            position_pattern="none",
        )
        if config.architecture == "paper_gemma2":
            self.encoder = PaperMatchedDecoder(
                config,
                vocabulary_size=self.vocabulary.size,
            )
        elif config.input_position_mode == "modular":
            self.encoder = SplitInputDecoderTransformer(
                model_config,
                content_dim=config.d_model // 2,
            )
            for block in self.encoder.blocks:
                block.attention = AdaptiveEntmaxSelfAttention(config)
        else:
            self.encoder = DecoderTransformer(model_config)
            for block in self.encoder.blocks:
                block.attention = AdaptiveEntmaxSelfAttention(config)

        if config.input_position_mode == "modular":
            position_dimension = config.d_model // 2
            self.position_embedding: ModularPositionEmbedding | None = (
                ModularPositionEmbedding(
                    position_dimension,
                    config.position_moduli,
                )
            )
            with torch.no_grad():
                _set_modular_fourier_codebooks(self.position_embedding)
            self.position_embedding.requires_grad_(False)
        else:
            self.position_embedding = None
        self.output = nn.Linear(config.d_model, config.symbol_count)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def position_embeddings(self, prompt_ids: Tensor, offsets: Tensor) -> Tensor:
        if self.position_embedding is None:
            raise ValueError("NAPE-only models have no modular position embeddings")
        token_offsets = torch.arange(
            prompt_ids.shape[1],
            device=prompt_ids.device,
        )
        return self.position_embedding(
            offsets[:, None] + token_offsets[None, :]
        )

    def forward(self, prompt_ids: Tensor, *, offsets: Tensor) -> Tensor:
        if self.position_embedding is None:
            hidden = self.encoder.hidden_states(prompt_ids)
        else:
            hidden = self.encoder.hidden_states(
                prompt_ids,
                extra_input_embeddings=self.position_embeddings(
                    prompt_ids,
                    offsets,
                ),
            )
        return self.output(hidden[:, -1])

    def attention_metrics(self) -> dict[str, float]:
        summaries: dict[str, list[float]] = {}
        for block in self.encoder.blocks:
            attention = block.attention
            if not isinstance(attention, AdaptiveEntmaxSelfAttention):
                continue
            for name, value in attention.last_metrics.items():
                summaries.setdefault(name, []).append(float(value))
        return {
            f"attention/{name}": sum(values) / len(values)
            for name, values in summaries.items()
        }


def pointer_targets(batch: PointerNextBatch) -> Tensor:
    return batch.values.gather(1, (batch.pointers + 1).unsqueeze(1)).squeeze(1)


def make_position_offsets(
    config: SparseAttentionAdamConfig,
    examples: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if config.input_position_mode == "nape_only":
        return torch.zeros(examples, dtype=torch.long, device=device)
    return sample_position_offsets(
        examples,
        minimum=config.position_offset_min,
        maximum=config.position_offset_max,
        generator=generator,
        device=device,
    )


def make_evaluation_data(
    config: SparseAttentionAdamConfig,
    *,
    lengths: tuple[int, ...],
    examples: int,
    long_examples: int | None = None,
    long_min_length: int | None = None,
) -> dict[int, tuple[PointerNextBatch, Tensor]]:
    generator = torch.Generator().manual_seed(config.seed + 30_400)
    return {
        length: (
            make_pointer_next_batch(
                (
                    long_examples
                    if long_examples is not None
                    and long_min_length is not None
                    and length >= long_min_length
                    else examples
                ),
                length,
                generator=generator,
                vocabulary=PointerNextVocabulary(
                    "numbers",
                    config.symbol_count,
                ),
            ),
            make_position_offsets(
                config,
                (
                    long_examples
                    if long_examples is not None
                    and long_min_length is not None
                    and length >= long_min_length
                    else examples
                ),
                generator=generator,
                device=torch.device("cpu"),
            ),
        )
        for length in lengths
    }


def evaluation_batch_size(
    config: SparseAttentionAdamConfig,
    *,
    prompt_length: int,
) -> int:
    attention_limited = config.eval_attention_element_budget // (
        config.heads * prompt_length * prompt_length
    )
    return min(config.eval_batch_size, max(1, attention_limited))


@torch.inference_mode()
def evaluate_model(
    model: SparseAttentionPointerTransformer,
    evaluation_data: dict[int, tuple[PointerNextBatch, Tensor]],
    *,
    device: torch.device,
    config: SparseAttentionAdamConfig,
) -> dict[str, float]:
    model.eval()
    summary: dict[str, float] = {}
    in_domain = []
    out_of_domain = []
    for length, (batch, offsets) in evaluation_data.items():
        targets = pointer_targets(batch)
        batch_size = evaluation_batch_size(
            config,
            prompt_length=batch.prompt_length,
        )
        loss_sum = 0.0
        correct = 0
        for start in range(0, targets.shape[0], batch_size):
            end = min(start + batch_size, targets.shape[0])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(
                    config.precision == "bfloat16" and device.type == "cuda"
                ),
            ):
                logits = model(
                    batch.prompt_ids[start:end].to(device),
                    offsets=offsets[start:end].to(device),
                )
            target = targets[start:end].to(device)
            loss_sum += float(
                F.cross_entropy(logits.float(), target, reduction="sum")
            )
            correct += int(logits.argmax(dim=-1).eq(target).sum())
        accuracy = correct / targets.shape[0]
        summary[f"eval/length_{length}/loss"] = loss_sum / targets.shape[0]
        summary[f"eval/length_{length}/accuracy"] = accuracy
        summary[f"eval/length_{length}/examples"] = float(targets.shape[0])
        summary[f"eval/length_{length}/batch_size"] = float(batch_size)
        (
            in_domain if length <= config.train_max_length else out_of_domain
        ).append(accuracy)
    if in_domain:
        summary["eval/in_domain_accuracy_mean"] = sum(in_domain) / len(in_domain)
    if out_of_domain:
        summary["eval/out_of_domain_accuracy_mean"] = sum(out_of_domain) / len(
            out_of_domain
        )
    return summary


def learning_rate_at_step(
    config: SparseAttentionAdamConfig,
    step: int,
) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        config.steps - config.warmup_steps,
        1,
    )
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    scale = config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * cosine
    return config.learning_rate * scale


def initialize_wandb(
    config: SparseAttentionAdamConfig,
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
    config: SparseAttentionAdamConfig,
    model: SparseAttentionPointerTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_generator: torch.Generator,
    wandb_run_id: str | None,
) -> None:
    torch.save(
        {
            "experiment": "pointer_next_asentmax_adam",
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "data_generator_state": data_generator.get_state(),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def run(config: SparseAttentionAdamConfig) -> Path:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    metrics_path = output_dir / "metrics.jsonl"

    model = SparseAttentionPointerTransformer(config).to(device)
    if config.precision == "bfloat16-true":
        if device.type != "cuda":
            raise ValueError("bfloat16-true precision requires a CUDA device")
        model = model.to(dtype=torch.bfloat16)
    optimizer_class = (
        torch.optim.AdamW if config.optimizer_name == "adamw" else torch.optim.Adam
    )
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    data_generator = torch.Generator().manual_seed(config.seed + 1_000)
    start_step = 1
    checkpoint = None
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        if checkpoint.get("experiment") != "pointer_next_asentmax_adam":
            raise ValueError("resume checkpoint belongs to another experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
        start_step = int(checkpoint["step"]) + 1

    evaluation_data = make_evaluation_data(
        config,
        lengths=config.eval_lengths,
        examples=config.eval_examples,
        long_examples=config.long_eval_examples,
        long_min_length=config.long_eval_min_length,
    )
    final_evaluation_data = make_evaluation_data(
        config,
        lengths=config.final_eval_lengths,
        examples=config.final_eval_examples,
    )
    resumed_wandb_id = (
        config.wandb_run_id
        or (checkpoint.get("wandb_run_id") if checkpoint is not None else None)
    )
    wandb_run = initialize_wandb(config, run_id=resumed_wandb_id)
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}", flush=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial = {
        "step": float(start_step - 1),
        "model/parameters": float(parameter_count),
        "train/min_length": float(config.train_min_length),
        "train/max_length": float(config.train_max_length),
    }
    with metrics_path.open("a") as metrics_file:
        metrics_file.write(json.dumps(initial) + "\n")
    if wandb_run is not None:
        wandb_run.log(initial, step=start_step - 1)

    started_at = time.monotonic()
    for step in range(start_step, config.steps + 1):
        length = int(
            torch.randint(
                config.train_min_length,
                config.train_max_length + 1,
                (),
                generator=data_generator,
            )
        )
        batch = make_pointer_next_batch(
            config.batch_size,
            length,
            generator=data_generator,
            vocabulary=model.vocabulary,
            device=device,
        )
        offsets = make_position_offsets(
            config,
            config.batch_size,
            generator=data_generator,
            device=device,
        )
        target = pointer_targets(batch)
        lr = learning_rate_at_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=config.precision == "bfloat16" and device.type == "cuda",
        ):
            logits = model(batch.prompt_ids, offsets=offsets)
            loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()

        should_log = step % config.log_interval == 0
        should_evaluate = step % config.eval_interval == 0
        should_checkpoint = step % config.checkpoint_interval == 0
        if should_log or should_evaluate or should_checkpoint or step == config.steps:
            summary = {
                "step": float(step),
                "train/length": float(length),
                "train/loss": float(loss),
                "train/accuracy": float(
                    logits.argmax(dim=-1).eq(target).float().mean()
                ),
                "optimizer/learning_rate": lr,
                "optimizer/gradient_norm": float(gradient_norm),
                "timing/steps_per_second": (
                    (step - start_step + 1) / (time.monotonic() - started_at)
                ),
            }
            summary.update(model.attention_metrics())
            if should_evaluate or step == config.steps:
                summary.update(
                    evaluate_model(
                        model,
                        evaluation_data,
                        device=device,
                        config=config,
                    )
                )
            if step == config.steps:
                summary.update(
                    evaluate_model(
                        model,
                        final_evaluation_data,
                        device=device,
                        config=config,
                    )
                )
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            if wandb_run is not None:
                wandb_run.log(summary, step=step)

        if should_checkpoint or step == config.steps:
            checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(
                checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                step=step,
                data_generator=data_generator,
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
            )
            save_checkpoint(
                output_dir / "latest.pt",
                config=config,
                model=model,
                optimizer=optimizer,
                step=step,
                data_generator=data_generator,
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
            )

    if wandb_run is not None:
        wandb_run.finish()
    return output_dir


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=SparseAttentionAdamConfig.run_name)
    parser.add_argument("--output-dir", default=SparseAttentionAdamConfig.output_dir)
    parser.add_argument("--steps", type=int, default=SparseAttentionAdamConfig.steps)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SparseAttentionAdamConfig.batch_size,
    )
    parser.add_argument(
        "--train-min-length",
        type=int,
        default=SparseAttentionAdamConfig.train_min_length,
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=SparseAttentionAdamConfig.train_max_length,
    )
    parser.add_argument(
        "--eval-lengths",
        type=parse_int_tuple,
        default=SparseAttentionAdamConfig.eval_lengths,
    )
    parser.add_argument(
        "--final-eval-lengths",
        type=parse_int_tuple,
        default=SparseAttentionAdamConfig.final_eval_lengths,
    )
    parser.add_argument(
        "--eval-examples",
        type=int,
        default=SparseAttentionAdamConfig.eval_examples,
    )
    parser.add_argument(
        "--long-eval-examples",
        type=int,
        default=SparseAttentionAdamConfig.long_eval_examples,
    )
    parser.add_argument(
        "--long-eval-min-length",
        type=int,
        default=SparseAttentionAdamConfig.long_eval_min_length,
    )
    parser.add_argument(
        "--final-eval-examples",
        type=int,
        default=SparseAttentionAdamConfig.final_eval_examples,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=SparseAttentionAdamConfig.eval_batch_size,
    )
    parser.add_argument(
        "--eval-attention-element-budget",
        type=int,
        default=SparseAttentionAdamConfig.eval_attention_element_budget,
    )
    parser.add_argument(
        "--symbol-count",
        type=int,
        default=SparseAttentionAdamConfig.symbol_count,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=SparseAttentionAdamConfig.d_model,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=SparseAttentionAdamConfig.layers,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=SparseAttentionAdamConfig.heads,
    )
    parser.add_argument(
        "--ffn-multiplier",
        type=float,
        default=SparseAttentionAdamConfig.ffn_multiplier,
    )
    parser.add_argument(
        "--alibi-heads",
        type=int,
        default=SparseAttentionAdamConfig.alibi_heads,
    )
    parser.add_argument(
        "--attention-normalizer",
        choices=("entmax15", "softmax"),
        default=SparseAttentionAdamConfig.attention_normalizer,
    )
    parser.add_argument(
        "--scaling-mode",
        choices=("adaptive", "none"),
        default=SparseAttentionAdamConfig.scaling_mode,
    )
    parser.add_argument(
        "--architecture",
        choices=("standard", "paper_gemma2"),
        default=SparseAttentionAdamConfig.architecture,
    )
    parser.add_argument(
        "--input-position-mode",
        choices=("modular", "nape_only"),
        default=SparseAttentionAdamConfig.input_position_mode,
    )
    parser.add_argument(
        "--value-input-mode",
        choices=("embedding", "embedding_plus_scalar"),
        default=SparseAttentionAdamConfig.value_input_mode,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=SparseAttentionAdamConfig.learning_rate,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=SparseAttentionAdamConfig.weight_decay,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=SparseAttentionAdamConfig.warmup_steps,
    )
    parser.add_argument(
        "--minimum-lr-ratio",
        type=float,
        default=SparseAttentionAdamConfig.minimum_lr_ratio,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=SparseAttentionAdamConfig.gradient_clip,
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "bfloat16-true"),
        default=SparseAttentionAdamConfig.precision,
    )
    parser.add_argument(
        "--optimizer-name",
        choices=("adam", "adamw"),
        default=SparseAttentionAdamConfig.optimizer_name,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=SparseAttentionAdamConfig.log_interval,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=SparseAttentionAdamConfig.eval_interval,
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=SparseAttentionAdamConfig.checkpoint_interval,
    )
    parser.add_argument("--seed", type=int, default=SparseAttentionAdamConfig.seed)
    parser.add_argument("--device", default=SparseAttentionAdamConfig.device)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project",
        default=SparseAttentionAdamConfig.wandb_project,
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--resume")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(SparseAttentionAdamConfig(**vars(args)))


if __name__ == "__main__":
    main()
