"""Train a vector probe to recover the pointer token's absolute position."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import make_pointer_next_batch, sample_length
from .evaluate import resolve_device
from .evaluation import autocast_context
from .model import DecoderTransformer, ModelConfig
from .plots import plot_length_generalization, plot_training_history
from .tokens import PointerNextVocabulary


@dataclass(frozen=True)
class PointerPositionConfig:
    representation: str = "numbers"
    symbol_count: int = 10
    train_min_length: int = 2
    train_max_length: int = 20
    eval_max_length: int = 400
    steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    gradient_noise_scale: float = 0.0
    gradient_noise_decay: float = 0.25
    curriculum: bool = False
    curriculum_start_length: int = 2
    curriculum_threshold: float = 0.99
    curriculum_patience: int = 20
    curriculum_review_probability: float = 0.2
    log_interval: int = 50
    eval_interval: int = 1_000
    eval_examples: int = 512
    eval_batch_size: int = 256
    seed: int = 7
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    objective: str = "vector_mse"
    rotary_base: float = 10_000.0
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    checkpoint_interval: int = 1_000

    def __post_init__(self) -> None:
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("invalid representation")
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        if self.eval_max_length < self.train_max_length:
            raise ValueError("eval_max_length must include the training range")
        integer_fields = (
            self.steps,
            self.batch_size,
            self.log_interval,
            self.eval_interval,
            self.eval_examples,
            self.eval_batch_size,
            self.checkpoint_interval,
            self.d_model,
            self.n_layers,
            self.n_heads,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("integer settings must be positive")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal position pairs")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.lr_schedule not in {"cosine", "constant"}:
            raise ValueError("lr_schedule must be 'cosine' or 'constant'")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.gradient_noise_scale < 0 or self.gradient_noise_decay < 0:
            raise ValueError("gradient noise settings must be non-negative")
        if not (
            self.train_min_length
            <= self.curriculum_start_length
            <= self.train_max_length
        ):
            raise ValueError("curriculum_start_length must be in the training range")
        if not 0.0 <= self.curriculum_threshold <= 1.0:
            raise ValueError("curriculum_threshold must be in [0, 1]")
        if self.curriculum_patience < 1:
            raise ValueError("curriculum_patience must be positive")
        if not 0.0 <= self.curriculum_review_probability <= 1.0:
            raise ValueError("curriculum_review_probability must be in [0, 1]")
        if self.objective not in {"vector_mse", "pointer_ce"}:
            raise ValueError(
                "objective must be 'vector_mse' or 'pointer_ce'"
            )
        if self.rotary_base <= 1.0:
            raise ValueError("rotary_base must be greater than one")
        if self.position_offset_min > self.position_offset_max:
            raise ValueError("position_offset_min must be <= position_offset_max")


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed absolute position embeddings with Transformer sinusoidal features."""

    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, positions: Tensor) -> Tensor:
        angles = (
            positions.to(dtype=self.inverse_frequency.dtype).unsqueeze(-1)
            * self.inverse_frequency
        )
        embedding = torch.empty(
            *positions.shape,
            self.inverse_frequency.shape[0] * 2,
            device=positions.device,
            dtype=torch.float32,
        )
        embedding[..., 0::2] = angles.sin().to(dtype=torch.float32)
        embedding[..., 1::2] = angles.cos().to(dtype=torch.float32)
        return embedding


class PointerPositionProbe(nn.Module):
    """Emit the absolute sinusoidal position vector added at ``<PTR>``."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.encoder = DecoderTransformer(model_config)
        self.query_projection = nn.Linear(model_config.d_model, model_config.d_model)
        self.position_embedding = SinusoidalPositionEmbedding(
            model_config.d_model,
            model_config.rotary_base,
        )
        nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.query_projection.bias)

    @property
    def layer_position_modes(self) -> tuple[str, ...]:
        return self.encoder.layer_position_modes

    def candidate_token_offsets(self, length: int, *, device: torch.device) -> Tensor:
        if length < 2:
            raise ValueError("length must be at least two")
        list_indices = torch.arange(length - 1, device=device)
        return 1 + 2 * list_indices

    def candidate_positions(
        self,
        length: int,
        *,
        device: torch.device,
        offsets: Tensor | None = None,
    ) -> Tensor:
        token_offsets = self.candidate_token_offsets(length, device=device)
        if offsets is None:
            return token_offsets
        return offsets[:, None] + token_offsets[None, :]

    def input_position_embeddings(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        offsets: Tensor | None = None,
    ) -> Tensor:
        token_offsets = torch.arange(sequence_length, device=device)
        positions = (
            token_offsets
            if offsets is None
            else offsets[:, None] + token_offsets[None, :]
        )
        return self.position_embedding(positions)

    def target_token_offsets(self, pointers: Tensor) -> Tensor:
        return 1 + 2 * pointers

    def target_positions(self, pointers: Tensor, offsets: Tensor | None = None) -> Tensor:
        token_offsets = self.target_token_offsets(pointers)
        if offsets is None:
            return token_offsets
        return offsets + token_offsets

    def target_embeddings(
        self,
        pointers: Tensor,
        offsets: Tensor | None = None,
    ) -> Tensor:
        return self.position_embedding(self.target_positions(pointers, offsets))

    def forward(self, prompt_ids: Tensor, *, offsets: Tensor | None = None) -> Tensor:
        hidden = self.hidden_states(prompt_ids, offsets=offsets)
        return self.query_projection(hidden[:, -1])

    def hidden_states(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor | None = None,
    ) -> Tensor:
        input_positions = self.input_position_embeddings(
            prompt_ids.shape[1],
            device=prompt_ids.device,
            offsets=offsets,
        )
        return self.encoder.hidden_states(
            prompt_ids,
            extra_input_embeddings=input_positions,
        )

    def pointer_logits(
        self,
        prompt_ids: Tensor,
        *,
        length: int,
        offsets: Tensor | None = None,
    ) -> Tensor:
        hidden = self.hidden_states(prompt_ids, offsets=offsets)
        query = self.query_projection(hidden[:, -1])
        candidate_offsets = self.candidate_token_offsets(
            length,
            device=prompt_ids.device,
        )
        candidate_states = hidden[:, candidate_offsets]
        return (candidate_states * query[:, None, :]).sum(dim=-1) / math.sqrt(
            query.shape[-1]
        )

    def logits_for_length(
        self,
        predictions: Tensor,
        *,
        length: int,
        offsets: Tensor | None = None,
    ) -> Tensor:
        positions = self.candidate_positions(
            length,
            device=predictions.device,
            offsets=offsets,
        )
        candidates = self.position_embedding(positions).to(dtype=predictions.dtype)
        if candidates.ndim == 2:
            candidates = candidates[None, :, :]
        squared_distances = (predictions[:, None, :] - candidates).square()
        return -squared_distances.mean(dim=-1)


def learning_rate_at_step(config: PointerPositionConfig, step: int) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if config.lr_schedule == "constant":
        return config.learning_rate
    decay_steps = max(config.steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (0.1 + 0.9 * cosine)


def gradient_noise_std(config: PointerPositionConfig, step: int) -> float:
    if config.gradient_noise_scale == 0:
        return 0.0
    return config.gradient_noise_scale / (step ** config.gradient_noise_decay)


def add_gradient_noise(
    model: nn.Module,
    *,
    config: PointerPositionConfig,
    step: int,
) -> float:
    std = gradient_noise_std(config, step)
    if std == 0.0:
        return 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.add_(torch.randn_like(parameter.grad) * std)
    return std


def sample_training_length(
    config: PointerPositionConfig,
    *,
    current_max_length: int,
    generator: torch.Generator,
) -> int:
    if not config.curriculum:
        return sample_length(
            config.train_min_length,
            config.train_max_length,
            generator=generator,
        )
    review_draw = torch.rand((), generator=generator).item()
    if review_draw < config.curriculum_review_probability:
        return sample_length(
            config.train_min_length,
            config.train_max_length,
            generator=generator,
        )
    return sample_length(
        config.train_min_length,
        current_max_length,
        generator=generator,
    )


def selected_evaluation_lengths(config: PointerPositionConfig) -> list[int]:
    midpoint = (config.train_min_length + config.train_max_length) // 2
    candidates = {
        config.train_min_length,
        midpoint,
        config.train_max_length,
        min(config.eval_max_length, config.train_max_length + 5),
        min(config.eval_max_length, 40),
        config.eval_max_length,
    }
    return sorted(candidates)


def sample_position_offsets(
    batch_size: int,
    *,
    config: PointerPositionConfig,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    offsets = torch.randint(
        config.position_offset_min,
        config.position_offset_max + 1,
        (batch_size,),
        generator=generator,
    )
    return offsets.to(device)


def pointer_position_metrics(
    emitted_vectors: Tensor,
    pointers: Tensor,
    *,
    model: PointerPositionProbe,
    length: int,
    offsets: Tensor | None = None,
    train_max_length: int,
) -> dict[str, float]:
    targets = model.target_embeddings(pointers, offsets).to(dtype=emitted_vectors.dtype)
    logits = model.logits_for_length(emitted_vectors, length=length, offsets=offsets)
    predicted_classes = logits.argmax(dim=-1)
    target_token_offsets = model.target_token_offsets(pointers)
    predicted_token_offsets = model.candidate_token_offsets(
        length,
        device=pointers.device,
    )[predicted_classes]
    absolute_errors = (predicted_token_offsets - target_token_offsets).abs()
    correct = predicted_token_offsets.eq(target_token_offsets)
    unseen = pointers.gt(train_max_length - 2)
    seen = ~unseen
    return {
        "loss": float(F.mse_loss(emitted_vectors, targets).item()),
        "argmax_accuracy": float(correct.float().mean().item()),
        "argmax_token_mae": float(absolute_errors.float().mean().item()),
        "seen_argmax_accuracy": float(correct[seen].float().mean().item())
        if bool(seen.any())
        else 0.0,
        "seen_argmax_token_mae": float(absolute_errors[seen].float().mean().item())
        if bool(seen.any())
        else 0.0,
        "unseen_argmax_accuracy": float(correct[unseen].float().mean().item())
        if bool(unseen.any())
        else 0.0,
        "unseen_argmax_token_mae": float(
            absolute_errors[unseen].float().mean().item()
        )
        if bool(unseen.any())
        else 0.0,
        "unseen_pointer_fraction": float(unseen.float().mean().item()),
    }


def pointer_position_ce_metrics(
    logits: Tensor,
    pointers: Tensor,
    *,
    model: PointerPositionProbe,
    length: int,
    train_max_length: int,
) -> dict[str, float]:
    predicted_classes = logits.argmax(dim=-1)
    target_token_offsets = model.target_token_offsets(pointers)
    predicted_token_offsets = model.candidate_token_offsets(
        length,
        device=pointers.device,
    )[predicted_classes]
    absolute_errors = (predicted_token_offsets - target_token_offsets).abs()
    correct = predicted_classes.eq(pointers)
    unseen = pointers.gt(train_max_length - 2)
    seen = ~unseen
    return {
        "loss": float(F.cross_entropy(logits, pointers).item()),
        "argmax_accuracy": float(correct.float().mean().item()),
        "argmax_token_mae": float(absolute_errors.float().mean().item()),
        "seen_argmax_accuracy": float(correct[seen].float().mean().item())
        if bool(seen.any())
        else 0.0,
        "seen_argmax_token_mae": float(absolute_errors[seen].float().mean().item())
        if bool(seen.any())
        else 0.0,
        "unseen_argmax_accuracy": float(correct[unseen].float().mean().item())
        if bool(unseen.any())
        else 0.0,
        "unseen_argmax_token_mae": float(
            absolute_errors[unseen].float().mean().item()
        )
        if bool(unseen.any())
        else 0.0,
        "unseen_pointer_fraction": float(unseen.float().mean().item()),
    }


def batch_loss_and_metrics(
    model: PointerPositionProbe,
    batch_prompt_ids: Tensor,
    pointers: Tensor,
    *,
    length: int,
    offsets: Tensor,
    config: PointerPositionConfig,
) -> tuple[Tensor, dict[str, float]]:
    if config.objective == "pointer_ce":
        logits = model.pointer_logits(batch_prompt_ids, length=length, offsets=offsets)
        loss = F.cross_entropy(logits, pointers)
        metrics = pointer_position_ce_metrics(
            logits.detach().float(),
            pointers,
            model=model,
            length=length,
            train_max_length=config.train_max_length,
        )
        return loss, metrics

    emitted_vectors = model(batch_prompt_ids, offsets=offsets)
    targets = model.target_embeddings(pointers, offsets).to(dtype=emitted_vectors.dtype)
    loss = F.mse_loss(emitted_vectors, targets)
    metrics = pointer_position_metrics(
        emitted_vectors.detach().float(),
        pointers,
        model=model,
        length=length,
        offsets=offsets,
        train_max_length=config.train_max_length,
    )
    return loss, metrics


@torch.inference_mode()
def evaluate_lengths(
    model: PointerPositionProbe,
    vocabulary: PointerNextVocabulary,
    lengths: list[int],
    *,
    config: PointerPositionConfig,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    was_training = model.training
    model.eval()
    results = {}
    for length in lengths:
        generator = torch.Generator().manual_seed(seed + 104_729 * int(length))
        totals: dict[str, float] = {}
        processed = 0
        while processed < config.eval_examples:
            current_batch_size = min(
                config.eval_batch_size,
                config.eval_examples - processed,
            )
            batch = make_pointer_next_batch(
                current_batch_size,
                int(length),
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
            offsets = sample_position_offsets(
                current_batch_size,
                config=config,
                generator=generator,
                device=device,
            )
            with autocast_context(device):
                _, metrics = batch_loss_and_metrics(
                    model,
                    batch.prompt_ids,
                    batch.pointers,
                    length=int(length),
                    offsets=offsets,
                    config=config,
                )
            metrics = {
                name: float(value)
                for name, value in metrics.items()
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * current_batch_size
            processed += current_batch_size
        results[int(length)] = {
            name: value / config.eval_examples
            for name, value in totals.items()
        }
    model.train(was_training)
    return results


def save_checkpoint(
    path: Path,
    *,
    model: PointerPositionProbe,
    optimizer: torch.optim.Optimizer,
    config: PointerPositionConfig,
    step: int,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "probe": "pointer_position",
            "model_config": model.encoder.config.as_dict(),
            "train_config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
        },
        path,
    )


def aggregate_length_ranges(
    per_length: dict[int, dict[str, float]],
    *,
    train_min_length: int,
    train_max_length: int,
) -> dict[str, dict[str, float]]:
    groups = {
        "in_domain": [
            metrics
            for length, metrics in per_length.items()
            if train_min_length <= length <= train_max_length
        ],
        "out_of_domain": [
            metrics
            for length, metrics in per_length.items()
            if length > train_max_length
        ],
    }
    return {
        group_name: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in rows[0]
        }
        for group_name, rows in groups.items()
        if rows
    }


def train(
    model: PointerPositionProbe,
    config: PointerPositionConfig,
    *,
    vocabulary: PointerNextVocabulary,
    output_directory: Path,
    device: torch.device,
    tracker: Any | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")

    generator = torch.Generator().manual_seed(config.seed + 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    history = []
    evaluations = []
    started_at = time.monotonic()
    model.train()
    current_max_length = (
        config.curriculum_start_length
        if config.curriculum
        else config.train_max_length
    )
    curriculum_streak = 0

    for step in range(1, config.steps + 1):
        current_learning_rate = learning_rate_at_step(config, step)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        length = sample_training_length(
            config,
            current_max_length=current_max_length,
            generator=generator,
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
            config=config,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, batch_metrics = batch_loss_and_metrics(
                model,
                batch.prompt_ids,
                batch.pointers,
                length=length,
                offsets=offsets,
                config=config,
            )
        loss.backward()
        noise_std = add_gradient_noise(model, config=config, step=step)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()
        if (
            config.curriculum
            and current_max_length < config.train_max_length
            and length == current_max_length
        ):
            if batch_metrics["argmax_accuracy"] >= config.curriculum_threshold:
                curriculum_streak += 1
            else:
                curriculum_streak = 0
            if curriculum_streak >= config.curriculum_patience:
                current_max_length += 1
                curriculum_streak = 0

        if step % config.checkpoint_interval == 0:
            save_checkpoint(
                output_directory / "checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                generator=generator,
            )

        if step == 1 or step % config.log_interval == 0:
            row = {
                "step": float(step),
                "length": float(length),
                "learning_rate": current_learning_rate,
                "gradient_norm": float(gradient_norm),
                "gradient_noise_std": noise_std,
                "curriculum_max_length": float(current_max_length),
                "elapsed_seconds": time.monotonic() - started_at,
                **batch_metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if tracker is not None:
                tracker.log(
                    {
                        "step": step,
                        **{
                            f"train/{name}": value
                            for name, value in row.items()
                            if name != "step"
                        },
                    }
                )

        if step % config.eval_interval == 0 or step == config.steps:
            if step % config.checkpoint_interval:
                save_checkpoint(
                    output_directory / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    generator=generator,
                )
            per_length = evaluate_lengths(
                model,
                vocabulary,
                selected_evaluation_lengths(config),
                config=config,
                seed=config.seed + 20_000,
                device=device,
            )
            evaluation_row = {
                "step": step,
                "per_length": {
                    str(eval_length): metrics
                    for eval_length, metrics in per_length.items()
                },
            }
            evaluations.append(evaluation_row)
            print(
                json.dumps(
                    {
                        "step": step,
                        "evaluation_argmax_accuracy": {
                            str(eval_length): metrics["argmax_accuracy"]
                            for eval_length, metrics in per_length.items()
                        },
                    }
                ),
                flush=True,
            )
            if tracker is not None:
                tracker.log(
                    {
                        "step": step,
                        **{
                            f"eval/length_{eval_length}/{name}": value
                            for eval_length, metrics in per_length.items()
                            for name, value in metrics.items()
                        },
                    }
                )
            model.train()

    checkpoint_path = output_directory / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        step=config.steps,
        generator=generator,
    )
    final_lengths = list(range(config.train_min_length, config.eval_max_length + 1))
    final_per_length = evaluate_lengths(
        model,
        vocabulary,
        final_lengths,
        config=config,
        seed=config.seed + 30_000,
        device=device,
    )
    results = {
        "probe": "pointer_position",
        "model_config": model.encoder.config.as_dict(),
        "train_config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "layer_position_modes": model.layer_position_modes,
        "wall_time_seconds": time.monotonic() - started_at,
        "history": history,
        "intermediate_evaluations": evaluations,
        "final_per_length": {
            str(length): metrics for length, metrics in final_per_length.items()
        },
        "final_aggregate": aggregate_length_ranges(
            final_per_length,
            train_min_length=config.train_min_length,
            train_max_length=config.train_max_length,
        ),
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    plot_training_history(history, output_directory / "training.png")
    plot_length_generalization(
        final_per_length,
        output_directory / "length_generalization.png",
        train_max_length=config.train_max_length,
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=("alphabet", "numbers"),
        default="numbers",
    )
    parser.add_argument("--symbol-count", type=int, default=10)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=400)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant"),
        default="cosine",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-noise-scale", type=float, default=0.0)
    parser.add_argument("--gradient-noise-decay", type=float, default=0.25)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-start-length", type=int, default=2)
    parser.add_argument("--curriculum-threshold", type=float, default=0.99)
    parser.add_argument("--curriculum-patience", type=int, default=20)
    parser.add_argument("--curriculum-review-probability", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--objective",
        choices=("vector_mse", "pointer_ce"),
        default="vector_mse",
        help="train against the PTR position vector or classify the PTR slot",
    )
    parser.add_argument("--rotary-base", type=float, default=10_000.0)
    parser.add_argument("--position-offset-min", type=int, default=-1_000_000)
    parser.add_argument("--position-offset-max", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = PointerPositionConfig(
        representation=args.representation,
        symbol_count=args.symbol_count,
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        eval_max_length=args.eval_max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        gradient_noise_scale=args.gradient_noise_scale,
        gradient_noise_decay=args.gradient_noise_decay,
        curriculum=args.curriculum,
        curriculum_start_length=args.curriculum_start_length,
        curriculum_threshold=args.curriculum_threshold,
        curriculum_patience=args.curriculum_patience,
        curriculum_review_probability=args.curriculum_review_probability,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_examples=args.eval_examples,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_multiplier=args.ffn_multiplier,
        dropout=args.dropout,
        objective=args.objective,
        rotary_base=args.rotary_base,
        position_offset_min=args.position_offset_min,
        position_offset_max=args.position_offset_max,
        checkpoint_interval=args.checkpoint_interval,
    )
    vocabulary = PointerNextVocabulary(config.representation, config.symbol_count)
    model_config = ModelConfig(
        vocab_size=vocabulary.size,
        symbol_count=config.symbol_count,
        representation=config.representation,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_multiplier=config.ffn_multiplier,
        dropout=config.dropout,
        position_pattern="none",
        rotary_base=config.rotary_base,
        rotate_values_with_rope=False,
    )
    torch.manual_seed(config.seed)
    model = PointerPositionProbe(model_config)
    device = resolve_device(args.device)
    model.to(device)
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = (
            Path("artifacts")
            / f"pointer_position_random_offset_{config.objective}_seed7"
        )
    metadata = {
        "probe": "pointer_position",
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_directory": str(output_directory),
        "layer_position_modes": model.layer_position_modes,
    }
    print(json.dumps(metadata), flush=True)
    tracker = None
    if args.wandb_project is not None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "install the 'tracking' extra to use W&B logging"
            ) from error
        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "probe": f"pointer_position_{config.objective}",
                "model": model.encoder.config.as_dict(),
                "training": asdict(config),
                "parameter_count": metadata["parameter_count"],
            },
        )
        print(json.dumps({"wandb_url": tracker.url}), flush=True)
    try:
        results = train(
            model,
            config,
            vocabulary=vocabulary,
            output_directory=output_directory,
            device=device,
            tracker=tracker,
        )
    finally:
        if tracker is not None:
            tracker.finish()
    print(
        json.dumps(
            {
                "completed": True,
                "output_directory": str(output_directory),
                "aggregate": results["final_aggregate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
