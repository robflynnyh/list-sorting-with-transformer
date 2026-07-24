"""Train an autoregressive modular sequence from PTR position to PTR + 1."""

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
from .model import DecoderTransformer, ModelConfig, SplitInputDecoderTransformer
from .pointer_position_probe import aggregate_length_ranges
from .positions import ModularPositionEmbedding, sample_position_offsets
from .tokens import PAD, PointerNextVocabulary


@dataclass(frozen=True)
class PositionSequenceConfig:
    representation: str = "numbers"
    symbol_count: int = 10
    train_min_length: int = 2
    train_max_length: int = 20
    eval_max_length: int = 400
    steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    gradient_noise_scale: float = 0.0
    gradient_noise_decay: float = 0.25
    successor_attention_supervision_weight: float = 0.0
    log_interval: int = 250
    eval_interval: int = 1_000
    eval_examples: int = 512
    eval_batch_size: int = 32
    checkpoint_interval: int = 1_000
    seed: int = 7
    input_layout: str = "split"
    position_moduli: tuple[int, ...] = (31, 37, 41, 47)
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    successor_attention_isolation_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.representation not in {"alphabet", "numbers"}:
            raise ValueError("invalid representation")
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        if self.eval_max_length < self.train_max_length:
            raise ValueError("eval_max_length must include training lengths")
        if any(
            value < 1
            for value in (
                self.steps,
                self.batch_size,
                self.log_interval,
                self.eval_interval,
                self.eval_examples,
                self.eval_batch_size,
                self.checkpoint_interval,
            )
        ):
            raise ValueError("integer settings must be positive")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.gradient_noise_scale < 0 or self.gradient_noise_decay < 0:
            raise ValueError("gradient noise settings must be nonnegative")
        if self.successor_attention_supervision_weight < 0:
            raise ValueError(
                "successor attention supervision weight must be nonnegative"
            )
        if not 0.0 <= self.successor_attention_isolation_probability <= 1.0:
            raise ValueError(
                "successor_attention_isolation_probability must be in [0, 1]"
            )
        if (
            self.successor_attention_supervision_weight > 0
            and self.successor_attention_isolation_probability > 0
        ):
            raise ValueError(
                "successor attention supervision replaces successor isolation"
            )
        if self.input_layout not in {"additive", "split"}:
            raise ValueError("input_layout must be 'additive' or 'split'")
        if not self.position_moduli or any(
            modulus < 2 for modulus in self.position_moduli
        ):
            raise ValueError("position moduli must all be at least two")
        if any(
            math.gcd(left, right) != 1
            for index, left in enumerate(self.position_moduli)
            for right in self.position_moduli[index + 1 :]
        ):
            raise ValueError("position moduli must be pairwise coprime")
        required_span = (
            self.position_offset_max
            - self.position_offset_min
            + 2 * self.eval_max_length
            + 2
        )
        if math.prod(self.position_moduli) < required_span:
            raise ValueError("position moduli do not cover the evaluation span")


class ModularPositionSequenceModel(nn.Module):
    """Generate one product key for PTR, then one product key for PTR + 1."""

    def __init__(
        self,
        model_config: ModelConfig,
        position_moduli: tuple[int, ...],
        *,
        split_input: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = (
            SplitInputDecoderTransformer(
                model_config,
                content_dim=model_config.d_model // 2,
            )
            if split_input
            else DecoderTransformer(model_config)
        )
        position_dim = (
            self.encoder.position_dim
            if isinstance(self.encoder, SplitInputDecoderTransformer)
            else model_config.d_model
        )
        self.query_projection = nn.Linear(model_config.d_model, position_dim)
        self.position_embedding = ModularPositionEmbedding(
            position_dim,
            position_moduli,
        )
        nn.init.normal_(self.query_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.query_projection.bias)

    @property
    def moduli(self) -> tuple[int, ...]:
        return self.position_embedding.moduli

    @property
    def output_steps(self) -> int:
        return 2

    def target_sequence(
        self,
        pointers: Tensor,
        offsets: Tensor,
    ) -> Tensor:
        pointer_positions = offsets + 1 + 2 * pointers
        positions = torch.stack((pointer_positions, pointer_positions + 1), dim=1)
        return torch.stack(
            [positions.remainder(modulus) for modulus in self.moduli],
            dim=-1,
        )

    def history_embeddings(self, history: Tensor) -> Tensor:
        if (
            history.ndim != 3
            or history.shape[1] > self.output_steps
            or history.shape[2] != len(self.moduli)
        ):
            raise ValueError(
                "history must be [batch, at most two positions, moduli]"
            )
        return torch.cat(
            [
                codebook(history[..., component_index])
                for component_index, codebook in enumerate(
                    self.position_embedding.codebooks
                )
            ],
            dim=-1,
        )

    def input_hidden_states(
        self,
        prompt_ids: Tensor,
        history: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        batch_size, prompt_length = prompt_ids.shape
        history_length = history.shape[1]
        stream_length = prompt_length + history_length
        stream_positions = (
            offsets[:, None]
            + torch.arange(stream_length, device=prompt_ids.device)[None, :]
        )
        position_embeddings = self.position_embedding(stream_positions)
        if isinstance(self.encoder, SplitInputDecoderTransformer):
            content_embeddings = self.encoder.embed(prompt_ids)
            if history_length:
                content_embeddings = torch.cat(
                    (content_embeddings, self.history_embeddings(history)),
                    dim=1,
                )
            return torch.cat((content_embeddings, position_embeddings), dim=-1)

        placeholder_ids = torch.full(
            (batch_size, history_length),
            PAD,
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
        token_ids = torch.cat((prompt_ids, placeholder_ids), dim=1)
        extra_embeddings = position_embeddings
        if history_length:
            placeholder_embeddings = self.encoder.embed(placeholder_ids)
            extra_embeddings[:, prompt_length:] = (
                self.history_embeddings(history) - placeholder_embeddings
            )
        return self.encoder.embed(token_ids) + extra_embeddings

    def hidden_states(
        self,
        prompt_ids: Tensor,
        history: Tensor,
        *,
        offsets: Tensor,
        isolate_successor: Tensor | None = None,
    ) -> Tensor:
        hidden = self.input_hidden_states(
            prompt_ids,
            history,
            offsets=offsets,
        )
        attention_mask = self.successor_attention_mask(
            batch_size=prompt_ids.shape[0],
            stream_length=hidden.shape[1],
            history_length=history.shape[1],
            isolate_successor=isolate_successor,
            device=prompt_ids.device,
        )
        for block in self.encoder.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.encoder.final_norm(hidden)

    def hidden_states_with_successor_attention(
        self,
        prompt_ids: Tensor,
        history: Tensor,
        *,
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if history.shape[1] != 1:
            raise ValueError(
                "successor attention supervision requires one position in history"
            )
        hidden = self.input_hidden_states(
            prompt_ids,
            history,
            offsets=offsets,
        )
        layer_logits = []
        for block in self.encoder.blocks:
            normalized = block.attention_norm(hidden)
            layer_logits.append(
                block.attention.query_key_logits(
                    normalized,
                    query_index=-1,
                )
            )
            hidden = block(hidden)
        return self.encoder.final_norm(hidden), torch.stack(layer_logits, dim=1)

    @staticmethod
    def successor_attention_mask(
        *,
        batch_size: int,
        stream_length: int,
        history_length: int,
        isolate_successor: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if isolate_successor is None:
            return None
        if history_length != 1:
            raise ValueError(
                "successor isolation requires exactly one position in history"
            )
        if isolate_successor.shape != (batch_size,):
            raise ValueError("isolate_successor must have shape [batch]")
        if isolate_successor.dtype != torch.bool:
            raise ValueError("isolate_successor must be boolean")
        isolate_successor = isolate_successor.to(device=device)
        if not bool(isolate_successor.any()):
            return None
        mask = torch.ones(
            batch_size,
            stream_length,
            stream_length,
            device=device,
            dtype=torch.bool,
        )
        mask[isolate_successor, -1, :] = False
        mask[isolate_successor, -1, -1] = True
        return mask

    def position_logits(self, hidden: Tensor) -> tuple[Tensor, ...]:
        query = self.query_projection(hidden)
        components = query.split(
            self.position_embedding.component_dim,
            dim=-1,
        )
        return tuple(
            component @ codebook.weight.T
            / math.sqrt(self.position_embedding.component_dim)
            for component, codebook in zip(
                components,
                self.position_embedding.codebooks,
            )
        )

    def teacher_forced_logits(
        self,
        prompt_ids: Tensor,
        targets: Tensor,
        *,
        offsets: Tensor,
        isolate_successor: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], ...]:
        history = targets[:, :-1]
        hidden = self.hidden_states(
            prompt_ids,
            history,
            offsets=offsets,
            isolate_successor=isolate_successor,
        )
        first_prediction_index = prompt_ids.shape[1] - 1
        prediction_states = hidden[
            :,
            first_prediction_index : first_prediction_index + self.output_steps,
        ]
        return tuple(
            self.position_logits(prediction_states[:, step])
            for step in range(self.output_steps)
        )

    def teacher_forced_logits_with_successor_attention(
        self,
        prompt_ids: Tensor,
        targets: Tensor,
        *,
        offsets: Tensor,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], Tensor]:
        history = targets[:, :-1]
        hidden, attention_logits = self.hidden_states_with_successor_attention(
            prompt_ids,
            history,
            offsets=offsets,
        )
        first_prediction_index = prompt_ids.shape[1] - 1
        prediction_states = hidden[
            :,
            first_prediction_index : first_prediction_index + self.output_steps,
        ]
        logits = tuple(
            self.position_logits(prediction_states[:, step])
            for step in range(self.output_steps)
        )
        return logits, attention_logits

    @torch.inference_mode()
    def generate_positions(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        history = torch.empty(
            prompt_ids.shape[0],
            0,
            len(self.moduli),
            device=prompt_ids.device,
            dtype=torch.long,
        )
        for step in range(self.output_steps):
            hidden = self.hidden_states(prompt_ids, history, offsets=offsets)
            logits = self.position_logits(hidden[:, -1])
            prediction = torch.stack(
                [component.argmax(dim=-1) for component in logits],
                dim=-1,
            )
            history = torch.cat((history, prediction[:, None, :]), dim=1)
        return history


def load_stage_one_checkpoint(
    model: ModularPositionSequenceModel,
    checkpoint_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("stage-one checkpoint is missing model_state")
    missing, unexpected = model.load_state_dict(source_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "stage-one architecture mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "stage_one_step": int(checkpoint.get("step", 0)),
        "transferred_tensors": len(source_state),
    }


def sequence_loss_and_metrics(
    model: ModularPositionSequenceModel,
    prompt_ids: Tensor,
    pointers: Tensor,
    offsets: Tensor,
    *,
    isolate_successor: Tensor | None = None,
    successor_attention_supervision_weight: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    targets = model.target_sequence(pointers, offsets)
    attention_logits = None
    if successor_attention_supervision_weight:
        if isolate_successor is not None and bool(isolate_successor.any()):
            raise ValueError(
                "attention supervision cannot be combined with isolation"
            )
        logits, attention_logits = (
            model.teacher_forced_logits_with_successor_attention(
                prompt_ids,
                targets,
                offsets=offsets,
            )
        )
    else:
        logits = model.teacher_forced_logits(
            prompt_ids,
            targets,
            offsets=offsets,
            isolate_successor=isolate_successor,
        )
    losses = [
        F.cross_entropy(component_logits, targets[:, step, component])
        for step, step_logits in enumerate(logits)
        for component, component_logits in enumerate(step_logits)
    ]
    position_loss = torch.stack(losses).mean()
    total_loss = position_loss
    attention_metrics = {}
    if attention_logits is not None:
        flattened_logits = attention_logits.flatten(0, 2)
        attention_targets = torch.full(
            (flattened_logits.shape[0],),
            flattened_logits.shape[-1] - 1,
            dtype=torch.long,
            device=flattened_logits.device,
        )
        attention_loss = F.cross_entropy(
            flattened_logits,
            attention_targets,
        )
        attention_metrics = {
            "successor_attention_supervision_loss": float(
                attention_loss.detach().item()
            ),
            "successor_attention_target_accuracy": float(
                attention_logits.argmax(dim=-1)
                .eq(attention_logits.shape[-1] - 1)
                .float()
                .mean()
                .item()
            ),
            "successor_attention_target_probability": float(
                attention_logits.softmax(dim=-1)[..., -1].mean().item()
            ),
        }
        total_loss = (
            position_loss
            + successor_attention_supervision_weight * attention_loss
        )
    predictions = torch.stack(
        [
            torch.stack(
                [component.argmax(dim=-1) for component in step_logits],
                dim=-1,
            )
            for step_logits in logits
        ],
        dim=1,
    )
    component_correct = predictions.eq(targets)
    pointer_exact = component_correct[:, 0].all(dim=1)
    next_exact = component_correct[:, 1].all(dim=1)
    return total_loss, {
        "loss": float(total_loss.detach().item()),
        "position_loss": float(position_loss.detach().item()),
        "successor_attention_isolation_fraction": (
            float(isolate_successor.float().mean().item())
            if isolate_successor is not None
            else 0.0
        ),
        "teacher_forced_pointer_position_accuracy": float(
            pointer_exact.float().mean().item()
        ),
        "teacher_forced_next_position_accuracy": float(
            next_exact.float().mean().item()
        ),
        "teacher_forced_both_positions_accuracy": float(
            (pointer_exact & next_exact).float().mean().item()
        ),
        "teacher_forced_residue_accuracy": float(
            component_correct.float().mean().item()
        ),
        **attention_metrics,
    }


def generated_metrics(
    generated: Tensor,
    targets: Tensor,
    *,
    moduli: tuple[int, ...],
) -> dict[str, float]:
    correct = generated.eq(targets)
    pointer_exact = correct[:, 0].all(dim=1)
    next_exact = correct[:, 1].all(dim=1)
    expected_successor = torch.stack(
        [
            (generated[:, 0, index] + 1).remainder(modulus)
            for index, modulus in enumerate(moduli)
        ],
        dim=1,
    )
    successor_consistent = generated[:, 1].eq(expected_successor).all(dim=1)
    metrics = {
        "pointer_position_accuracy": float(pointer_exact.float().mean().item()),
        "next_position_accuracy": float(next_exact.float().mean().item()),
        "both_positions_accuracy": float(
            (pointer_exact & next_exact).float().mean().item()
        ),
        "residue_accuracy": float(correct.float().mean().item()),
        "successor_consistency": float(
            successor_consistent.float().mean().item()
        ),
    }
    for index, modulus in enumerate(moduli):
        metrics[f"pointer_mod_{modulus}_accuracy"] = float(
            correct[:, 0, index].float().mean().item()
        )
        metrics[f"next_mod_{modulus}_accuracy"] = float(
            correct[:, 1, index].float().mean().item()
        )
        wraps = targets[:, 0, index].eq(modulus - 1)
        metrics[f"next_mod_{modulus}_wrap_correct_fraction"] = float(
            (wraps & correct[:, 1, index]).float().mean().item()
        )
        metrics[f"next_mod_{modulus}_wrap_fraction"] = float(
            wraps.float().mean().item()
        )
    return metrics


def selected_evaluation_lengths(config: PositionSequenceConfig) -> list[int]:
    candidates = {
        config.train_min_length,
        (config.train_min_length + config.train_max_length) // 2,
        config.train_max_length,
        min(config.eval_max_length, config.train_max_length + 5),
        min(config.eval_max_length, 40),
        config.eval_max_length,
    }
    return sorted(candidates)


@torch.inference_mode()
def evaluate_lengths(
    model: ModularPositionSequenceModel,
    vocabulary: PointerNextVocabulary,
    lengths: list[int],
    *,
    config: PositionSequenceConfig,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    was_training = model.training
    model.eval()
    results = {}
    for length in lengths:
        generator = torch.Generator().manual_seed(seed + 104_729 * length)
        totals: dict[str, float] = {}
        processed = 0
        while processed < config.eval_examples:
            batch_size = min(
                config.eval_batch_size,
                config.eval_examples - processed,
            )
            batch = make_pointer_next_batch(
                batch_size,
                length,
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
            offsets = sample_position_offsets(
                batch_size,
                minimum=config.position_offset_min,
                maximum=config.position_offset_max,
                generator=generator,
                device=device,
            )
            targets = model.target_sequence(batch.pointers, offsets)
            with autocast_context(device):
                generated = model.generate_positions(
                    batch.prompt_ids,
                    offsets=offsets,
                )
                _, teacher_forced = sequence_loss_and_metrics(
                    model,
                    batch.prompt_ids,
                    batch.pointers,
                    offsets,
                )
            metrics = {
                **generated_metrics(
                    generated,
                    targets,
                    moduli=model.moduli,
                ),
                **teacher_forced,
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            processed += batch_size
        averaged = {
            name: value / config.eval_examples
            for name, value in totals.items()
        }
        for modulus in model.moduli:
            correct_fraction = averaged[
                f"next_mod_{modulus}_wrap_correct_fraction"
            ]
            wrap_fraction = averaged[f"next_mod_{modulus}_wrap_fraction"]
            averaged[f"next_mod_{modulus}_wrap_accuracy"] = (
                correct_fraction / wrap_fraction
                if wrap_fraction
                else 0.0
            )
        results[length] = averaged
    model.train(was_training)
    return results


def learning_rate_at_step(config: PositionSequenceConfig, step: int) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    return config.learning_rate


def gradient_noise_std(config: PositionSequenceConfig, step: int) -> float:
    if config.gradient_noise_scale == 0:
        return 0.0
    return config.gradient_noise_scale / (step ** config.gradient_noise_decay)


def add_gradient_noise(
    model: nn.Module,
    *,
    config: PositionSequenceConfig,
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


def save_checkpoint(
    path: Path,
    *,
    model: ModularPositionSequenceModel,
    optimizer: torch.optim.Optimizer,
    config: PositionSequenceConfig,
    step: int,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "probe": "pointer_position_sequence",
            "model_config": model.encoder.config.as_dict(),
            "train_config": asdict(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
        },
        path,
    )


def train(
    model: ModularPositionSequenceModel,
    config: PositionSequenceConfig,
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
    isolation_generator = torch.Generator().manual_seed(config.seed + 2)
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
    for step in range(1, config.steps + 1):
        learning_rate = learning_rate_at_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        length = sample_length(
            config.train_min_length,
            config.train_max_length,
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
            minimum=config.position_offset_min,
            maximum=config.position_offset_max,
            generator=generator,
            device=device,
        )
        isolate_successor = (
            torch.rand(config.batch_size, generator=isolation_generator)
            < config.successor_attention_isolation_probability
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, metrics = sequence_loss_and_metrics(
                model,
                batch.prompt_ids,
                batch.pointers,
                offsets,
                isolate_successor=isolate_successor,
                successor_attention_supervision_weight=(
                    config.successor_attention_supervision_weight
                ),
            )
        loss.backward()
        noise_std = add_gradient_noise(model, config=config, step=step)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()
        if step == 1 or step % config.log_interval == 0:
            row = {
                "step": float(step),
                "length": float(length),
                "learning_rate": learning_rate,
                "gradient_noise_std": noise_std,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started_at,
                **metrics,
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
        if step % config.checkpoint_interval == 0:
            save_checkpoint(
                output_directory / "checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                generator=generator,
            )
        if step % config.eval_interval == 0 or step == config.steps:
            per_length = evaluate_lengths(
                model,
                vocabulary,
                selected_evaluation_lengths(config),
                config=config,
                seed=config.seed + 20_000,
                device=device,
            )
            evaluations.append(
                {
                    "step": step,
                    "per_length": {
                        str(length): metrics
                        for length, metrics in per_length.items()
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "evaluation_both_positions_accuracy": {
                            str(length): metrics["both_positions_accuracy"]
                            for length, metrics in per_length.items()
                        },
                        "evaluation_successor_consistency": {
                            str(length): metrics["successor_consistency"]
                            for length, metrics in per_length.items()
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
                            f"eval/length_{length}/{name}": value
                            for length, metrics in per_length.items()
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
    final_per_length = evaluate_lengths(
        model,
        vocabulary,
        selected_evaluation_lengths(config),
        config=config,
        seed=config.seed + 30_000,
        device=device,
    )
    results = {
        "probe": "pointer_position_sequence",
        "model_config": model.encoder.config.as_dict(),
        "train_config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-one-checkpoint", type=Path)
    parser.add_argument("--representation", choices=("alphabet", "numbers"), default="numbers")
    parser.add_argument("--symbol-count", type=int, default=10)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=400)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-noise-scale", type=float, default=0.0)
    parser.add_argument("--gradient-noise-decay", type=float, default=0.25)
    parser.add_argument(
        "--successor-attention-supervision-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--input-layout",
        choices=("additive", "split"),
        default="split",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="override checkpoint dropout; defaults to 0 for new models",
    )
    parser.add_argument("--position-moduli", default="31,37,41,47")
    parser.add_argument("--position-offset-min", type=int, default=-1_000_000)
    parser.add_argument("--position-offset-max", type=int, default=1_000_000)
    parser.add_argument(
        "--successor-attention-isolation-probability",
        type=float,
        default=0.0,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    moduli = tuple(
        int(value.strip())
        for value in args.position_moduli.split(",")
        if value.strip()
    )
    config = PositionSequenceConfig(
        representation=args.representation,
        symbol_count=args.symbol_count,
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        eval_max_length=args.eval_max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        gradient_noise_scale=args.gradient_noise_scale,
        gradient_noise_decay=args.gradient_noise_decay,
        successor_attention_supervision_weight=(
            args.successor_attention_supervision_weight
        ),
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_examples=args.eval_examples,
        eval_batch_size=args.eval_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        input_layout=args.input_layout,
        position_moduli=moduli,
        position_offset_min=args.position_offset_min,
        position_offset_max=args.position_offset_max,
        successor_attention_isolation_probability=(
            args.successor_attention_isolation_probability
        ),
    )
    vocabulary = PointerNextVocabulary(config.representation, config.symbol_count)
    checkpoint = (
        torch.load(args.stage_one_checkpoint, map_location="cpu")
        if args.stage_one_checkpoint is not None
        else None
    )
    if checkpoint is not None:
        checkpoint_model_config = checkpoint.get("model_config")
        if not isinstance(checkpoint_model_config, dict):
            raise ValueError("stage-one checkpoint is missing model_config")
        checkpoint_model_config = dict(checkpoint_model_config)
        if args.dropout is not None:
            checkpoint_model_config["dropout"] = args.dropout
        checkpoint_train_config = checkpoint.get("train_config", {})
        checkpoint_layout = checkpoint_train_config.get(
            "input_layout",
            "additive",
        )
        if checkpoint_layout != config.input_layout:
            raise ValueError(
                "stage-one checkpoint input layout does not match stage two"
            )
        model_config = ModelConfig(**checkpoint_model_config)
    else:
        model_config = ModelConfig(
            vocab_size=vocabulary.size,
            symbol_count=config.symbol_count,
            representation=config.representation,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_multiplier=args.ffn_multiplier,
            dropout=0.0 if args.dropout is None else args.dropout,
            position_pattern="none",
            rotate_values_with_rope=False,
        )
    if model_config.vocab_size != vocabulary.size:
        raise ValueError("stage-one checkpoint vocabulary does not match")
    torch.manual_seed(config.seed)
    model = ModularPositionSequenceModel(
        model_config,
        config.position_moduli,
        split_input=config.input_layout == "split",
    )
    initialization = (
        load_stage_one_checkpoint(model, args.stage_one_checkpoint)
        if args.stage_one_checkpoint is not None
        else {
            "stage_one_step": 0,
            "transferred_tensors": 0,
            "initialized_from_scratch": True,
        }
    )
    device = resolve_device(args.device)
    model.to(device)
    metadata = {
        "probe": "pointer_position_sequence",
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_directory": str(args.output_directory),
        "stage_one_checkpoint": (
            str(args.stage_one_checkpoint)
            if args.stage_one_checkpoint is not None
            else None
        ),
        **initialization,
    }
    print(json.dumps(metadata), flush=True)
    tracker = None
    if args.wandb_project is not None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("install W&B to enable tracking") from error
        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "probe": "pointer_position_sequence",
                "model": model.encoder.config.as_dict(),
                "training": asdict(config),
                "initialization": initialization,
            },
        )
        print(json.dumps({"wandb_url": tracker.url}), flush=True)
    try:
        results = train(
            model,
            config,
            vocabulary=vocabulary,
            output_directory=args.output_directory,
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
                "output_directory": str(args.output_directory),
                "aggregate": results["final_aggregate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
