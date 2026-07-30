"""Retrieve the next list value from an autoregressively computed position."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.data import PointerNextBatch, make_pointer_pair_batch, sample_length
from ..core.evaluate import resolve_device
from ..core.evaluation import autocast_context
from ..core.model import ModelConfig, SplitInputDecoderTransformer
from .pointer_next_value_position import (
    ModularNextValuePositionModel,
    NextValuePositionConfig,
    generated_stage_four_metrics,
    next_value_position_loss_and_metrics,
    relative_logit_distillation_loss,
)
from .pointer_position_probe import aggregate_length_ranges
from .pointer_position_sequence import (
    add_gradient_noise,
    learning_rate_at_step,
    selected_evaluation_lengths,
)
from .pointer_value_from_position import target_token_ids
from ..core.positions import sample_position_offsets
from ..core.tokens import VALUE_OFFSET, PointerNextVocabulary


@dataclass(frozen=True)
class NextValueFromPositionConfig(NextValuePositionConfig):
    next_value_token_loss_weight: float = 1.0
    stage_four_distillation_weight: float = 1.0
    learning_rate_decay_start: int | None = None
    learning_rate_decay_end: int | None = None
    minimum_learning_rate: float = 0.0
    ema_decay: float = 0.0
    ema_start_step: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.next_value_token_loss_weight <= 0:
            raise ValueError("next_value_token_loss_weight must be positive")
        if self.stage_four_distillation_weight < 0:
            raise ValueError(
                "stage_four_distillation_weight must be nonnegative"
            )
        if self.learning_rate_decay_start is not None:
            decay_end = self.learning_rate_decay_end or self.steps
            if not (
                self.warmup_steps
                <= self.learning_rate_decay_start
                < decay_end
                <= self.steps
            ):
                raise ValueError(
                    "learning-rate decay must start after warmup and end no "
                    "later than the final step"
                )
            if not 0 <= self.minimum_learning_rate <= self.learning_rate:
                raise ValueError(
                    "minimum_learning_rate must be between zero and "
                    "learning_rate"
                )
        elif (
            self.learning_rate_decay_end is not None
            or self.minimum_learning_rate != 0
        ):
            raise ValueError("decay settings require learning-rate decay")
        if self.ema_decay == 0:
            if self.ema_start_step != 0:
                raise ValueError("ema_start_step requires EMA")
        elif not (
            0 < self.ema_decay < 1
            and 1 <= self.ema_start_step <= self.steps
        ):
            raise ValueError(
                "EMA requires decay in (0, 1) and a valid start step"
            )


class ModularNextValueFromPositionModel(ModularNextValuePositionModel):
    """Generate the Stage-4 trace, then copy the token at its final address."""

    def stage_five_hidden_states(
        self,
        prompt_ids: Tensor,
        position_history: Tensor,
        token_history: Tensor,
        next_position_history: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        batch_size, prompt_length = prompt_ids.shape
        if position_history.shape != (
            batch_size,
            2,
            len(self.moduli),
        ):
            raise ValueError(
                "position_history must be [batch, two positions, moduli]"
            )
        if token_history.shape != (batch_size, 1):
            raise ValueError("token_history must be [batch, one token]")
        if next_position_history.shape != (
            batch_size,
            1,
            len(self.moduli),
        ):
            raise ValueError(
                "next_position_history must be [batch, one position, moduli]"
            )

        stream_length = prompt_length + 4
        stream_positions = (
            offsets[:, None]
            + torch.arange(stream_length, device=prompt_ids.device)[None, :]
        )
        position_embeddings = self.position_embedding(stream_positions)
        prompt_content = self.encoder.embed(prompt_ids)
        address_content = self.history_embeddings(position_history)
        token_content = self.encoder.embed(token_history)
        next_address_content = self.history_embeddings(next_position_history)
        content_embeddings = torch.cat(
            (
                prompt_content,
                address_content,
                token_content,
                next_address_content,
            ),
            dim=1,
        )

        if isinstance(self.encoder, SplitInputDecoderTransformer):
            hidden = torch.cat(
                (content_embeddings, position_embeddings),
                dim=-1,
            )
        else:
            hidden = content_embeddings + position_embeddings
            hidden[:, prompt_length : prompt_length + 2] = address_content
            hidden[:, -1:] = next_address_content
        for block in self.encoder.blocks:
            hidden = block(hidden)
        return self.encoder.final_norm(hidden)

    def teacher_forced_next_value_token_logits(
        self,
        prompt_ids: Tensor,
        position_targets: Tensor,
        token_targets: Tensor,
        next_position_targets: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        hidden = self.stage_five_hidden_states(
            prompt_ids,
            position_targets,
            token_targets[:, None],
            next_position_targets[:, None],
            offsets=offsets,
        )
        return self.token_logits(hidden[:, -1])

    @torch.inference_mode()
    def generate_stage_five_trace(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        positions, tokens, next_positions = self.generate_stage_four_trace(
            prompt_ids,
            offsets=offsets,
        )
        hidden = self.stage_five_hidden_states(
            prompt_ids,
            positions,
            tokens[:, None],
            next_positions[:, None],
            offsets=offsets,
        )
        next_tokens = self.token_logits(hidden[:, -1]).argmax(dim=-1)
        return positions, tokens, next_positions, next_tokens


def target_next_value_token_ids(batch: PointerNextBatch) -> Tensor:
    """Return the list value immediately following the marked value."""

    row_indices = torch.arange(
        batch.values.shape[0],
        device=batch.values.device,
    )
    return batch.values[row_indices, batch.pointers + 1] + VALUE_OFFSET


def stage_five_learning_rate_at_step(
    config: NextValueFromPositionConfig,
    step: int,
) -> float:
    """Apply inherited warmup followed by optional cosine decay."""

    warmup_rate = learning_rate_at_step(config, step)
    decay_start = config.learning_rate_decay_start
    if decay_start is None or step <= decay_start:
        return warmup_rate
    decay_end = config.learning_rate_decay_end or config.steps
    if step >= decay_end:
        return config.minimum_learning_rate
    progress = (step - decay_start) / (decay_end - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


@torch.no_grad()
def update_ema_model(
    ema_model: ModularNextValueFromPositionModel,
    model: ModularNextValueFromPositionModel,
    *,
    decay: float,
    initialize: bool = False,
) -> None:
    """Update a detached parameter-average model in place."""

    if initialize:
        ema_model.load_state_dict(model.state_dict())
        return
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise ValueError("EMA and training models do not match")
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.lerp_(
            model_parameters[name].detach(),
            1.0 - decay,
        )
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA and training model buffers do not match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


def load_stage_four_checkpoint(
    model: ModularNextValueFromPositionModel,
    checkpoint_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("probe") != "pointer_next_value_position":
        raise ValueError("checkpoint is not a Stage-4 next-position model")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("stage-four checkpoint is missing model_state")
    missing, unexpected = model.load_state_dict(source_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            "stage-four architecture mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "stage_four_step": int(checkpoint.get("step", 0)),
        "transferred_tensors": len(source_state),
    }


def stage_four_distillation_loss(
    model: ModularNextValueFromPositionModel,
    teacher: ModularNextValuePositionModel,
    batch: PointerNextBatch,
    offsets: Tensor,
) -> Tensor:
    position_targets = model.target_sequence(batch.pointers, offsets)
    token_targets = target_token_ids(batch)
    next_position_targets = model.target_next_value_position(
        batch.pointers,
        offsets,
    )
    student_position_logits = model.teacher_forced_logits(
        batch.prompt_ids,
        position_targets,
        offsets=offsets,
    )
    student_marked_token_logits = model.teacher_forced_token_logits(
        batch.prompt_ids,
        position_targets,
        offsets=offsets,
    )
    student_next_position_logits = (
        model.teacher_forced_next_value_position_logits(
            batch.prompt_ids,
            position_targets,
            token_targets,
            offsets=offsets,
        )
    )
    with torch.no_grad():
        teacher_position_logits = teacher.teacher_forced_logits(
            batch.prompt_ids,
            position_targets,
            offsets=offsets,
        )
        teacher_marked_token_logits = teacher.teacher_forced_token_logits(
            batch.prompt_ids,
            position_targets,
            offsets=offsets,
        )
        teacher_next_position_logits = (
            teacher.teacher_forced_next_value_position_logits(
                batch.prompt_ids,
                position_targets,
                token_targets,
                offsets=offsets,
            )
        )

    losses = [
        relative_logit_distillation_loss(student, source)
        for student_step, teacher_step in zip(
            student_position_logits,
            teacher_position_logits,
        )
        for student, source in zip(student_step, teacher_step)
    ]
    losses.append(
        relative_logit_distillation_loss(
            student_marked_token_logits,
            teacher_marked_token_logits,
        )
    )
    losses.extend(
        relative_logit_distillation_loss(student, source)
        for student, source in zip(
            student_next_position_logits,
            teacher_next_position_logits,
        )
    )
    return torch.stack(losses).mean()


def next_value_token_loss_and_metrics(
    model: ModularNextValueFromPositionModel,
    batch: PointerNextBatch,
    offsets: Tensor,
    *,
    config: NextValueFromPositionConfig,
    isolate_successor: Tensor | None = None,
    isolate_next_value_position: Tensor | None = None,
    stage_four_teacher: ModularNextValuePositionModel | None = None,
) -> tuple[Tensor, dict[str, float]]:
    inherited_loss, metrics = next_value_position_loss_and_metrics(
        model,
        batch,
        offsets,
        config=config,
        isolate_successor=isolate_successor,
        isolate_next_value_position=isolate_next_value_position,
    )
    position_targets = model.target_sequence(batch.pointers, offsets)
    marked_token_targets = target_token_ids(batch)
    next_position_targets = model.target_next_value_position(
        batch.pointers,
        offsets,
    )
    next_token_targets = target_next_value_token_ids(batch)
    next_token_logits = model.teacher_forced_next_value_token_logits(
        batch.prompt_ids,
        position_targets,
        marked_token_targets,
        next_position_targets,
        offsets=offsets,
    )
    next_token_loss = F.cross_entropy(next_token_logits, next_token_targets)
    next_token_predictions = next_token_logits.argmax(dim=-1)
    distillation_loss = next_token_loss.new_zeros(())
    if stage_four_teacher is not None:
        distillation_loss = stage_four_distillation_loss(
            model,
            stage_four_teacher,
            batch,
            offsets,
        )
    total_loss = (
        inherited_loss
        + config.next_value_token_loss_weight * next_token_loss
        + config.stage_four_distillation_weight * distillation_loss
    )
    return total_loss, {
        **metrics,
        "loss": float(total_loss.detach().item()),
        "stage_four_loss": float(inherited_loss.detach().item()),
        "next_value_token_loss": float(next_token_loss.detach().item()),
        "stage_four_distillation_loss": float(
            distillation_loss.detach().item()
        ),
        "teacher_forced_next_value_token_accuracy": float(
            next_token_predictions.eq(next_token_targets).float().mean().item()
        ),
    }


def generated_stage_five_metrics(
    generated_positions: Tensor,
    generated_marked_tokens: Tensor,
    generated_next_positions: Tensor,
    generated_next_tokens: Tensor,
    target_positions: Tensor,
    target_marked_tokens: Tensor,
    target_next_positions: Tensor,
    target_next_tokens: Tensor,
    *,
    moduli: tuple[int, ...],
) -> dict[str, float]:
    metrics = generated_stage_four_metrics(
        generated_positions,
        generated_marked_tokens,
        generated_next_positions,
        target_positions,
        target_marked_tokens,
        target_next_positions,
        moduli=moduli,
    )
    stage_four_correct = (
        generated_positions.eq(target_positions).flatten(1).all(dim=1)
        & generated_marked_tokens.eq(target_marked_tokens)
        & generated_next_positions.eq(target_next_positions).all(dim=1)
    )
    next_position_correct = generated_next_positions.eq(
        target_next_positions
    ).all(dim=1)
    next_token_correct = generated_next_tokens.eq(target_next_tokens)
    correct_next_positions = int(next_position_correct.sum().item())
    metrics.update(
        {
            "stage_four_complete_trace_accuracy": metrics[
                "complete_trace_accuracy"
            ],
            "next_value_token_accuracy": float(
                next_token_correct.float().mean().item()
            ),
            "next_value_token_accuracy_given_correct_position": (
                float(
                    (next_position_correct & next_token_correct).sum().item()
                    / correct_next_positions
                )
                if correct_next_positions
                else 0.0
            ),
            "complete_trace_accuracy": float(
                (stage_four_correct & next_token_correct)
                .float()
                .mean()
                .item()
            ),
        }
    )
    return metrics


@torch.inference_mode()
def evaluate_lengths(
    model: ModularNextValueFromPositionModel,
    vocabulary: PointerNextVocabulary,
    lengths: list[int],
    *,
    config: NextValueFromPositionConfig,
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
            batch = make_pointer_pair_batch(
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
            position_targets = model.target_sequence(batch.pointers, offsets)
            marked_token_targets = target_token_ids(batch)
            next_position_targets = model.target_next_value_position(
                batch.pointers,
                offsets,
            )
            next_token_targets = target_next_value_token_ids(batch)
            with autocast_context(device):
                generated = model.generate_stage_five_trace(
                    batch.prompt_ids,
                    offsets=offsets,
                )
                _, teacher_forced = next_value_token_loss_and_metrics(
                    model,
                    batch,
                    offsets,
                    config=config,
                )
            metrics = {
                **generated_stage_five_metrics(
                    *generated,
                    position_targets,
                    marked_token_targets,
                    next_position_targets,
                    next_token_targets,
                    moduli=model.moduli,
                ),
                **teacher_forced,
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            processed += batch_size
        results[length] = {
            name: value / config.eval_examples
            for name, value in totals.items()
        }
    model.train(was_training)
    return results


def save_checkpoint(
    path: Path,
    *,
    model: ModularNextValueFromPositionModel,
    optimizer: torch.optim.Optimizer,
    config: NextValueFromPositionConfig,
    step: int,
    generator: torch.Generator,
    averaging: dict[str, float | int] | None = None,
) -> None:
    payload = {
        "probe": "pointer_next_value_from_position",
        "model_config": model.encoder.config.as_dict(),
        "train_config": asdict(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "generator_state": generator.get_state(),
        "step": step,
    }
    if averaging is not None:
        payload["averaging"] = averaging
    torch.save(payload, path)


def train(
    model: ModularNextValueFromPositionModel,
    config: NextValueFromPositionConfig,
    *,
    vocabulary: PointerNextVocabulary,
    output_directory: Path,
    device: torch.device,
    tracker: Any | None = None,
    stage_four_teacher: ModularNextValuePositionModel | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(config.seed + 1)
    isolation_generator = torch.Generator().manual_seed(config.seed + 2)
    next_position_isolation_generator = torch.Generator().manual_seed(
        config.seed + 3
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    ema_model = None
    ema_initialized = False
    if config.ema_decay:
        ema_model = deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()
    history = []
    evaluations = []
    started_at = time.monotonic()
    model.train()
    for step in range(1, config.steps + 1):
        learning_rate = stage_five_learning_rate_at_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        length = sample_length(
            config.train_min_length,
            config.train_max_length,
            generator=generator,
        )
        batch = make_pointer_pair_batch(
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
        isolate_next_value_position = (
            torch.rand(
                config.batch_size,
                generator=next_position_isolation_generator,
            )
            < config.next_value_position_attention_isolation_probability
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, metrics = next_value_token_loss_and_metrics(
                model,
                batch,
                offsets,
                config=config,
                isolate_successor=isolate_successor,
                isolate_next_value_position=isolate_next_value_position,
                stage_four_teacher=stage_four_teacher,
            )
        loss.backward()
        noise_std = add_gradient_noise(model, config=config, step=step)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()
        if ema_model is not None and step >= config.ema_start_step:
            update_ema_model(
                ema_model,
                model,
                decay=config.ema_decay,
                initialize=not ema_initialized,
            )
            ema_initialized = True
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
            if ema_model is not None and ema_initialized:
                save_checkpoint(
                    output_directory / "checkpoint_ema.pt",
                    model=ema_model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    generator=generator,
                    averaging={
                        "decay": config.ema_decay,
                        "start_step": config.ema_start_step,
                    },
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
            ema_per_length = None
            if ema_model is not None and ema_initialized:
                ema_per_length = evaluate_lengths(
                    ema_model,
                    vocabulary,
                    selected_evaluation_lengths(config),
                    config=config,
                    seed=config.seed + 20_000,
                    device=device,
                )
            evaluation = {
                "step": step,
                "per_length": {
                    str(length): metrics
                    for length, metrics in per_length.items()
                },
            }
            if ema_per_length is not None:
                evaluation["ema_per_length"] = {
                    str(length): metrics
                    for length, metrics in ema_per_length.items()
                }
            evaluations.append(evaluation)
            print(
                json.dumps(
                    {
                        "step": step,
                        "evaluation_complete_trace_accuracy": {
                            str(length): metrics["complete_trace_accuracy"]
                            for length, metrics in per_length.items()
                        },
                        "evaluation_next_value_token_accuracy": {
                            str(length): metrics["next_value_token_accuracy"]
                            for length, metrics in per_length.items()
                        },
                        "ema_evaluation_complete_trace_accuracy": (
                            {
                                str(length): metrics[
                                    "complete_trace_accuracy"
                                ]
                                for length, metrics in ema_per_length.items()
                            }
                            if ema_per_length is not None
                            else None
                        ),
                        "ema_evaluation_next_value_token_accuracy": (
                            {
                                str(length): metrics[
                                    "next_value_token_accuracy"
                                ]
                                for length, metrics in ema_per_length.items()
                            }
                            if ema_per_length is not None
                            else None
                        ),
                    }
                ),
                flush=True,
            )
            if tracker is not None:
                tracking_metrics = {
                    "step": step,
                    **{
                        f"eval/length_{length}/{name}": value
                        for length, metrics in per_length.items()
                        for name, value in metrics.items()
                    },
                }
                if ema_per_length is not None:
                    tracking_metrics.update(
                        {
                            f"eval_ema/length_{length}/{name}": value
                            for length, metrics in ema_per_length.items()
                            for name, value in metrics.items()
                        }
                    )
                tracker.log(tracking_metrics)
            model.train()
    save_checkpoint(
        output_directory / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=config.steps,
        generator=generator,
    )
    if ema_model is not None and ema_initialized:
        save_checkpoint(
            output_directory / "checkpoint_ema.pt",
            model=ema_model,
            optimizer=optimizer,
            config=config,
            step=config.steps,
            generator=generator,
            averaging={
                "decay": config.ema_decay,
                "start_step": config.ema_start_step,
            },
        )
    final_per_length = evaluate_lengths(
        model,
        vocabulary,
        selected_evaluation_lengths(config),
        config=config,
        seed=config.seed + 30_000,
        device=device,
    )
    final_ema_per_length = None
    if ema_model is not None and ema_initialized:
        final_ema_per_length = evaluate_lengths(
            ema_model,
            vocabulary,
            selected_evaluation_lengths(config),
            config=config,
            seed=config.seed + 30_000,
            device=device,
        )
    results = {
        "probe": "pointer_next_value_from_position",
        "model_config": model.encoder.config.as_dict(),
        "train_config": asdict(config),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
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
    if final_ema_per_length is not None:
        results["final_ema_per_length"] = {
            str(length): metrics
            for length, metrics in final_ema_per_length.items()
        }
        results["final_ema_aggregate"] = aggregate_length_ranges(
            final_ema_per_length,
            train_min_length=config.train_min_length,
            train_max_length=config.train_max_length,
        )
    (output_directory / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-four-checkpoint", type=Path, required=True)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=400)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate-decay-start", type=int)
    parser.add_argument("--learning-rate-decay-end", type=int)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-start-step", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-noise-scale", type=float, default=0.0)
    parser.add_argument("--gradient-noise-decay", type=float, default=0.25)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--next-value-position-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--next-value-position-attention-isolation-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--next-value-position-consistency-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--next-value-token-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stage-four-distillation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--successor-attention-isolation-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    checkpoint = torch.load(args.stage_four_checkpoint, map_location="cpu")
    checkpoint_model_config = checkpoint.get("model_config")
    checkpoint_train_config = checkpoint.get("train_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("stage-four checkpoint is missing model_config")
    if not isinstance(checkpoint_train_config, dict):
        raise ValueError("stage-four checkpoint is missing train_config")
    model_config_values = dict(checkpoint_model_config)
    if args.dropout is not None:
        model_config_values["dropout"] = args.dropout
    model_config = ModelConfig(**model_config_values)
    position_moduli = tuple(checkpoint_train_config["position_moduli"])
    config = NextValueFromPositionConfig(
        representation=model_config.representation,
        symbol_count=model_config.symbol_count,
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        eval_max_length=args.eval_max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        learning_rate_decay_start=args.learning_rate_decay_start,
        learning_rate_decay_end=args.learning_rate_decay_end,
        minimum_learning_rate=args.minimum_learning_rate,
        ema_decay=args.ema_decay,
        ema_start_step=args.ema_start_step,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        gradient_noise_scale=args.gradient_noise_scale,
        gradient_noise_decay=args.gradient_noise_decay,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_examples=args.eval_examples,
        eval_batch_size=args.eval_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        input_layout=checkpoint_train_config["input_layout"],
        position_moduli=position_moduli,
        position_offset_min=checkpoint_train_config["position_offset_min"],
        position_offset_max=checkpoint_train_config["position_offset_max"],
        successor_attention_isolation_probability=(
            args.successor_attention_isolation_probability
        ),
        token_loss_weight=args.token_loss_weight,
        next_value_position_loss_weight=(
            args.next_value_position_loss_weight
        ),
        next_value_position_attention_isolation_probability=(
            args.next_value_position_attention_isolation_probability
        ),
        next_value_position_consistency_weight=(
            args.next_value_position_consistency_weight
        ),
        stage_three_distillation_weight=0.0,
        next_value_token_loss_weight=args.next_value_token_loss_weight,
        stage_four_distillation_weight=args.stage_four_distillation_weight,
    )
    vocabulary = PointerNextVocabulary(
        config.representation,
        config.symbol_count,
    )
    if model_config.vocab_size != vocabulary.size:
        raise ValueError("stage-four checkpoint vocabulary does not match")
    torch.manual_seed(config.seed)
    model = ModularNextValueFromPositionModel(
        model_config,
        config.position_moduli,
        split_input=config.input_layout == "split",
    )
    initialization = load_stage_four_checkpoint(
        model,
        args.stage_four_checkpoint,
    )
    stage_four_teacher = ModularNextValuePositionModel(
        model_config,
        config.position_moduli,
        split_input=config.input_layout == "split",
    )
    teacher_missing, teacher_unexpected = stage_four_teacher.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    if teacher_missing or teacher_unexpected:
        raise ValueError("could not initialize the Stage-4 teacher")
    stage_four_teacher.requires_grad_(False)
    stage_four_teacher.eval()
    device = resolve_device(args.device)
    model.to(device)
    stage_four_teacher.to(device)
    metadata = {
        "probe": "pointer_next_value_from_position",
        "device": str(device),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "output_directory": str(args.output_directory),
        "stage_four_checkpoint": str(args.stage_four_checkpoint),
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
                "probe": "pointer_next_value_from_position",
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
            stage_four_teacher=stage_four_teacher,
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
                "ema_aggregate": results.get("final_ema_aggregate"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
