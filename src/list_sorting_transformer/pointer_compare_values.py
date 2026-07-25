"""Compare two autoregressively retrieved values and emit KEEP or SWAP."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import PointerNextBatch, make_pointer_pair_batch, sample_length
from .evaluate import resolve_device
from .evaluation import autocast_context
from .model import ModelConfig, SplitInputDecoderTransformer
from .pointer_next_value_from_position import (
    ModularNextValueFromPositionModel,
    NextValueFromPositionConfig,
    generated_stage_five_metrics,
    next_value_token_loss_and_metrics,
    stage_five_learning_rate_at_step,
    target_next_value_token_ids,
)
from .pointer_next_value_position import relative_logit_distillation_loss
from .pointer_position_probe import aggregate_length_ranges
from .pointer_position_sequence import (
    add_gradient_noise,
    selected_evaluation_lengths,
)
from .pointer_value_from_position import target_token_ids
from .positions import sample_position_offsets
from .tokens import (
    POINTER_COMPARE_ACTIONS,
    PointerCompareVocabulary,
)


@dataclass(frozen=True)
class PointerCompareConfig(NextValueFromPositionConfig):
    action_loss_weight: float = 1.0
    action_attention_isolation_probability: float = 0.5
    action_consistency_weight: float = 1.0
    stage_five_distillation_weight: float = 1.0
    stage_five_parameter_anchor_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_loss_weight <= 0:
            raise ValueError("action_loss_weight must be positive")
        if self.action_consistency_weight < 0:
            raise ValueError("action_consistency_weight must be nonnegative")
        if self.stage_five_distillation_weight < 0:
            raise ValueError(
                "stage_five_distillation_weight must be nonnegative"
            )
        if self.stage_five_parameter_anchor_weight < 0:
            raise ValueError(
                "stage_five_parameter_anchor_weight must be nonnegative"
            )
        if not 0 <= self.action_attention_isolation_probability <= 1:
            raise ValueError(
                "action_attention_isolation_probability must be in [0, 1]"
            )


class ModularPointerCompareModel(ModularNextValueFromPositionModel):
    """Generate the Stage-5 trace, then compare its two retrieved values."""

    def __init__(
        self,
        model_config: ModelConfig,
        position_moduli: tuple[int, ...],
        *,
        action_token_offset: int,
        split_input: bool = True,
    ) -> None:
        super().__init__(
            model_config,
            position_moduli,
            split_input=split_input,
        )
        self.action_token_offset = action_token_offset
        if (
            action_token_offset + len(POINTER_COMPARE_ACTIONS)
            != model_config.vocab_size
        ):
            raise ValueError("comparison actions must end the vocabulary")

    def stage_six_hidden_states(
        self,
        prompt_ids: Tensor,
        position_history: Tensor,
        marked_token_history: Tensor,
        next_position_history: Tensor,
        next_token_history: Tensor,
        *,
        offsets: Tensor,
        isolate_action: Tensor | None = None,
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
        if marked_token_history.shape != (batch_size, 1):
            raise ValueError(
                "marked_token_history must be [batch, one token]"
            )
        if next_position_history.shape != (
            batch_size,
            1,
            len(self.moduli),
        ):
            raise ValueError(
                "next_position_history must be [batch, one position, moduli]"
            )
        if next_token_history.shape != (batch_size, 1):
            raise ValueError("next_token_history must be [batch, one token]")

        stream_length = prompt_length + 5
        stream_positions = (
            offsets[:, None]
            + torch.arange(stream_length, device=prompt_ids.device)[None, :]
        )
        position_embeddings = self.position_embedding(stream_positions)
        prompt_content = self.encoder.embed(prompt_ids)
        address_content = self.history_embeddings(position_history)
        marked_token_content = self.encoder.embed(marked_token_history)
        next_address_content = self.history_embeddings(next_position_history)
        next_token_content = self.encoder.embed(next_token_history)
        content_embeddings = torch.cat(
            (
                prompt_content,
                address_content,
                marked_token_content,
                next_address_content,
                next_token_content,
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
            hidden[:, prompt_length + 3 : prompt_length + 4] = (
                next_address_content
            )
        attention_mask = self.action_attention_mask(
            batch_size=batch_size,
            prompt_length=prompt_length,
            stream_length=stream_length,
            isolate_action=isolate_action,
            device=prompt_ids.device,
        )
        for block in self.encoder.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return self.encoder.final_norm(hidden)

    @staticmethod
    def action_attention_mask(
        *,
        batch_size: int,
        prompt_length: int,
        stream_length: int,
        isolate_action: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if isolate_action is None:
            return None
        if isolate_action.shape != (batch_size,):
            raise ValueError("isolate_action must have shape [batch]")
        if isolate_action.dtype != torch.bool:
            raise ValueError("isolate_action must be boolean")
        isolate_action = isolate_action.to(device=device)
        if not bool(isolate_action.any()):
            return None
        mask = torch.ones(
            batch_size,
            stream_length,
            stream_length,
            device=device,
            dtype=torch.bool,
        )
        mask[isolate_action, -1, :] = False
        mask[isolate_action, -1, prompt_length + 2] = True
        mask[isolate_action, -1, -1] = True
        return mask

    def action_query(
        self,
        prompt_ids: Tensor,
        position_history: Tensor,
        marked_token_history: Tensor,
        next_position_history: Tensor,
        next_token_history: Tensor,
        *,
        offsets: Tensor,
        isolate_action: Tensor | None = None,
    ) -> Tensor:
        hidden = self.stage_six_hidden_states(
            prompt_ids,
            position_history,
            marked_token_history,
            next_position_history,
            next_token_history,
            offsets=offsets,
            isolate_action=isolate_action,
        )
        return self.token_query_projection(hidden[:, -1])

    def action_logits_from_query(self, query: Tensor) -> Tensor:
        action_embeddings = self.encoder.token_embedding.weight[
            self.action_token_offset :
        ]
        return F.linear(query, action_embeddings)

    def teacher_forced_action_query(
        self,
        batch: PointerNextBatch,
        offsets: Tensor,
        *,
        isolate_action: Tensor | None = None,
    ) -> Tensor:
        positions = self.target_sequence(batch.pointers, offsets)
        marked_tokens = target_token_ids(batch)
        next_positions = self.target_next_value_position(
            batch.pointers,
            offsets,
        )
        next_tokens = target_next_value_token_ids(batch)
        return self.action_query(
            batch.prompt_ids,
            positions,
            marked_tokens[:, None],
            next_positions[:, None],
            next_tokens[:, None],
            offsets=offsets,
            isolate_action=isolate_action,
        )

    @torch.inference_mode()
    def generate_stage_six_trace(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            positions,
            marked_tokens,
            next_positions,
            next_tokens,
        ) = self.generate_stage_five_trace(
            prompt_ids,
            offsets=offsets,
        )
        query = self.action_query(
            prompt_ids,
            positions,
            marked_tokens[:, None],
            next_positions[:, None],
            next_tokens[:, None],
            offsets=offsets,
        )
        action_classes = self.action_logits_from_query(query).argmax(dim=-1)
        action_tokens = action_classes + self.action_token_offset
        return (
            positions,
            marked_tokens,
            next_positions,
            next_tokens,
            action_tokens,
        )


def target_action_classes(batch: PointerNextBatch) -> Tensor:
    """Return KEEP for ordered pairs and SWAP for descending pairs."""

    row_indices = torch.arange(
        batch.values.shape[0],
        device=batch.values.device,
    )
    marked = batch.values[row_indices, batch.pointers]
    following = batch.values[row_indices, batch.pointers + 1]
    return marked.gt(following).long()


def target_action_token_ids(
    batch: PointerNextBatch,
    vocabulary: PointerCompareVocabulary,
) -> Tensor:
    return target_action_classes(batch) + vocabulary.action_token_offset


def load_stage_five_checkpoint(
    model: ModularPointerCompareModel,
    checkpoint_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("probe") != "pointer_next_value_from_position":
        raise ValueError("checkpoint is not a Stage-5 next-value model")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("stage-five checkpoint is missing model_state")
    target_state = model.state_dict()
    embedding_name = "encoder.token_embedding.weight"
    for name, source in source_state.items():
        if name not in target_state:
            raise ValueError(f"unexpected Stage-5 tensor: {name}")
        target = target_state[name]
        if name == embedding_name:
            if (
                source.shape[1:] != target.shape[1:]
                or source.shape[0] != model.action_token_offset
            ):
                raise ValueError("Stage-5 token embedding shape mismatch")
            target[: source.shape[0]].copy_(source)
        elif source.shape != target.shape:
            raise ValueError(f"Stage-5 tensor shape mismatch: {name}")
        else:
            target.copy_(source)
    missing = set(target_state) - set(source_state)
    if missing:
        raise ValueError(f"Stage-5 checkpoint is missing tensors: {missing}")
    model.load_state_dict(target_state, strict=True)
    return {
        "stage_five_step": int(checkpoint.get("step", 0)),
        "legacy_vocab_size": int(source_state[embedding_name].shape[0]),
        "transferred_tensors": len(source_state),
    }


def stage_five_distillation_loss(
    model: ModularPointerCompareModel,
    teacher: ModularNextValueFromPositionModel,
    batch: PointerNextBatch,
    offsets: Tensor,
) -> Tensor:
    positions = model.target_sequence(batch.pointers, offsets)
    marked_tokens = target_token_ids(batch)
    next_positions = model.target_next_value_position(
        batch.pointers,
        offsets,
    )
    legacy_vocab_size = teacher.encoder.config.vocab_size

    student_position_logits = model.teacher_forced_logits(
        batch.prompt_ids,
        positions,
        offsets=offsets,
    )
    student_marked_logits = model.teacher_forced_token_logits(
        batch.prompt_ids,
        positions,
        offsets=offsets,
    )[:, :legacy_vocab_size]
    student_next_position_logits = (
        model.teacher_forced_next_value_position_logits(
            batch.prompt_ids,
            positions,
            marked_tokens,
            offsets=offsets,
        )
    )
    student_next_token_logits = (
        model.teacher_forced_next_value_token_logits(
            batch.prompt_ids,
            positions,
            marked_tokens,
            next_positions,
            offsets=offsets,
        )[:, :legacy_vocab_size]
    )
    with torch.no_grad():
        teacher_position_logits = teacher.teacher_forced_logits(
            batch.prompt_ids,
            positions,
            offsets=offsets,
        )
        teacher_marked_logits = teacher.teacher_forced_token_logits(
            batch.prompt_ids,
            positions,
            offsets=offsets,
        )
        teacher_next_position_logits = (
            teacher.teacher_forced_next_value_position_logits(
                batch.prompt_ids,
                positions,
                marked_tokens,
                offsets=offsets,
            )
        )
        teacher_next_token_logits = (
            teacher.teacher_forced_next_value_token_logits(
                batch.prompt_ids,
                positions,
                marked_tokens,
                next_positions,
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
            student_marked_logits,
            teacher_marked_logits,
        )
    )
    losses.extend(
        relative_logit_distillation_loss(student, source)
        for student, source in zip(
            student_next_position_logits,
            teacher_next_position_logits,
        )
    )
    losses.append(
        relative_logit_distillation_loss(
            student_next_token_logits,
            teacher_next_token_logits,
        )
    )
    return torch.stack(losses).mean()


def stage_five_parameter_anchor_loss(
    model: ModularPointerCompareModel,
    teacher: ModularNextValueFromPositionModel,
) -> Tensor:
    """Squared L2 distance from all inherited Stage-5 parameters."""

    teacher_parameters = dict(teacher.named_parameters())
    loss = next(model.parameters()).new_zeros(())
    embedding_name = "encoder.token_embedding.weight"
    for name, parameter in model.named_parameters():
        source = teacher_parameters.get(name)
        if source is None:
            raise ValueError(f"Stage-5 teacher is missing parameter: {name}")
        if name == embedding_name:
            inherited = parameter[: source.shape[0]]
            if inherited.shape != source.shape:
                raise ValueError("Stage-5 token embedding shape mismatch")
        else:
            inherited = parameter
            if inherited.shape != source.shape:
                raise ValueError(
                    f"Stage-5 parameter shape mismatch: {name}"
                )
        loss = loss + (inherited - source.detach()).square().sum()
    return loss


def pointer_compare_loss_and_metrics(
    model: ModularPointerCompareModel,
    batch: PointerNextBatch,
    offsets: Tensor,
    *,
    config: PointerCompareConfig,
    isolate_successor: Tensor | None = None,
    isolate_next_value_position: Tensor | None = None,
    isolate_action: Tensor | None = None,
    stage_five_teacher: ModularNextValueFromPositionModel | None = None,
) -> tuple[Tensor, dict[str, float]]:
    inherited_loss, metrics = next_value_token_loss_and_metrics(
        model,
        batch,
        offsets,
        config=config,
        isolate_successor=isolate_successor,
        isolate_next_value_position=isolate_next_value_position,
    )
    action_targets = target_action_classes(batch)
    student_query = model.teacher_forced_action_query(batch, offsets)
    student_logits = model.action_logits_from_query(student_query)
    student_loss = F.cross_entropy(student_logits, action_targets)
    student_predictions = student_logits.argmax(dim=-1)

    masked_loss = student_loss
    masked_predictions = student_predictions
    consistency_loss = student_query.new_zeros(())
    if isolate_action is not None:
        masked_query = model.teacher_forced_action_query(
            batch,
            offsets,
            isolate_action=isolate_action,
        )
        masked_logits = model.action_logits_from_query(masked_query)
        masked_loss = F.cross_entropy(masked_logits, action_targets)
        masked_predictions = masked_logits.argmax(dim=-1)
        if bool(isolate_action.any()):
            masked_target = masked_query[isolate_action].detach()
            student_target = student_query[isolate_action]
            masked_rms = (
                masked_target.square()
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .clamp_min(1e-6)
            )
            consistency_loss = F.mse_loss(
                student_target / masked_rms,
                masked_target / masked_rms,
            )
    action_loss = 0.5 * (student_loss + masked_loss)
    distillation_loss = student_query.new_zeros(())
    if stage_five_teacher is not None:
        distillation_loss = stage_five_distillation_loss(
            model,
            stage_five_teacher,
            batch,
            offsets,
        )
    parameter_anchor_loss = student_query.new_zeros(())
    if stage_five_teacher is not None:
        parameter_anchor_loss = stage_five_parameter_anchor_loss(
            model,
            stage_five_teacher,
        )
    total_loss = (
        inherited_loss
        + config.action_loss_weight * action_loss
        + config.action_consistency_weight * consistency_loss
        + config.stage_five_distillation_weight * distillation_loss
        + config.stage_five_parameter_anchor_weight
        * parameter_anchor_loss
    )
    return total_loss, {
        **metrics,
        "loss": float(total_loss.detach().item()),
        "stage_five_loss": float(inherited_loss.detach().item()),
        "action_loss": float(action_loss.detach().item()),
        "unrestricted_action_loss": float(student_loss.detach().item()),
        "masked_action_loss": float(masked_loss.detach().item()),
        "action_consistency_loss": float(consistency_loss.detach().item()),
        "stage_five_distillation_loss": float(
            distillation_loss.detach().item()
        ),
        "stage_five_parameter_anchor_loss": float(
            parameter_anchor_loss.detach().item()
        ),
        "action_attention_isolation_fraction": (
            float(isolate_action.float().mean().item())
            if isolate_action is not None
            else 0.0
        ),
        "teacher_forced_action_accuracy": float(
            student_predictions.eq(action_targets).float().mean().item()
        ),
        "masked_action_accuracy": float(
            masked_predictions.eq(action_targets).float().mean().item()
        ),
    }


def generated_stage_six_metrics(
    generated_positions: Tensor,
    generated_marked_tokens: Tensor,
    generated_next_positions: Tensor,
    generated_next_tokens: Tensor,
    generated_actions: Tensor,
    target_positions: Tensor,
    target_marked_tokens: Tensor,
    target_next_positions: Tensor,
    target_next_tokens: Tensor,
    target_actions: Tensor,
    *,
    moduli: tuple[int, ...],
) -> dict[str, float]:
    metrics = generated_stage_five_metrics(
        generated_positions,
        generated_marked_tokens,
        generated_next_positions,
        generated_next_tokens,
        target_positions,
        target_marked_tokens,
        target_next_positions,
        target_next_tokens,
        moduli=moduli,
    )
    stage_five_correct = (
        generated_positions.eq(target_positions).flatten(1).all(dim=1)
        & generated_marked_tokens.eq(target_marked_tokens)
        & generated_next_positions.eq(target_next_positions).all(dim=1)
        & generated_next_tokens.eq(target_next_tokens)
    )
    action_correct = generated_actions.eq(target_actions)
    stage_five_count = int(stage_five_correct.sum().item())
    metrics.update(
        {
            "stage_five_complete_trace_accuracy": metrics[
                "complete_trace_accuracy"
            ],
            "action_accuracy": float(
                action_correct.float().mean().item()
            ),
            "action_accuracy_given_stage_five_correct": (
                float(
                    (stage_five_correct & action_correct).sum().item()
                    / stage_five_count
                )
                if stage_five_count
                else 0.0
            ),
            "complete_trace_accuracy": float(
                (stage_five_correct & action_correct).float().mean().item()
            ),
        }
    )
    return metrics


@torch.inference_mode()
def evaluate_lengths(
    model: ModularPointerCompareModel,
    vocabulary: PointerCompareVocabulary,
    lengths: list[int],
    *,
    config: PointerCompareConfig,
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
            marked_targets = target_token_ids(batch)
            next_position_targets = model.target_next_value_position(
                batch.pointers,
                offsets,
            )
            next_token_targets = target_next_value_token_ids(batch)
            action_targets = target_action_token_ids(batch, vocabulary)
            with autocast_context(device):
                generated = model.generate_stage_six_trace(
                    batch.prompt_ids,
                    offsets=offsets,
                )
                _, teacher_forced = pointer_compare_loss_and_metrics(
                    model,
                    batch,
                    offsets,
                    config=config,
                )
            metrics = {
                **generated_stage_six_metrics(
                    *generated,
                    position_targets,
                    marked_targets,
                    next_position_targets,
                    next_token_targets,
                    action_targets,
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
    model: ModularPointerCompareModel,
    optimizer: torch.optim.Optimizer,
    config: PointerCompareConfig,
    step: int,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "probe": "pointer_compare_values",
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
    model: ModularPointerCompareModel,
    config: PointerCompareConfig,
    *,
    vocabulary: PointerCompareVocabulary,
    output_directory: Path,
    device: torch.device,
    tracker: Any | None = None,
    stage_five_teacher: ModularNextValueFromPositionModel | None = None,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(config.seed + 1)
    successor_isolation_generator = torch.Generator().manual_seed(
        config.seed + 2
    )
    next_position_isolation_generator = torch.Generator().manual_seed(
        config.seed + 3
    )
    action_isolation_generator = torch.Generator().manual_seed(
        config.seed + 4
    )
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
            torch.rand(
                config.batch_size,
                generator=successor_isolation_generator,
            )
            < config.successor_attention_isolation_probability
        ).to(device=device)
        isolate_next_value_position = (
            torch.rand(
                config.batch_size,
                generator=next_position_isolation_generator,
            )
            < config.next_value_position_attention_isolation_probability
        ).to(device=device)
        isolate_action = (
            torch.rand(
                config.batch_size,
                generator=action_isolation_generator,
            )
            < config.action_attention_isolation_probability
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            loss, metrics = pointer_compare_loss_and_metrics(
                model,
                batch,
                offsets,
                config=config,
                isolate_successor=isolate_successor,
                isolate_next_value_position=isolate_next_value_position,
                isolate_action=isolate_action,
                stage_five_teacher=stage_five_teacher,
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
                        "evaluation_complete_trace_accuracy": {
                            str(length): metrics["complete_trace_accuracy"]
                            for length, metrics in per_length.items()
                        },
                        "evaluation_action_accuracy": {
                            str(length): metrics["action_accuracy"]
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
    save_checkpoint(
        output_directory / "checkpoint.pt",
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
        "probe": "pointer_compare_values",
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
    (output_directory / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-five-checkpoint", type=Path, required=True)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=400)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--learning-rate-decay-start", type=int, default=1_000)
    parser.add_argument("--learning-rate-decay-end", type=int)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
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
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-attention-isolation-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--action-consistency-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stage-five-distillation-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stage-five-parameter-anchor-weight",
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
    checkpoint = torch.load(args.stage_five_checkpoint, map_location="cpu")
    checkpoint_model_config = checkpoint.get("model_config")
    checkpoint_train_config = checkpoint.get("train_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("stage-five checkpoint is missing model_config")
    if not isinstance(checkpoint_train_config, dict):
        raise ValueError("stage-five checkpoint is missing train_config")
    teacher_model_config = ModelConfig(**checkpoint_model_config)
    vocabulary = PointerCompareVocabulary(
        teacher_model_config.representation,
        teacher_model_config.symbol_count,
    )
    model_config_values = dict(checkpoint_model_config)
    model_config_values["vocab_size"] = vocabulary.size
    if args.dropout is not None:
        model_config_values["dropout"] = args.dropout
    model_config = ModelConfig(**model_config_values)
    position_moduli = tuple(checkpoint_train_config["position_moduli"])
    config = PointerCompareConfig(
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
        stage_four_distillation_weight=0.0,
        ema_decay=0.0,
        ema_start_step=0,
        action_loss_weight=args.action_loss_weight,
        action_attention_isolation_probability=(
            args.action_attention_isolation_probability
        ),
        action_consistency_weight=args.action_consistency_weight,
        stage_five_distillation_weight=(
            args.stage_five_distillation_weight
        ),
        stage_five_parameter_anchor_weight=(
            args.stage_five_parameter_anchor_weight
        ),
    )
    torch.manual_seed(config.seed)
    model = ModularPointerCompareModel(
        model_config,
        config.position_moduli,
        action_token_offset=vocabulary.action_token_offset,
        split_input=config.input_layout == "split",
    )
    initialization = load_stage_five_checkpoint(
        model,
        args.stage_five_checkpoint,
    )
    stage_five_teacher = ModularNextValueFromPositionModel(
        teacher_model_config,
        config.position_moduli,
        split_input=config.input_layout == "split",
    )
    teacher_missing, teacher_unexpected = stage_five_teacher.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    if teacher_missing or teacher_unexpected:
        raise ValueError("could not initialize the Stage-5 teacher")
    stage_five_teacher.requires_grad_(False)
    stage_five_teacher.eval()
    device = resolve_device(args.device)
    model.to(device)
    stage_five_teacher.to(device)
    metadata = {
        "probe": "pointer_compare_values",
        "device": str(device),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "output_directory": str(args.output_directory),
        "stage_five_checkpoint": str(args.stage_five_checkpoint),
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
                "probe": "pointer_compare_values",
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
            stage_five_teacher=stage_five_teacher,
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
