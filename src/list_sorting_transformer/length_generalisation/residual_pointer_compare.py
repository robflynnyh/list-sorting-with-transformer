"""Learn the pointer-comparison pipeline entirely inside the residual stream."""

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

from .compiled_pointer_compare import (
    DEFAULT_POSITION_MODULI,
    target_action_classes,
)
from ..core.data import PointerNextBatch, make_pointer_pair_batch
from ..core.evaluate import resolve_device
from ..core.model import ModelConfig, SplitInputDecoderTransformer
from ..core.positions import ModularPositionEmbedding, sample_position_offsets
from ..core.tokens import PointerCompareVocabulary, VALUE_OFFSET


@dataclass(frozen=True)
class ResidualPointerCompareConfig:
    train_min_length: int = 2
    train_max_length: int = 20
    steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.001
    gradient_clip: float = 1.0
    dropout: float = 0.02
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    address_auxiliary_weight: float = 0.25
    value_auxiliary_weight: float = 0.25
    routing_auxiliary_weight: float = 0.1
    routing_top_k: int | None = None
    routing_top_k_straight_through: bool = False
    log_interval: int = 100
    eval_interval: int = 1_000
    eval_examples: int = 256
    checkpoint_interval: int = 1_000
    seed: int = 7

    def __post_init__(self) -> None:
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        integer_fields = (
            self.steps,
            self.batch_size,
            self.warmup_steps,
            self.log_interval,
            self.eval_interval,
            self.eval_examples,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("integer training settings must be positive")
        if not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must not exceed steps")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("learning rates are invalid")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum learning rate exceeds learning rate")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("optimizer settings are invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.position_offset_min > self.position_offset_max:
            raise ValueError("position offset bounds are reversed")
        auxiliary_weights = (
            self.address_auxiliary_weight,
            self.value_auxiliary_weight,
            self.routing_auxiliary_weight,
        )
        if any(weight < 0 for weight in auxiliary_weights):
            raise ValueError("auxiliary weights must be nonnegative")
        if self.routing_top_k is not None and self.routing_top_k < 1:
            raise ValueError("routing_top_k must be positive")
        if self.routing_top_k_straight_through and self.routing_top_k is None:
            raise ValueError("straight-through routing requires routing_top_k")


class ResidualPointerCompareModel(nn.Module):
    """Predict all intermediate states and the action in one forward pass."""

    def __init__(
        self,
        model_config: ModelConfig,
        position_moduli: tuple[int, ...],
        *,
        routing_top_k: int | None = None,
        routing_top_k_straight_through: bool = False,
    ) -> None:
        super().__init__()
        if model_config.n_layers < 3:
            raise ValueError("the residual circuit requires at least three blocks")
        if model_config.n_heads < 2:
            raise ValueError("the residual circuit requires at least two heads")
        self.encoder = SplitInputDecoderTransformer(
            model_config,
            content_dim=model_config.d_model // 2,
        )
        self.position_embedding = ModularPositionEmbedding(
            self.encoder.position_dim,
            position_moduli,
        )
        self.encoder.blocks[1].attention.configure_top_k(
            routing_top_k,
            straight_through=routing_top_k_straight_through,
        )
        self.pointer_address_heads = _residue_heads(
            model_config.d_model,
            position_moduli,
        )
        self.marked_address_heads = _residue_heads(
            model_config.d_model,
            position_moduli,
        )
        self.following_address_heads = _residue_heads(
            model_config.d_model,
            position_moduli,
        )
        self.marked_value_head = nn.Linear(
            model_config.d_model,
            model_config.symbol_count,
        )
        self.following_value_head = nn.Linear(
            model_config.d_model,
            model_config.symbol_count,
        )
        self.action_head = nn.Linear(model_config.d_model, 2)

    def input_embeddings(self, prompt_ids: Tensor, offsets: Tensor) -> Tensor:
        token_offsets = torch.arange(
            prompt_ids.shape[1],
            device=prompt_ids.device,
        )
        positions = self.position_embedding(
            offsets[:, None] + token_offsets[None, :]
        )
        return torch.cat((self.encoder.embed(prompt_ids), positions), dim=-1)

    def forward(self, prompt_ids: Tensor, *, offsets: Tensor) -> dict[str, Any]:
        hidden = self.input_embeddings(prompt_ids, offsets)
        pointer_route_logits = (
            self.encoder.blocks[0].attention.query_key_logits(
                self.encoder.blocks[0].attention_norm(hidden),
                query_index=-1,
            )[:, 0]
        )
        hidden = self.encoder.blocks[0](hidden)
        final_after_pointer = hidden[:, -1]
        pointer_address_logits = _apply_residue_heads(
            self.pointer_address_heads,
            final_after_pointer,
        )
        marked_address_logits = _apply_residue_heads(
            self.marked_address_heads,
            final_after_pointer,
        )
        following_address_logits = _apply_residue_heads(
            self.following_address_heads,
            final_after_pointer,
        )

        marked_route_logits = (
            self.encoder.blocks[1].attention.query_key_logits(
                self.encoder.blocks[1].attention_norm(hidden),
                query_index=-1,
            )[:, 0]
        )
        following_route_logits = (
            self.encoder.blocks[1].attention.query_key_logits(
                self.encoder.blocks[1].attention_norm(hidden),
                query_index=-1,
            )[:, 1]
        )
        hidden = self.encoder.blocks[1](hidden)
        final_after_values = hidden[:, -1]
        marked_value_logits = self.marked_value_head(final_after_values)
        following_value_logits = self.following_value_head(
            final_after_values
        )

        for block in self.encoder.blocks[2:]:
            hidden = block(hidden)
        final_hidden = self.encoder.final_norm(hidden[:, -1])
        return {
            "pointer_route_logits": pointer_route_logits,
            "marked_route_logits": marked_route_logits,
            "following_route_logits": following_route_logits,
            "pointer_address_logits": pointer_address_logits,
            "marked_address_logits": marked_address_logits,
            "following_address_logits": following_address_logits,
            "marked_value_logits": marked_value_logits,
            "following_value_logits": following_value_logits,
            "action_logits": self.action_head(final_hidden),
        }


def _residue_heads(
    d_model: int,
    moduli: tuple[int, ...],
) -> nn.ModuleList:
    return nn.ModuleList(nn.Linear(d_model, modulus) for modulus in moduli)


def _apply_residue_heads(
    heads: nn.ModuleList,
    hidden: Tensor,
) -> tuple[Tensor, ...]:
    return tuple(head(hidden) for head in heads)


def _target_positions(
    batch: PointerNextBatch,
    offsets: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    pointer = offsets + 1 + 2 * batch.pointers
    return pointer, pointer + 1, pointer + 3


def _target_residues(
    positions: Tensor,
    moduli: tuple[int, ...],
) -> tuple[Tensor, ...]:
    return tuple(positions.remainder(modulus) for modulus in moduli)


def _residue_loss_and_accuracy(
    logits: tuple[Tensor, ...],
    positions: Tensor,
    moduli: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    targets = _target_residues(positions, moduli)
    losses = [
        F.cross_entropy(component_logits, target)
        for component_logits, target in zip(logits, targets)
    ]
    components_correct = torch.stack(
        [
            component_logits.argmax(-1).eq(target)
            for component_logits, target in zip(logits, targets)
        ],
        dim=1,
    )
    return (
        torch.stack(losses).mean(),
        components_correct.all(dim=1).float().mean(),
        components_correct.float().mean(),
    )


def residual_pointer_compare_loss(
    model: ResidualPointerCompareModel,
    batch: PointerNextBatch,
    offsets: Tensor,
    config: ResidualPointerCompareConfig,
) -> tuple[Tensor, dict[str, float]]:
    outputs = model(batch.prompt_ids, offsets=offsets)
    pointer_position, marked_position, following_position = _target_positions(
        batch,
        offsets,
    )
    (
        pointer_address_loss,
        pointer_address_accuracy,
        pointer_address_component_accuracy,
    ) = (
        _residue_loss_and_accuracy(
            outputs["pointer_address_logits"],
            pointer_position,
            config.position_moduli,
        )
    )
    (
        marked_address_loss,
        marked_address_accuracy,
        marked_address_component_accuracy,
    ) = (
        _residue_loss_and_accuracy(
            outputs["marked_address_logits"],
            marked_position,
            config.position_moduli,
        )
    )
    (
        following_address_loss,
        following_address_accuracy,
        following_address_component_accuracy,
    ) = (
        _residue_loss_and_accuracy(
            outputs["following_address_logits"],
            following_position,
            config.position_moduli,
        )
    )

    rows = torch.arange(batch.values.shape[0], device=batch.values.device)
    marked_values = batch.values[rows, batch.pointers]
    following_values = batch.values[rows, batch.pointers + 1]
    marked_value_loss = F.cross_entropy(
        outputs["marked_value_logits"],
        marked_values,
    )
    following_value_loss = F.cross_entropy(
        outputs["following_value_logits"],
        following_values,
    )
    action_targets = target_action_classes(batch)
    action_loss = F.cross_entropy(outputs["action_logits"], action_targets)

    pointer_token_offsets = 1 + 2 * batch.pointers
    pointer_route_loss = F.cross_entropy(
        outputs["pointer_route_logits"],
        pointer_token_offsets,
    )
    marked_route_loss = F.cross_entropy(
        outputs["marked_route_logits"],
        pointer_token_offsets + 1,
    )
    following_route_loss = F.cross_entropy(
        outputs["following_route_logits"],
        pointer_token_offsets + 3,
    )
    routing_loss = torch.stack(
        (pointer_route_loss, marked_route_loss, following_route_loss)
    ).mean()
    address_loss = torch.stack(
        (
            pointer_address_loss,
            marked_address_loss,
            following_address_loss,
        )
    ).sum()
    value_loss = marked_value_loss + following_value_loss
    loss = (
        action_loss
        + config.address_auxiliary_weight * address_loss
        + config.value_auxiliary_weight * value_loss
        + config.routing_auxiliary_weight * routing_loss
    )

    metrics = {
        "loss": float(loss.detach()),
        "action_loss": float(action_loss.detach()),
        "action_accuracy": float(
            outputs["action_logits"]
            .argmax(-1)
            .eq(action_targets)
            .float()
            .mean()
        ),
        "pointer_address_loss": float(pointer_address_loss.detach()),
        "pointer_address_accuracy": float(pointer_address_accuracy),
        "pointer_address_component_accuracy": float(
            pointer_address_component_accuracy
        ),
        "marked_address_loss": float(marked_address_loss.detach()),
        "marked_address_accuracy": float(marked_address_accuracy),
        "marked_address_component_accuracy": float(
            marked_address_component_accuracy
        ),
        "following_address_loss": float(following_address_loss.detach()),
        "following_address_accuracy": float(following_address_accuracy),
        "following_address_component_accuracy": float(
            following_address_component_accuracy
        ),
        "marked_value_loss": float(marked_value_loss.detach()),
        "marked_value_accuracy": float(
            outputs["marked_value_logits"]
            .argmax(-1)
            .eq(marked_values)
            .float()
            .mean()
        ),
        "following_value_loss": float(following_value_loss.detach()),
        "following_value_accuracy": float(
            outputs["following_value_logits"]
            .argmax(-1)
            .eq(following_values)
            .float()
            .mean()
        ),
        "routing_loss": float(routing_loss.detach()),
        "pointer_route_accuracy": float(
            outputs["pointer_route_logits"]
            .argmax(-1)
            .eq(pointer_token_offsets)
            .float()
            .mean()
        ),
        "marked_route_accuracy": float(
            outputs["marked_route_logits"]
            .argmax(-1)
            .eq(pointer_token_offsets + 1)
            .float()
            .mean()
        ),
        "following_route_accuracy": float(
            outputs["following_route_logits"]
            .argmax(-1)
            .eq(pointer_token_offsets + 3)
            .float()
            .mean()
        ),
    }
    return loss, metrics


def learning_rate_at_step(
    config: ResidualPointerCompareConfig,
    step: int,
) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        config.steps - config.warmup_steps,
        1,
    )
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


@torch.inference_mode()
def evaluate_model(
    model: ResidualPointerCompareModel,
    vocabulary: PointerCompareVocabulary,
    *,
    lengths: tuple[int, ...],
    examples: int,
    batch_size: int,
    seed: int,
    config: ResidualPointerCompareConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    results: dict[str, dict[str, float]] = {}
    for length in lengths:
        totals: dict[str, float] = {}
        completed = 0
        while completed < examples:
            current_batch_size = min(batch_size, examples - completed)
            batch = make_pointer_pair_batch(
                current_batch_size,
                length,
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
            offsets = sample_position_offsets(
                current_batch_size,
                minimum=config.position_offset_min,
                maximum=config.position_offset_max,
                generator=generator,
                device=device,
            )
            _, metrics = residual_pointer_compare_loss(
                model,
                batch,
                offsets,
                config,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * current_batch_size
            completed += current_batch_size
        results[str(length)] = {
            name: value / examples for name, value in totals.items()
        }
    model.train(was_training)
    return results


def _tracker_log(tracker: Any, metrics: dict[str, float], step: int) -> None:
    if tracker is not None:
        tracker.log(metrics, step=step)


def train(
    model: ResidualPointerCompareModel,
    vocabulary: PointerCompareVocabulary,
    *,
    config: ResidualPointerCompareConfig,
    output_directory: Path,
    device: torch.device,
    tracker: Any = None,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    started_at = time.monotonic()
    history: list[dict[str, float]] = []
    evaluations: list[dict[str, Any]] = []
    selected_lengths = (2, 11, 20, 40, 400)
    model.train()

    for step in range(1, config.steps + 1):
        length = random.Random(config.seed + step).randint(
            config.train_min_length,
            config.train_max_length,
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
        learning_rate = learning_rate_at_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = residual_pointer_compare_loss(
            model,
            batch,
            offsets,
            config,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()

        if step % config.log_interval == 0 or step == 1:
            record = {
                "step": float(step),
                "length": float(length),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started_at,
                **metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            _tracker_log(
                tracker,
                {f"train/{name}": value for name, value in record.items()},
                step,
            )

        if step % config.eval_interval == 0:
            per_length = evaluate_model(
                model,
                vocabulary,
                lengths=selected_lengths,
                examples=config.eval_examples,
                batch_size=min(32, config.eval_examples),
                seed=config.seed + 10_000 + step,
                config=config,
                device=device,
            )
            evaluation = {"step": step, "per_length": per_length}
            evaluations.append(evaluation)
            print(json.dumps(evaluation), flush=True)
            flat_metrics = {
                f"eval/length_{length}/{name}": value
                for length, length_metrics in per_length.items()
                for name, value in length_metrics.items()
            }
            _tracker_log(tracker, flat_metrics, step)

        if step % config.checkpoint_interval == 0 or step == config.steps:
            torch.save(
                {
                    "experiment": "residual_pointer_compare",
                    "step": step,
                    "model_config": model.encoder.config.as_dict(),
                    "train_config": asdict(config),
                    "model_state": model.state_dict(),
                },
                output_directory / f"checkpoint_step_{step}.pt",
            )

    final_per_length = evaluate_model(
        model,
        vocabulary,
        lengths=selected_lengths,
        examples=config.eval_examples,
        batch_size=min(32, config.eval_examples),
        seed=config.seed + 30_000,
        config=config,
        device=device,
    )
    results = {
        "experiment": "residual_pointer_compare",
        "model_config": model.encoder.config.as_dict(),
        "train_config": asdict(config),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "backbone_parameter_count": sum(
            parameter.numel() for parameter in model.encoder.parameters()
        )
        + sum(
            parameter.numel()
            for parameter in model.position_embedding.parameters()
        ),
        "wall_time_seconds": time.monotonic() - started_at,
        "history": history,
        "intermediate_evaluations": evaluations,
        "final_per_length": final_per_length,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--address-auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--value-auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--routing-auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--routing-top-k", type=int)
    parser.add_argument(
        "--routing-top-k-straight-through",
        action="store_true",
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = ResidualPointerCompareConfig(
        train_min_length=args.train_min_length,
        train_max_length=args.train_max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        dropout=args.dropout,
        address_auxiliary_weight=args.address_auxiliary_weight,
        value_auxiliary_weight=args.value_auxiliary_weight,
        routing_auxiliary_weight=args.routing_auxiliary_weight,
        routing_top_k=args.routing_top_k,
        routing_top_k_straight_through=args.routing_top_k_straight_through,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_examples=args.eval_examples,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model_config = ModelConfig(
        vocab_size=vocabulary.size,
        symbol_count=10,
        representation="numbers",
        d_model=128,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=4.0,
        dropout=config.dropout,
        position_pattern="none",
    )
    model = ResidualPointerCompareModel(
        model_config,
        config.position_moduli,
        routing_top_k=config.routing_top_k,
        routing_top_k_straight_through=config.routing_top_k_straight_through,
    )
    device = resolve_device(args.device)
    model.to(device)
    metadata = {
        "experiment": "residual_pointer_compare",
        "device": str(device),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "backbone_parameter_count": sum(
            parameter.numel() for parameter in model.encoder.parameters()
        )
        + sum(
            parameter.numel()
            for parameter in model.position_embedding.parameters()
        ),
        "output_directory": str(args.output_directory),
        "config": asdict(config),
    }
    print(json.dumps(metadata), flush=True)

    tracker = None
    if args.wandb_project is not None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "install the tracking extra to use W&B"
            ) from error
        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={**asdict(config), **model_config.as_dict()},
        )
        print(json.dumps({"wandb_url": tracker.url}), flush=True)
    try:
        train(
            model,
            vocabulary,
            config=config,
            output_directory=args.output_directory,
            device=device,
            tracker=tracker,
        )
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
