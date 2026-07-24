"""Retrieve the token stored at an autoregressively computed position."""

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
from torch import Tensor, nn

from .data import PointerNextBatch, make_pointer_value_batch, sample_length
from .evaluate import resolve_device
from .evaluation import autocast_context
from .model import ModelConfig
from .pointer_position_probe import aggregate_length_ranges
from .pointer_position_sequence import (
    ModularPositionSequenceModel,
    PositionSequenceConfig,
    add_gradient_noise,
    generated_metrics,
    learning_rate_at_step,
    selected_evaluation_lengths,
    sequence_loss_and_metrics,
)
from .positions import sample_position_offsets
from .tokens import PointerNextVocabulary


@dataclass(frozen=True)
class PositionValueConfig(PositionSequenceConfig):
    token_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.token_loss_weight <= 0:
            raise ValueError("token_loss_weight must be positive")


class ModularPositionValueModel(ModularPositionSequenceModel):
    """Generate PTR, PTR + 1, then copy the token stored at PTR + 1."""

    def __init__(
        self,
        model_config: ModelConfig,
        position_moduli: tuple[int, ...],
        *,
        split_input: bool = True,
    ) -> None:
        super().__init__(
            model_config,
            position_moduli,
            split_input=split_input,
        )
        token_dim = self.encoder.token_embedding.embedding_dim
        self.token_query_projection = nn.Linear(
            model_config.d_model,
            token_dim,
            bias=False,
        )
        nn.init.normal_(
            self.token_query_projection.weight,
            mean=0.0,
            std=0.02,
        )

    def token_logits(self, hidden: Tensor) -> Tensor:
        query = self.token_query_projection(hidden)
        return F.linear(query, self.encoder.token_embedding.weight)

    def teacher_forced_token_logits(
        self,
        prompt_ids: Tensor,
        position_targets: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        hidden = self.hidden_states(
            prompt_ids,
            position_targets,
            offsets=offsets,
        )
        return self.token_logits(hidden[:, -1])

    @torch.inference_mode()
    def generate_trace(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        positions = self.generate_positions(prompt_ids, offsets=offsets)
        hidden = self.hidden_states(
            prompt_ids,
            positions,
            offsets=offsets,
        )
        tokens = self.token_logits(hidden[:, -1]).argmax(dim=-1)
        return positions, tokens


def target_token_ids(batch: PointerNextBatch) -> Tensor:
    """Return the marked value token immediately after the prompt separator."""

    return batch.token_ids[:, batch.prompt_length]


def load_stage_two_checkpoint(
    model: ModularPositionValueModel,
    checkpoint_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state")
    if not isinstance(source_state, dict):
        raise ValueError("stage-two checkpoint is missing model_state")
    missing, unexpected = model.load_state_dict(source_state, strict=False)
    expected_missing = {"token_query_projection.weight"}
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            "stage-two architecture mismatch: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "stage_two_step": int(checkpoint.get("step", 0)),
        "transferred_tensors": len(source_state),
    }


def position_value_loss_and_metrics(
    model: ModularPositionValueModel,
    batch: PointerNextBatch,
    offsets: Tensor,
    *,
    config: PositionValueConfig,
    isolate_successor: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    position_loss, position_metrics = sequence_loss_and_metrics(
        model,
        batch.prompt_ids,
        batch.pointers,
        offsets,
        isolate_successor=isolate_successor,
    )
    position_targets = model.target_sequence(batch.pointers, offsets)
    token_targets = target_token_ids(batch)
    token_logits = model.teacher_forced_token_logits(
        batch.prompt_ids,
        position_targets,
        offsets=offsets,
    )
    token_loss = F.cross_entropy(token_logits, token_targets)
    token_predictions = token_logits.argmax(dim=-1)
    total_loss = position_loss + config.token_loss_weight * token_loss
    return total_loss, {
        **position_metrics,
        "loss": float(total_loss.detach().item()),
        "token_loss": float(token_loss.detach().item()),
        "teacher_forced_token_accuracy": float(
            token_predictions.eq(token_targets).float().mean().item()
        ),
    }


def generated_trace_metrics(
    generated_positions: Tensor,
    generated_tokens: Tensor,
    target_positions: Tensor,
    target_tokens: Tensor,
    *,
    moduli: tuple[int, ...],
) -> dict[str, float]:
    metrics = generated_metrics(
        generated_positions,
        target_positions,
        moduli=moduli,
    )
    position_correct = (
        generated_positions.eq(target_positions).flatten(1).all(dim=1)
    )
    token_correct = generated_tokens.eq(target_tokens)
    correct_positions = int(position_correct.sum().item())
    metrics.update(
        {
            "token_accuracy": float(token_correct.float().mean().item()),
            "complete_trace_accuracy": float(
                (position_correct & token_correct).float().mean().item()
            ),
            "token_accuracy_given_correct_positions": (
                float(
                    (position_correct & token_correct).sum().item()
                    / correct_positions
                )
                if correct_positions
                else 0.0
            ),
            "correct_position_fraction": float(
                position_correct.float().mean().item()
            ),
        }
    )
    return metrics


@torch.inference_mode()
def evaluate_lengths(
    model: ModularPositionValueModel,
    vocabulary: PointerNextVocabulary,
    lengths: list[int],
    *,
    config: PositionValueConfig,
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
            batch = make_pointer_value_batch(
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
            token_targets = target_token_ids(batch)
            with autocast_context(device):
                generated_positions, generated_tokens = model.generate_trace(
                    batch.prompt_ids,
                    offsets=offsets,
                )
                _, teacher_forced = position_value_loss_and_metrics(
                    model,
                    batch,
                    offsets,
                    config=config,
                )
            metrics = {
                **generated_trace_metrics(
                    generated_positions,
                    generated_tokens,
                    position_targets,
                    token_targets,
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


def save_checkpoint(
    path: Path,
    *,
    model: ModularPositionValueModel,
    optimizer: torch.optim.Optimizer,
    config: PositionValueConfig,
    step: int,
    generator: torch.Generator,
) -> None:
    torch.save(
        {
            "probe": "pointer_value_from_position",
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
    model: ModularPositionValueModel,
    config: PositionValueConfig,
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
        batch = make_pointer_value_batch(
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
            loss, metrics = position_value_loss_and_metrics(
                model,
                batch,
                offsets,
                config=config,
                isolate_successor=isolate_successor,
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
                        "evaluation_token_accuracy": {
                            str(length): metrics["token_accuracy"]
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
        "probe": "pointer_value_from_position",
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
    parser.add_argument("--stage-two-checkpoint", type=Path, required=True)
    parser.add_argument("--train-min-length", type=int, default=2)
    parser.add_argument("--train-max-length", type=int, default=20)
    parser.add_argument("--eval-max-length", type=int, default=400)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-noise-scale", type=float, default=0.0)
    parser.add_argument("--gradient-noise-decay", type=float, default=0.25)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
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
    checkpoint = torch.load(args.stage_two_checkpoint, map_location="cpu")
    checkpoint_model_config = checkpoint.get("model_config")
    checkpoint_train_config = checkpoint.get("train_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("stage-two checkpoint is missing model_config")
    if not isinstance(checkpoint_train_config, dict):
        raise ValueError("stage-two checkpoint is missing train_config")
    model_config_values = dict(checkpoint_model_config)
    if args.dropout is not None:
        model_config_values["dropout"] = args.dropout
    model_config = ModelConfig(**model_config_values)
    position_moduli = tuple(checkpoint_train_config["position_moduli"])
    config = PositionValueConfig(
        representation=model_config.representation,
        symbol_count=model_config.symbol_count,
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
    )
    vocabulary = PointerNextVocabulary(
        config.representation,
        config.symbol_count,
    )
    if model_config.vocab_size != vocabulary.size:
        raise ValueError("stage-two checkpoint vocabulary does not match")
    torch.manual_seed(config.seed)
    model = ModularPositionValueModel(
        model_config,
        config.position_moduli,
        split_input=config.input_layout == "split",
    )
    initialization = load_stage_two_checkpoint(
        model,
        args.stage_two_checkpoint,
    )
    device = resolve_device(args.device)
    model.to(device)
    metadata = {
        "probe": "pointer_value_from_position",
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_directory": str(args.output_directory),
        "stage_two_checkpoint": str(args.stage_two_checkpoint),
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
                "probe": "pointer_value_from_position",
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
