"""Persistent one-step router MAML on the random-position shortcut task."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from .evaluate import resolve_device
from .maml_length_generalization import (
    make_router,
    router_summary,
    second_order_attention_context,
)
from .shortcut_credit import (
    AttentionRoutingRule,
    ShortcutBatch,
    ShortcutDecoderTransformer,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_forward_model_config,
    make_shortcut_batch,
)


@dataclass(frozen=True)
class MAMLShortcutConfig:
    run_name: str = "random-shortcut-router-maml-seed7"
    output_dir: str = "artifacts/maml_shortcut"
    method: str = "router_maml"
    steps: int = 2_000
    batch_size: int = 64
    min_length: int = 8
    max_length: int = 32
    fitness_examples: int = 512
    fitness_batch_size: int = 32
    eval_examples: int = 256
    eval_batch_size: int = 64
    inner_learning_rate: float = 3e-4
    ordinary_learning_rate: float = 3e-4
    router_learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    router_d_model: int = 128
    router_heads: int = 4
    router_credit_mode: str = "suppress_renorm"
    router_initial_gate: float = 1e-3
    router_minimum_gate: float = 1e-6
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    seed: int = 7
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "list-sorting-maml-shortcut"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        positive_integers = (
            self.steps,
            self.batch_size,
            self.min_length,
            self.max_length,
            self.fitness_examples,
            self.fitness_batch_size,
            self.eval_examples,
            self.eval_batch_size,
            self.d_model,
            self.layers,
            self.heads,
            self.router_d_model,
            self.router_heads,
            self.log_interval,
            self.eval_interval,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("integer configuration values must be positive")
        if not 2 <= self.min_length <= self.max_length:
            raise ValueError("invalid length range")
        if self.method not in {"ordinary", "router_maml"}:
            raise ValueError("method must be ordinary or router_maml")
        if self.router_credit_mode != "suppress_renorm":
            raise ValueError(
                "the shortcut diagnostic initially requires suppress-only routing"
            )
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.router_d_model % self.router_heads:
            raise ValueError("router_d_model must be divisible by router_heads")
        if self.fitness_examples % (2 * self.fitness_batch_size):
            raise ValueError(
                "fitness examples must divide into balanced mode pairs"
            )
        if self.eval_examples % self.eval_batch_size:
            raise ValueError("eval examples must divide into full batches")
        if min(
            self.inner_learning_rate,
            self.ordinary_learning_rate,
            self.router_learning_rate,
            self.gradient_clip,
            self.router_initial_gate,
            self.router_minimum_gate,
        ) <= 0:
            raise ValueError("learning rates and gradient scales must be positive")


@dataclass(frozen=True)
class ShortcutMAMLObjective:
    short_loss: Tensor
    meta_loss: Tensor
    meta_accuracy: Tensor


def make_model(
    config: MAMLShortcutConfig,
    vocabulary: ShortcutPointerVocabulary,
    *,
    device: torch.device,
) -> ShortcutDecoderTransformer:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    return ShortcutDecoderTransformer(
        make_forward_model_config(
            vocabulary,
            d_model=config.d_model,
            n_layers=config.layers,
            n_heads=config.heads,
        )
    ).to(device)


def make_fitness_pairs(
    config: MAMLShortcutConfig,
    *,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
    seed_offset: int,
) -> tuple[tuple[ShortcutBatch, ShortcutBatch], ...]:
    """Build fixed balanced masked/incorrect pairs at sampled lengths."""

    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    pair_count = config.fitness_examples // (2 * config.fitness_batch_size)
    pairs = []
    for _ in range(pair_count):
        length = int(
            torch.randint(
                config.min_length,
                config.max_length + 1,
                (),
                generator=generator,
            )
        )
        pairs.append(
            (
                make_shortcut_batch(
                    config.fitness_batch_size,
                    length,
                    leak_mode="masked",
                    leak_placement="random_list",
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                ),
                make_shortcut_batch(
                    config.fitness_batch_size,
                    length,
                    leak_mode="incorrect",
                    leak_placement="random_list",
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                ),
            )
        )
    return tuple(pairs)


def make_evaluation_batches(
    config: MAMLShortcutConfig,
    *,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
    leak_mode: str,
    seed_offset: int,
) -> tuple[ShortcutBatch, ...]:
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    batches = []
    for _ in range(config.eval_examples // config.eval_batch_size):
        length = int(
            torch.randint(
                config.min_length,
                config.max_length + 1,
                (),
                generator=generator,
            )
        )
        batches.append(
            make_shortcut_batch(
                config.eval_batch_size,
                length,
                leak_mode=leak_mode,  # type: ignore[arg-type]
                leak_placement="random_list",
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
        )
    return tuple(batches)


def one_step_router_objective(
    model: ShortcutDecoderTransformer,
    router: AttentionRoutingRule,
    biased_batch: ShortcutBatch,
    fitness_pair: tuple[ShortcutBatch, ShortcutBatch],
    *,
    inner_learning_rate: float,
) -> ShortcutMAMLObjective:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    short_logits = model.forward_with_backward_rule(
        biased_batch.input_ids,
        router,
    )[:, -1]
    short_loss = F.cross_entropy(short_logits, biased_batch.targets)
    short_gradients = torch.autograd.grad(
        short_loss,
        tuple(parameters.values()),
        create_graph=True,
    )
    adapted_parameters = {
        name: parameter - inner_learning_rate * gradient
        for (name, parameter), gradient in zip(
            parameters.items(),
            short_gradients,
        )
    }

    losses = []
    correct = []
    for fitness_batch in fitness_pair:
        logits = functional_call(
            model,
            (adapted_parameters, buffers),
            (fitness_batch.input_ids,),
        )[:, -1]
        losses.append(F.cross_entropy(logits, fitness_batch.targets))
        correct.append(logits.argmax(dim=-1).eq(fitness_batch.targets))
    return ShortcutMAMLObjective(
        short_loss=short_loss,
        meta_loss=torch.stack(losses).mean(),
        meta_accuracy=torch.cat(correct).float().mean(),
    )


def batch_loss(
    model: ShortcutDecoderTransformer,
    batch: ShortcutBatch,
) -> Tensor:
    return F.cross_entropy(model(batch.input_ids)[:, -1], batch.targets)


def metric_summary(prefix: str, metrics: ShortcutMetrics) -> dict[str, float]:
    summary = {
        f"{prefix}/loss": metrics.loss,
        f"{prefix}/accuracy": metrics.accuracy,
        f"{prefix}/unique_value_predictions": float(
            metrics.unique_value_prediction_count
        ),
        f"{prefix}/prediction_mode_fraction": metrics.prediction_mode_fraction,
    }
    for mode, accuracy in metrics.mode_accuracy.items():
        summary[f"{prefix}/{mode}_accuracy"] = accuracy
    for mode, loss in metrics.mode_loss.items():
        summary[f"{prefix}/{mode}_loss"] = loss
    return summary


def evaluate_all(
    model: ShortcutDecoderTransformer,
    *,
    fixed_fitness_pairs: tuple[tuple[ShortcutBatch, ShortcutBatch], ...],
    heldout_fitness_pairs: tuple[tuple[ShortcutBatch, ShortcutBatch], ...],
    correct_batches: tuple[ShortcutBatch, ...],
    eval_batch_size: int,
) -> dict[str, float]:
    fixed_batches = tuple(
        batch for pair in fixed_fitness_pairs for batch in pair
    )
    heldout_batches = tuple(
        batch for pair in heldout_fitness_pairs for batch in pair
    )
    summary = metric_summary(
        "fitness_fixed",
        evaluate_shortcut_batches(
            model,
            fixed_batches,
            evaluation_batch_size=eval_batch_size,
        ),
    )
    summary.update(
        metric_summary(
            "fitness_heldout",
            evaluate_shortcut_batches(
                model,
                heldout_batches,
                evaluation_batch_size=eval_batch_size,
            ),
        )
    )
    summary.update(
        metric_summary(
            "correct_leak",
            evaluate_shortcut_batches(
                model,
                correct_batches,
                evaluation_batch_size=eval_batch_size,
            ),
        )
    )
    return summary


def initialize_wandb(config: MAMLShortcutConfig) -> Any | None:
    if not config.wandb:
        return None
    import wandb

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_group,
        name=config.run_name,
        config=asdict(config),
    )


def save_checkpoint(
    path: Path,
    *,
    config: MAMLShortcutConfig,
    model: ShortcutDecoderTransformer,
    router: AttentionRoutingRule | None,
    router_optimizer: torch.optim.Optimizer | None,
    ordinary_optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    step: int,
) -> None:
    torch.save(
        {
            "experiment": "random_position_shortcut_router_maml",
            "config": asdict(config),
            "model": model.state_dict(),
            "router": None if router is None else router.state_dict(),
            "router_optimizer": (
                None if router_optimizer is None else router_optimizer.state_dict()
            ),
            "ordinary_optimizer": ordinary_optimizer.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "step": step,
        },
        path,
    )


def run(config: MAMLShortcutConfig) -> Path:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    metrics_path = output_dir / "metrics.jsonl"

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=device)
    parameters = tuple(model.parameters())
    router = (
        make_router(config, vocabulary, device=device)
        if config.method == "router_maml"
        else None
    )
    router_optimizer = (
        torch.optim.Adam(router.parameters(), lr=config.router_learning_rate)
        if router is not None
        else None
    )
    ordinary_optimizer = torch.optim.Adam(
        parameters,
        lr=config.ordinary_learning_rate,
    )
    train_generator = torch.Generator().manual_seed(config.seed + 1_000)
    start_step = 1
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        if checkpoint.get("experiment") != (
            "random_position_shortcut_router_maml"
        ):
            raise ValueError("resume checkpoint belongs to another experiment")
        model.load_state_dict(checkpoint["model"])
        if router is not None:
            router.load_state_dict(checkpoint["router"])
            if router_optimizer is None:
                raise RuntimeError("router optimizer is missing")
            router_optimizer.load_state_dict(checkpoint["router_optimizer"])
        ordinary_optimizer.load_state_dict(checkpoint["ordinary_optimizer"])
        train_generator.set_state(checkpoint["train_generator_state"])
        start_step = int(checkpoint["step"]) + 1

    fixed_fitness_pairs = make_fitness_pairs(
        config,
        vocabulary=vocabulary,
        device=device,
        seed_offset=10_000,
    )
    heldout_fitness_pairs = make_fitness_pairs(
        config,
        vocabulary=vocabulary,
        device=device,
        seed_offset=20_000,
    )
    correct_batches = make_evaluation_batches(
        config,
        vocabulary=vocabulary,
        device=device,
        leak_mode="correct",
        seed_offset=30_000,
    )
    wandb_run = initialize_wandb(config)
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}", flush=True)

    initial_summary = {"step": 0.0}
    initial_summary.update(
        evaluate_all(
            model,
            fixed_fitness_pairs=fixed_fitness_pairs,
            heldout_fitness_pairs=heldout_fitness_pairs,
            correct_batches=correct_batches,
            eval_batch_size=config.eval_batch_size,
        )
    )
    with metrics_path.open("a") as metrics_file:
        metrics_file.write(json.dumps(initial_summary) + "\n")
    if wandb_run is not None:
        wandb_run.log(initial_summary, step=0)

    started_at = time.monotonic()
    for step in range(start_step, config.steps + 1):
        model.train()
        length = int(
            torch.randint(
                config.min_length,
                config.max_length + 1,
                (),
                generator=train_generator,
            )
        )
        biased_batch = make_shortcut_batch(
            config.batch_size,
            length,
            leak_mode="correct",
            leak_placement="random_list",
            generator=train_generator,
            vocabulary=vocabulary,
            device=device,
        )
        fitness_pair = fixed_fitness_pairs[
            (step - 1) % len(fixed_fitness_pairs)
        ]
        report_step = (
            step % config.log_interval == 0
            or step % config.eval_interval == 0
            or step == config.steps
        )

        objective = None
        meta_gradient_norm = None
        if router is not None:
            if router_optimizer is None:
                raise RuntimeError("router optimizer is missing")
            router_optimizer.zero_grad(set_to_none=True)
            ordinary_optimizer.zero_grad(set_to_none=True)
            with second_order_attention_context(device):
                objective = one_step_router_objective(
                    model,
                    router,
                    biased_batch,
                    fitness_pair,
                    inner_learning_rate=config.inner_learning_rate,
                )
                router_gradients = torch.autograd.grad(
                    objective.meta_loss,
                    tuple(router.parameters()),
                )
                for parameter, gradient in zip(
                    router.parameters(),
                    router_gradients,
                ):
                    parameter.grad = gradient
            meta_gradient_norm = torch.nn.utils.clip_grad_norm_(
                router.parameters(),
                config.gradient_clip,
            )
            router_optimizer.step()
            with torch.no_grad():
                router.gates.clamp_(min=config.router_minimum_gate)
            router_optimizer.zero_grad(set_to_none=True)

        ordinary_optimizer.zero_grad(set_to_none=True)
        logits = (
            model.forward_with_backward_rule(
                biased_batch.input_ids,
                router,
            )[:, -1]
            if router is not None
            else model(biased_batch.input_ids)[:, -1]
        )
        ordinary_loss = F.cross_entropy(logits, biased_batch.targets)
        ordinary_accuracy = logits.argmax(dim=-1).eq(
            biased_batch.targets
        ).float().mean()
        ordinary_loss.backward()
        ordinary_gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            config.gradient_clip,
        )
        ordinary_optimizer.step()

        if report_step:
            summary = {
                "step": float(step),
                "train/length": float(length),
                "train/correct_leak_loss": float(ordinary_loss.detach()),
                "train/correct_leak_accuracy": float(
                    ordinary_accuracy.detach()
                ),
                "gradient/ordinary_norm": float(ordinary_gradient_norm),
                "timing/steps_per_second": step
                / max(time.monotonic() - started_at, 1e-9),
            }
            if objective is not None and meta_gradient_norm is not None:
                summary.update(
                    {
                        "train/virtual_short_loss": float(
                            objective.short_loss.detach()
                        ),
                        "train/fixed_meta_loss_after_virtual": float(
                            objective.meta_loss.detach()
                        ),
                        "train/fixed_meta_accuracy_after_virtual": float(
                            objective.meta_accuracy.detach()
                        ),
                        "gradient/meta_norm": float(meta_gradient_norm),
                    }
                )
                summary.update(router_summary(router, biased_batch))
            if step % config.eval_interval == 0 or step == config.steps:
                summary.update(
                    evaluate_all(
                        model,
                        fixed_fitness_pairs=fixed_fitness_pairs,
                        heldout_fitness_pairs=heldout_fitness_pairs,
                        correct_batches=correct_batches,
                        eval_batch_size=config.eval_batch_size,
                    )
                )
                print(
                    f"method={config.method} step={step} "
                    f"correct={summary['correct_leak/accuracy']:.4f} "
                    f"masked={summary['fitness_heldout/masked_accuracy']:.4f} "
                    f"incorrect="
                    f"{summary['fitness_heldout/incorrect_accuracy']:.4f}",
                    flush=True,
                )
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(summary) + "\n")
            if wandb_run is not None:
                wandb_run.log(summary, step=step)

        if (
            step % config.checkpoint_interval == 0
            or step == config.steps
        ):
            for path in (
                output_dir / f"checkpoint_{step:06d}.pt",
                output_dir / "latest.pt",
            ):
                save_checkpoint(
                    path,
                    config=config,
                    model=model,
                    router=router,
                    router_optimizer=router_optimizer,
                    ordinary_optimizer=ordinary_optimizer,
                    train_generator=train_generator,
                    step=step,
                )

    if wandb_run is not None:
        wandb_run.finish()
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field in MAMLShortcutConfig.__dataclass_fields__.items():
        default = field.default
        argument = f"--{field_name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(
                argument,
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        elif default is None:
            parser.add_argument(argument)
        else:
            parser.add_argument(argument, type=type(default), default=default)
    return parser


def main() -> None:
    config = MAMLShortcutConfig(**vars(build_parser().parse_args()))
    output_dir = run(config)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
