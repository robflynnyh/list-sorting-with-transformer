"""One-step MAML for pointer-next length generalization."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from .evaluate import resolve_device
from .shortcut_credit import (
    ShortcutBatch,
    ShortcutDecoderTransformer,
    evaluate_shortcut_batches,
    make_clean_pointer_batch,
    make_forward_model_config,
)
from .tokens import PointerNextVocabulary


@dataclass(frozen=True)
class MAMLLengthConfig:
    run_name: str = "pointer-next-maml-meta40-100-heldout400-seed7"
    output_dir: str = "artifacts/maml_length_generalization"
    method: str = "maml"
    steps: int = 10_000
    batch_size: int = 64
    min_length: int = 2
    max_length: int = 20
    meta_lengths: str = "40,60,70,70,80,90,100"
    meta_examples: int = 256
    meta_batch_size: int = 64
    heldout_length: int = 400
    eval_examples: int = 128
    eval_batch_size: int = 32
    inner_learning_rate: float = 3e-4
    meta_learning_rate: float = 3e-4
    ordinary_learning_rate: float = 3e-4
    gradient_clip: float = 1.0
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    seed: int = 7
    device: str = "auto"
    wandb: bool = False
    wandb_project: str = "list-sorting-maml"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    resume: str | None = None

    def __post_init__(self) -> None:
        positive_integers = (
            self.steps,
            self.batch_size,
            self.min_length,
            self.max_length,
            self.meta_examples,
            self.meta_batch_size,
            self.heldout_length,
            self.eval_examples,
            self.eval_batch_size,
            self.d_model,
            self.layers,
            self.heads,
            self.log_interval,
            self.eval_interval,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("integer configuration values must be positive")
        if not 2 <= self.min_length <= self.max_length:
            raise ValueError("invalid training length range")
        if self.method not in {"maml", "ordinary"}:
            raise ValueError("method must be maml or ordinary")
        meta_lengths = parse_meta_lengths(self.meta_lengths)
        if meta_lengths[0] <= self.max_length:
            raise ValueError("meta lengths must exceed the training range")
        if self.heldout_length <= meta_lengths[-1]:
            raise ValueError("held-out length must exceed all meta lengths")
        if self.meta_examples % self.meta_batch_size:
            raise ValueError(
                "meta_examples must be divisible by meta_batch_size"
            )
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if min(
            self.inner_learning_rate,
            self.meta_learning_rate,
            self.ordinary_learning_rate,
            self.gradient_clip,
        ) <= 0:
            raise ValueError("learning rates and gradient clip must be positive")


def parse_meta_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("meta lengths must be integers") from error
    if (
        not lengths
        or any(length < 2 for length in lengths)
        or tuple(sorted(lengths)) != lengths
    ):
        raise ValueError(
            "meta lengths must be nondecreasing integers of at least two"
        )
    return lengths


@dataclass(frozen=True)
class MAMLObjective:
    short_loss: Tensor
    meta_loss: Tensor
    meta_accuracy: Tensor


def make_fixed_batches(
    example_count: int,
    *,
    batch_size: int,
    length: int,
    vocabulary: PointerNextVocabulary,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[ShortcutBatch, ...]:
    batches = []
    remaining = example_count
    while remaining:
        current_size = min(batch_size, remaining)
        batches.append(
            make_clean_pointer_batch(
                current_size,
                length,
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
        )
        remaining -= current_size
    return tuple(batches)


def make_meta_batches(
    config: MAMLLengthConfig,
    *,
    vocabulary: PointerNextVocabulary,
    device: torch.device,
) -> tuple[ShortcutBatch, ...]:
    """Create fixed per-length sets and interleave their minibatches."""

    generator = torch.Generator().manual_seed(config.seed + 10_000)
    groups = tuple(
        make_fixed_batches(
            config.meta_examples,
            batch_size=config.meta_batch_size,
            length=length,
            vocabulary=vocabulary,
            generator=generator,
            device=device,
        )
        for length in parse_meta_lengths(config.meta_lengths)
    )
    batches_per_length = len(groups[0])
    if any(len(group) != batches_per_length for group in groups):
        raise RuntimeError("meta lengths produced unequal batch counts")
    return tuple(
        groups[length_index][batch_index]
        for batch_index in range(batches_per_length)
        for length_index in range(len(groups))
    )


def make_model(
    config: MAMLLengthConfig,
    vocabulary: PointerNextVocabulary,
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


def second_order_attention_context(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    )


def one_step_maml_objective(
    model: ShortcutDecoderTransformer,
    short_batch: ShortcutBatch,
    meta_batch: ShortcutBatch,
    *,
    inner_learning_rate: float,
    create_graph: bool = True,
) -> MAMLObjective:
    """Differentiate meta loss through one virtual short-task SGD step."""

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    short_logits = model(short_batch.input_ids)[:, -1]
    short_loss = F.cross_entropy(short_logits, short_batch.targets)
    short_gradients = torch.autograd.grad(
        short_loss,
        tuple(parameters.values()),
        create_graph=create_graph,
    )
    adapted_parameters = {
        name: parameter - inner_learning_rate * gradient
        for (name, parameter), gradient in zip(
            parameters.items(),
            short_gradients,
        )
    }
    meta_logits = functional_call(
        model,
        (adapted_parameters, buffers),
        (meta_batch.input_ids,),
    )[:, -1]
    meta_loss = F.cross_entropy(meta_logits, meta_batch.targets)
    return MAMLObjective(
        short_loss=short_loss,
        meta_loss=meta_loss,
        meta_accuracy=meta_logits.argmax(dim=-1).eq(
            meta_batch.targets
        ).float().mean(),
    )


def batch_loss(
    model: ShortcutDecoderTransformer,
    batch: ShortcutBatch,
) -> Tensor:
    return F.cross_entropy(
        model(batch.input_ids)[:, -1],
        batch.targets,
    )


def evaluate_lengths(
    model: ShortcutDecoderTransformer,
    batches_by_length: dict[int, tuple[ShortcutBatch, ...]],
    *,
    evaluation_batch_size: int,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for length, batches in batches_by_length.items():
        metrics = evaluate_shortcut_batches(
            model,
            batches,
            evaluation_batch_size=evaluation_batch_size,
        )
        prefix = f"eval/length_{length}"
        summary[f"{prefix}/loss"] = metrics.loss
        summary[f"{prefix}/accuracy"] = metrics.accuracy
        summary[f"{prefix}/unique_value_predictions"] = float(
            metrics.unique_value_prediction_count
        )
        summary[f"{prefix}/prediction_mode_fraction"] = (
            metrics.prediction_mode_fraction
        )
    return summary


def initialize_wandb(config: MAMLLengthConfig) -> Any | None:
    if not config.wandb:
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking requires the project tracking dependencies"
        ) from error
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
    config: MAMLLengthConfig,
    model: ShortcutDecoderTransformer,
    meta_optimizer: torch.optim.Optimizer,
    ordinary_optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    step: int,
) -> None:
    torch.save(
        {
            "experiment": "pointer_next_one_step_maml",
            "config": asdict(config),
            "model": model.state_dict(),
            "meta_optimizer": meta_optimizer.state_dict(),
            "ordinary_optimizer": ordinary_optimizer.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "step": step,
        },
        path,
    )


def run(config: MAMLLengthConfig) -> Path:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    metrics_path = output_dir / "metrics.jsonl"

    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=device)
    parameters = tuple(model.parameters())
    meta_optimizer = torch.optim.Adam(
        parameters,
        lr=config.meta_learning_rate,
    )
    ordinary_optimizer = torch.optim.Adam(
        parameters,
        lr=config.ordinary_learning_rate,
    )
    train_generator = torch.Generator().manual_seed(config.seed + 1_000)
    start_step = 1
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        if checkpoint.get("experiment") != "pointer_next_one_step_maml":
            raise ValueError("resume checkpoint belongs to another experiment")
        model.load_state_dict(checkpoint["model"])
        meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        ordinary_optimizer.load_state_dict(checkpoint["ordinary_optimizer"])
        train_generator.set_state(checkpoint["train_generator_state"])
        start_step = int(checkpoint["step"]) + 1

    meta_batches = (
        make_meta_batches(
            config,
            vocabulary=vocabulary,
            device=device,
        )
        if config.method == "maml"
        else ()
    )
    meta_lengths = parse_meta_lengths(config.meta_lengths)
    evaluation_lengths = tuple(
        sorted(
            {
                config.min_length,
                config.max_length,
                *meta_lengths,
                50,
                config.heldout_length,
            }
        )
    )
    evaluation_batches = {
        length: make_fixed_batches(
            config.eval_examples,
            batch_size=config.eval_batch_size,
            length=length,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(
                config.seed + 20_000 + length
            ),
            device=device,
        )
        for length in evaluation_lengths
    }
    wandb_run = initialize_wandb(config)
    if wandb_run is not None:
        print(f"W&B: {wandb_run.url}", flush=True)

    initial_summary = {"step": 0.0}
    initial_summary.update(
        evaluate_lengths(
            model,
            evaluation_batches,
            evaluation_batch_size=config.eval_batch_size,
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
        short_batch = make_clean_pointer_batch(
            config.batch_size,
            length,
            generator=train_generator,
            vocabulary=vocabulary,
            device=device,
        )
        meta_batch = (
            meta_batches[(step - 1) % len(meta_batches)]
            if meta_batches
            else None
        )
        report_step = (
            step % config.log_interval == 0
            or step % config.eval_interval == 0
            or step == config.steps
        )
        meta_loss_before = None
        if report_step and meta_batch is not None:
            with torch.no_grad():
                meta_loss_before = float(batch_loss(model, meta_batch))

        objective = None
        meta_gradient_norm = None
        if meta_batch is not None:
            meta_optimizer.zero_grad(set_to_none=True)
            ordinary_optimizer.zero_grad(set_to_none=True)
            with second_order_attention_context(device):
                objective = one_step_maml_objective(
                    model,
                    short_batch,
                    meta_batch,
                    inner_learning_rate=config.inner_learning_rate,
                )
                objective.meta_loss.backward()
            meta_gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                config.gradient_clip,
            )
            meta_optimizer.step()

        ordinary_optimizer.zero_grad(set_to_none=True)
        ordinary_logits = model(short_batch.input_ids)[:, -1]
        ordinary_loss = F.cross_entropy(
            ordinary_logits,
            short_batch.targets,
        )
        ordinary_accuracy = (
            ordinary_logits.argmax(dim=-1)
            .eq(short_batch.targets)
            .float()
            .mean()
        )
        ordinary_loss.backward()
        ordinary_gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            config.gradient_clip,
        )
        ordinary_optimizer.step()

        if report_step:
            elapsed = time.monotonic() - started_at
            summary = {
                "step": float(step),
                "train/length": float(length),
                "train/ordinary_short_loss": float(ordinary_loss.detach()),
                "train/ordinary_short_accuracy": float(
                    ordinary_accuracy.detach()
                ),
                "gradient/ordinary_norm": float(ordinary_gradient_norm),
                "timing/steps_per_second": step / max(elapsed, 1e-9),
            }
            if objective is not None:
                if meta_batch is None or meta_gradient_norm is None:
                    raise RuntimeError("MAML reporting state is incomplete")
                summary.update(
                    {
                        "train/meta_batch_length": float(meta_batch.length),
                        "train/virtual_short_loss": float(
                            objective.short_loss.detach()
                        ),
                        "train/meta_loss_after_virtual": float(
                            objective.meta_loss.detach()
                        ),
                        "train/meta_accuracy_after_virtual": float(
                            objective.meta_accuracy.detach()
                        ),
                        "gradient/meta_norm": float(meta_gradient_norm),
                    }
                )
            if meta_loss_before is not None and objective is not None:
                summary["train/meta_loss_before_virtual"] = (
                    meta_loss_before
                )
                summary["train/virtual_step_meta_loss_change"] = (
                    float(objective.meta_loss.detach()) - meta_loss_before
                )
            if step % config.eval_interval == 0 or step == config.steps:
                summary.update(
                    evaluate_lengths(
                        model,
                        evaluation_batches,
                        evaluation_batch_size=config.eval_batch_size,
                    )
                )
                print(
                    f"method={config.method} "
                    f"step={step} "
                    f"short_loss={float(ordinary_loss.detach()):.4f} "
                    f"length400_acc="
                    f"{summary[f'eval/length_{config.heldout_length}/accuracy']:.4f}",
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
            save_checkpoint(
                output_dir / f"checkpoint_{step:06d}.pt",
                config=config,
                model=model,
                meta_optimizer=meta_optimizer,
                ordinary_optimizer=ordinary_optimizer,
                train_generator=train_generator,
                step=step,
            )
            save_checkpoint(
                output_dir / "latest.pt",
                config=config,
                model=model,
                meta_optimizer=meta_optimizer,
                ordinary_optimizer=ordinary_optimizer,
                train_generator=train_generator,
                step=step,
            )

    if wandb_run is not None:
        wandb_run.finish()
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field in MAMLLengthConfig.__dataclass_fields__.items():
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
    config = MAMLLengthConfig(**vars(build_parser().parse_args()))
    output_dir = run(config)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
