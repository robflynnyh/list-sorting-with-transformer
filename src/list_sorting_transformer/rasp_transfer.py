"""Measure transfer from a compiled pointer circuit to related puzzle tasks."""

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

from .compiled_pointer_compare import (
    DEFAULT_POSITION_MODULI,
    CompiledPointerCompareConfig,
    CompiledPointerCompareTransformer,
)
from .data import PointerNextBatch, make_pointer_pair_batch
from .evaluate import resolve_device
from .model import ModelConfig, SplitInputDecoderTransformer
from .positions import ModularPositionEmbedding, sample_position_offsets
from .tokens import BOS, SEP, VALUE_OFFSET, PointerCompareVocabulary


TASKS = (
    "following_value",
    "three_way_relation",
    "associative_recall",
    "dyck_2_completion",
)
INITIALIZATIONS = ("random", "compiled_prefix", "compiled_full")
EVALUATION_LENGTHS = (20, 40, 100, 200, 400)
EVALUATION_STEPS = (
    0,
    10,
    50,
    100,
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
)
ROUND_OPEN = VALUE_OFFSET
ROUND_CLOSE = VALUE_OFFSET + 1
SQUARE_OPEN = VALUE_OFFSET + 2
SQUARE_CLOSE = VALUE_OFFSET + 3


@dataclass(frozen=True)
class TransferBatch:
    prompt_ids: Tensor
    targets: Tensor
    pointer_offsets: Tensor | None = None


@dataclass(frozen=True)
class RaspTransferConfig:
    task: str
    initialization: str
    seed: int
    steps: int = 2_000
    batch_size: int = 256
    train_min_length: int = 2
    train_max_length: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.001
    gradient_clip: float = 1.0
    eval_examples: int = 256
    position_offset_min: int = -1_000_000
    position_offset_max: int = 1_000_000
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(
                f"initialization must be one of {INITIALIZATIONS}"
            )
        if self.steps < 1 or self.batch_size < 1 or self.eval_examples < 1:
            raise ValueError("steps, batch size, and eval examples must be positive")
        if not 2 <= self.train_min_length <= self.train_max_length:
            raise ValueError("invalid training length range")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")


class RaspTransferModel(nn.Module):
    """The project Transformer with a fresh downstream classification head."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        position_moduli: tuple[int, ...],
        output_classes: int,
    ) -> None:
        super().__init__()
        self.encoder = SplitInputDecoderTransformer(
            model_config,
            content_dim=model_config.d_model // 2,
        )
        self.position_embedding = ModularPositionEmbedding(
            self.encoder.position_dim,
            position_moduli,
        )
        self.output_head = nn.Linear(model_config.d_model, output_classes)

    def input_embeddings(self, prompt_ids: Tensor, offsets: Tensor) -> Tensor:
        token_offsets = torch.arange(
            prompt_ids.shape[1],
            device=prompt_ids.device,
        )
        positions = self.position_embedding(
            offsets[:, None] + token_offsets[None, :]
        )
        return torch.cat((self.encoder.embed(prompt_ids), positions), dim=-1)

    def forward(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
        diagnostics: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        hidden = self.input_embeddings(prompt_ids, offsets)
        pointer_route_logits = (
            self.encoder.blocks[0].attention.query_key_logits(
                self.encoder.blocks[0].attention_norm(hidden),
                query_index=-1,
            )[:, 0]
        )
        hidden = self.encoder.blocks[0](hidden)
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
        for block in self.encoder.blocks[1:]:
            hidden = block(hidden)
        logits = self.output_head(self.encoder.final_norm(hidden[:, -1]))
        if not diagnostics:
            return logits
        return {
            "logits": logits,
            "pointer_route_logits": pointer_route_logits,
            "marked_route_logits": marked_route_logits,
            "following_route_logits": following_route_logits,
        }


def output_classes(task: str) -> int:
    if task in {"following_value", "associative_recall"}:
        return 10
    if task == "three_way_relation":
        return 3
    if task == "dyck_2_completion":
        return 2
    raise ValueError(f"unknown task: {task}")


def task_targets(task: str, batch: PointerNextBatch) -> Tensor:
    rows = torch.arange(batch.values.shape[0], device=batch.values.device)
    marked = batch.values[rows, batch.pointers]
    following = batch.values[rows, batch.pointers + 1]
    if task == "following_value":
        return following
    if task == "three_way_relation":
        return torch.where(
            marked.lt(following),
            torch.zeros_like(marked),
            torch.where(
                marked.eq(following),
                torch.ones_like(marked),
                torch.full_like(marked, 2),
            ),
        )
    raise ValueError(f"unknown task: {task}")


def _random_balanced_dyck_2(
    pair_count: int,
    *,
    generator: torch.Generator,
) -> list[int]:
    sequence: list[int] = []
    stack: list[int] = []
    opens_remaining = pair_count
    while opens_remaining or stack:
        should_open = opens_remaining > 0 and (
            not stack
            or bool(
                torch.randint(
                    0,
                    2,
                    (),
                    generator=generator,
                )
            )
        )
        if should_open:
            bracket_type = int(
                torch.randint(0, 2, (), generator=generator)
            )
            stack.append(bracket_type)
            sequence.append(
                ROUND_OPEN if bracket_type == 0 else SQUARE_OPEN
            )
            opens_remaining -= 1
        else:
            bracket_type = stack.pop()
            sequence.append(
                ROUND_CLOSE if bracket_type == 0 else SQUARE_CLOSE
            )
    return sequence


def make_dyck_2_completion_batch(
    batch_size: int,
    pair_count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> TransferBatch:
    if pair_count < 2:
        raise ValueError("Dyck-2 completion requires at least two pairs")
    prompts: list[list[int]] = []
    targets: list[int] = []
    for _ in range(batch_size):
        segment_assignments = torch.randint(
            0,
            5,
            (pair_count,),
            generator=generator,
        )
        segment_sizes = torch.bincount(
            segment_assignments,
            minlength=5,
        ).tolist()
        unmatched_types = torch.randint(
            0,
            2,
            (4,),
            generator=generator,
        ).tolist()
        sequence = _random_balanced_dyck_2(
            segment_sizes[0],
            generator=generator,
        )
        for index, bracket_type in enumerate(unmatched_types):
            sequence.append(
                ROUND_OPEN if bracket_type == 0 else SQUARE_OPEN
            )
            sequence.extend(
                _random_balanced_dyck_2(
                    segment_sizes[index + 1],
                    generator=generator,
                )
            )
        prompts.append([BOS, *sequence, SEP])
        targets.append(unmatched_types[-1])
    return TransferBatch(
        prompt_ids=torch.tensor(
            prompts,
            dtype=torch.long,
            device=device,
        ),
        targets=torch.tensor(
            targets,
            dtype=torch.long,
            device=device,
        ),
    )


def make_associative_recall_batch(
    batch_size: int,
    pair_count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> TransferBatch:
    mappings = torch.stack(
        [
            torch.randperm(10, generator=generator)
            for _ in range(batch_size)
        ]
    )
    keys = torch.randint(
        0,
        10,
        (batch_size, pair_count),
        generator=generator,
    )
    values = mappings.gather(1, keys)
    query_indices = torch.randint(
        0,
        pair_count,
        (batch_size, 1),
        generator=generator,
    )
    query_keys = keys.gather(1, query_indices).squeeze(1)
    targets = mappings.gather(1, query_keys[:, None]).squeeze(1)

    prompts = torch.empty(
        batch_size,
        2 * pair_count + 3,
        dtype=torch.long,
    )
    prompts[:, 0] = BOS
    prompts[:, 1 : 2 * pair_count + 1 : 2] = VALUE_OFFSET + keys
    prompts[:, 2 : 2 * pair_count + 1 : 2] = VALUE_OFFSET + values
    prompts[:, -2] = VALUE_OFFSET + query_keys
    prompts[:, -1] = SEP
    return TransferBatch(
        prompt_ids=prompts.to(device),
        targets=targets.to(device),
    )


def build_transfer_model(config: RaspTransferConfig) -> RaspTransferModel:
    torch.manual_seed(config.seed)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    model_config = ModelConfig(
        vocab_size=vocabulary.size,
        symbol_count=10,
        representation="numbers",
        d_model=128,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=4.0,
        dropout=0.0,
        position_pattern="none",
    )
    model = RaspTransferModel(
        model_config,
        position_moduli=config.position_moduli,
        output_classes=output_classes(config.task),
    )
    if config.initialization == "random":
        return model

    compiled = CompiledPointerCompareTransformer(
        CompiledPointerCompareConfig(
            pointer_selection_logit=20.0,
            address_score_scale=5.0,
            pointer_scratch_scale=1.0,
        )
    )
    model.encoder.token_embedding.load_state_dict(
        compiled.encoder.token_embedding.state_dict()
    )
    model.position_embedding.load_state_dict(
        compiled.position_embedding.state_dict()
    )
    if config.initialization == "compiled_prefix":
        for index in (0, 1):
            model.encoder.blocks[index].load_state_dict(
                compiled.encoder.blocks[index].state_dict()
            )
    elif config.initialization == "compiled_full":
        model.encoder.load_state_dict(compiled.encoder.state_dict())
    return model


def _make_batch(
    config: RaspTransferConfig,
    *,
    length: int,
    batch_size: int,
    generator: torch.Generator,
    vocabulary: PointerCompareVocabulary,
    device: torch.device,
) -> tuple[TransferBatch, Tensor]:
    if config.task == "dyck_2_completion":
        transfer_batch = make_dyck_2_completion_batch(
            batch_size,
            length,
            generator=generator,
            device=device,
        )
        offsets = sample_position_offsets(
            batch_size,
            minimum=config.position_offset_min,
            maximum=config.position_offset_max,
            generator=generator,
            device=device,
        )
        return transfer_batch, offsets
    if config.task == "associative_recall":
        transfer_batch = make_associative_recall_batch(
            batch_size,
            length,
            generator=generator,
            device=device,
        )
        offsets = sample_position_offsets(
            batch_size,
            minimum=config.position_offset_min,
            maximum=config.position_offset_max,
            generator=generator,
            device=device,
        )
        return transfer_batch, offsets

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
    return (
        TransferBatch(
            prompt_ids=batch.prompt_ids,
            targets=task_targets(config.task, batch),
            pointer_offsets=1 + 2 * batch.pointers,
        ),
        offsets,
    )


@torch.inference_mode()
def evaluate(
    model: RaspTransferModel,
    config: RaspTransferConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    vocabulary = PointerCompareVocabulary("numbers", 10)
    generator = torch.Generator().manual_seed(seed)
    results: dict[str, dict[str, float]] = {}
    for length in EVALUATION_LENGTHS:
        totals = {
            "task_accuracy": 0,
            "pointer_route_accuracy": 0,
            "marked_route_accuracy": 0,
            "following_route_accuracy": 0,
        }
        completed = 0
        while completed < config.eval_examples:
            current_size = min(32, config.eval_examples - completed)
            batch, offsets = _make_batch(
                config,
                length=length,
                batch_size=current_size,
                generator=generator,
                vocabulary=vocabulary,
                device=device,
            )
            outputs = model(
                batch.prompt_ids,
                offsets=offsets,
                diagnostics=True,
            )
            totals["task_accuracy"] += int(
                outputs["logits"].argmax(-1).eq(batch.targets).sum()
            )
            if batch.pointer_offsets is not None:
                totals["pointer_route_accuracy"] += int(
                    outputs["pointer_route_logits"]
                    .argmax(-1)
                    .eq(batch.pointer_offsets)
                    .sum()
                )
                totals["marked_route_accuracy"] += int(
                    outputs["marked_route_logits"]
                    .argmax(-1)
                    .eq(batch.pointer_offsets + 1)
                    .sum()
                )
                totals["following_route_accuracy"] += int(
                    outputs["following_route_logits"]
                    .argmax(-1)
                    .eq(batch.pointer_offsets + 3)
                    .sum()
                )
            completed += current_size
        metric_names = ["task_accuracy"]
        if config.task not in {"associative_recall", "dyck_2_completion"}:
            metric_names.extend(
                (
                    "pointer_route_accuracy",
                    "marked_route_accuracy",
                    "following_route_accuracy",
                )
            )
        results[str(length)] = {
            name: totals[name] / config.eval_examples
            for name in metric_names
        }
    return results


def train(
    config: RaspTransferConfig,
    *,
    output_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(config.seed)
    model = build_transfer_model(config).to(device)
    vocabulary = PointerCompareVocabulary("numbers", 10)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    evaluation_steps = tuple(
        step for step in EVALUATION_STEPS if step <= config.steps
    )
    history: list[dict[str, Any]] = []
    started_at = time.monotonic()

    def record(step: int) -> None:
        metrics = evaluate(
            model,
            config,
            device=device,
            seed=config.seed + 100_000 + step,
        )
        entry = {"step": step, "per_length": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)

    record(0)
    model.train()
    for step in range(1, config.steps + 1):
        length = random.Random(config.seed + step).randint(
            config.train_min_length,
            config.train_max_length,
        )
        batch, offsets = _make_batch(
            config,
            length=length,
            batch_size=config.batch_size,
            generator=generator,
            vocabulary=vocabulary,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.prompt_ids, offsets=offsets)
        loss = F.cross_entropy(logits, batch.targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )
        optimizer.step()
        if step in evaluation_steps[1:]:
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                    }
                ),
                flush=True,
            )
            record(step)
            model.train()

    result = {
        "experiment": "rasp_transfer",
        "config": asdict(config),
        "model_config": model.encoder.config.as_dict(),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "wall_time_seconds": time.monotonic() - started_at,
        "history": history,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=len(values) > 1))


def load_results(input_root: Path) -> list[dict[str, Any]]:
    paths = sorted(input_root.glob("*/*/seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"no metrics found below {input_root}")
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in TASKS:
        summary[task] = {}
        for initialization in INITIALIZATIONS:
            matching = [
                result
                for result in results
                if result["config"]["task"] == task
                and result["config"]["initialization"] == initialization
            ]
            if not matching:
                continue
            by_step: dict[str, Any] = {}
            for index, entry in enumerate(matching[0]["history"]):
                step = str(entry["step"])
                by_step[step] = {}
                for length in EVALUATION_LENGTHS:
                    length_key = str(length)
                    by_step[step][length_key] = {}
                    metric_names = tuple(
                        matching[0]["history"][index]["per_length"][
                            length_key
                        ]
                    )
                    for metric in metric_names:
                        values = [
                            result["history"][index]["per_length"][length_key][
                                metric
                            ]
                            for result in matching
                        ]
                        mean, std = _mean_and_std(values)
                        by_step[step][length_key][metric] = {
                            "mean": mean,
                            "std": std,
                        }
            summary[task][initialization] = {
                "seeds": sorted(result["config"]["seed"] for result in matching),
                "by_step": by_step,
            }
    return summary


def render_plots(summary: dict[str, Any], output_directory: Path) -> None:
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    labels = {
        "random": "Random",
        "compiled_prefix": "Compiled prefix",
        "compiled_full": "Compiled full",
    }
    task_labels = {
        "following_value": "Following value",
        "three_way_relation": "Three-way relation",
        "associative_recall": "Associative recall",
        "dyck_2_completion": "Dyck-2 completion",
    }
    colors = {
        "random": "#4C78A8",
        "compiled_prefix": "#F58518",
        "compiled_full": "#54A24B",
    }

    figure, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(10, 3.3 * len(TASKS)),
        sharex="row",
        sharey=True,
    )
    for row, task in enumerate(TASKS):
        for column, length in enumerate((20, 400)):
            axis = axes[row, column]
            for initialization in INITIALIZATIONS:
                entry = summary[task][initialization]["by_step"]
                steps = [int(step) for step in entry]
                means = [
                    entry[str(step)][str(length)]["task_accuracy"]["mean"]
                    for step in steps
                ]
                stds = [
                    entry[str(step)][str(length)]["task_accuracy"]["std"]
                    for step in steps
                ]
                axis.plot(
                    steps,
                    means,
                    marker="o",
                    markersize=3,
                    label=labels[initialization],
                    color=colors[initialization],
                )
                axis.fill_between(
                    steps,
                    [
                        max(0.0, mean - std)
                        for mean, std in zip(means, stds)
                    ],
                    [
                        min(1.0, mean + std)
                        for mean, std in zip(means, stds)
                    ],
                    color=colors[initialization],
                    alpha=0.15,
                )
            axis.set_title(f"{task_labels[task]}, length {length}")
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Fine-tuning updates")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy")
    axes[0, 0].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(
        output_directory / "rasp_transfer_learning_curves.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=(5 * len(TASKS), 3.8),
        sharey=True,
    )
    for axis, task in zip(axes, TASKS):
        for initialization in INITIALIZATIONS:
            entry = summary[task][initialization]["by_step"]
            final_step = str(max(int(step) for step in entry))
            means = [
                entry[final_step][str(length)]["task_accuracy"]["mean"]
                for length in EVALUATION_LENGTHS
            ]
            axis.plot(
                EVALUATION_LENGTHS,
                means,
                marker="o",
                label=labels[initialization],
                color=colors[initialization],
            )
        axis.axvspan(20, 400, color="#999999", alpha=0.05)
        axis.axvline(20, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(task_labels[task])
        axis.set_xlabel("Problem length")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Final-checkpoint accuracy")
    axes[0].legend(loc="lower left")
    axes[0].set_ylim(0, 1.02)
    figure.tight_layout()
    figure.savefig(
        output_directory / "rasp_transfer_length_generalization.png",
        dpi=180,
    )
    plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--task", choices=TASKS, required=True)
    train_parser.add_argument(
        "--initialization",
        choices=INITIALIZATIONS,
        required=True,
    )
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--steps", type=int, default=2_000)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--eval-examples", type=int, default=256)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--output-directory", type=Path, required=True)

    report_parser = subparsers.add_parser("summarize")
    report_parser.add_argument("--input-root", type=Path, required=True)
    report_parser.add_argument("--output-json", type=Path, required=True)
    report_parser.add_argument("--plot-directory", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.command == "train":
        config = RaspTransferConfig(
            task=args.task,
            initialization=args.initialization,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_examples=args.eval_examples,
        )
        result = train(
            config,
            output_directory=args.output_directory,
            device=resolve_device(args.device),
        )
        print(
            json.dumps(
                {
                    "output_directory": str(args.output_directory),
                    "wall_time_seconds": result["wall_time_seconds"],
                }
            )
        )
        return

    results = load_results(args.input_root)
    summary = summarize_results(results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_plots(summary, args.plot_directory)


if __name__ == "__main__":
    main()
