"""Evaluate hard-attention checkpoints across sequence lengths and plot them."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
import torch.nn.functional as F
from torch import Tensor

from .data import make_pointer_next_batch
from .hard_attention_eggroll import (
    HardAttentionEggrollConfig,
    HardAttentionPointerTransformer,
    make_model,
    pointer_targets,
    restore_curriculum_state,
)
from .positions import sample_position_offsets


DEFAULT_LENGTHS = (
    2,
    5,
    10,
    20,
    40,
    80,
    100,
    200,
    400,
    600,
    800,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
)


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    generation: int
    current_max_length: int
    attention_top_k: int | None
    active_head_indices: tuple[tuple[int, ...], ...]

    @property
    def active_heads(self) -> int:
        return len(self.active_head_indices[0])


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers >= 2")
    return values


def checkpoint_spec(path: Path) -> CheckpointSpec:
    checkpoint = torch.load(path, map_location="cpu")
    config = HardAttentionEggrollConfig(**checkpoint["config"])
    state = restore_curriculum_state(checkpoint["curriculum_state"], config)
    return CheckpointSpec(
        path=path,
        generation=int(checkpoint["generation"]),
        current_max_length=state.current_max_length,
        attention_top_k=state.attention_top_k,
        active_head_indices=state.active_head_indices,
    )


def discover_checkpoints(
    checkpoint_dirs: tuple[Path, ...],
    extra_checkpoints: tuple[Path, ...],
) -> tuple[CheckpointSpec, ...]:
    paths = [
        *(path for directory in checkpoint_dirs for path in directory.glob("checkpoint_*.pt")),
        *extra_checkpoints,
    ]
    by_generation: dict[int, CheckpointSpec] = {}
    for path in paths:
        spec = checkpoint_spec(path)
        by_generation[spec.generation] = spec
    if not by_generation:
        raise ValueError("no checkpoints found")
    return tuple(by_generation[generation] for generation in sorted(by_generation))


@torch.inference_mode()
def exact_top1_logits(
    model: HardAttentionPointerTransformer,
    prompt_ids: Tensor,
    offsets: Tensor,
    *,
    query_chunk_size: int,
) -> Tensor:
    """Run exact top-1 attention while bounding score-matrix memory."""

    if model.current_top_k != 1:
        raise ValueError("checkpoint sweep requires exact top-1 attention")
    content = model.encoder.embed(prompt_ids)
    positions = model.position_embeddings(prompt_ids, offsets)
    hidden = torch.cat((content, positions), dim=-1)
    batch_size, sequence_length, model_dim = hidden.shape

    for layer_index, block in enumerate(model.encoder.blocks):
        normalized = block.attention_norm(hidden)
        query, key, value = block.attention.qkv(normalized).chunk(3, dim=-1)
        query = block.attention._split_heads(query)
        key = block.attention._split_heads(key)
        value = block.attention._split_heads(value)
        active_heads = model.current_active_head_indices[layer_index]
        attended = torch.zeros(
            batch_size,
            sequence_length,
            block.attention.n_heads,
            block.attention.head_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        key_positions = torch.arange(sequence_length, device=hidden.device)

        for start in range(0, sequence_length, query_chunk_size):
            end = min(start + query_chunk_size, sequence_length)
            active_query = query[:, active_heads, start:end]
            active_key = key[:, active_heads]
            scores = (
                active_query @ active_key.transpose(-2, -1)
                / block.attention.head_dim**0.5
            )
            query_positions = torch.arange(start, end, device=hidden.device)
            scores.masked_fill_(
                key_positions[None, None, None, :]
                > query_positions[None, None, :, None],
                float("-inf"),
            )
            selected = scores.argmax(dim=-1)
            active_value = value[:, active_heads]
            chosen = active_value.gather(
                2,
                selected.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    block.attention.head_dim,
                ),
            )
            for active_index, head_index in enumerate(active_heads):
                attended[:, start:end, head_index] = chosen[:, active_index]

        hidden = hidden + block.attention.output(
            attended.reshape(batch_size, sequence_length, model_dim)
        )
        hidden = hidden + block.ffn(block.ffn_norm(hidden))

    hidden = model.encoder.final_norm(hidden)
    return model.output(hidden[:, -1])


def examples_for_length(
    length: int,
    *,
    short_examples: int,
    long_examples: int,
    long_min_length: int,
) -> int:
    return long_examples if length >= long_min_length else short_examples


def evaluation_batch(
    model: HardAttentionPointerTransformer,
    *,
    length: int,
    examples: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed + length * 10_003)
    batch = make_pointer_next_batch(
        examples,
        length,
        generator=generator,
        vocabulary=model.vocabulary,
    )
    offsets = sample_position_offsets(
        examples,
        minimum=model.config.position_offset_min,
        maximum=model.config.position_offset_max,
        generator=generator,
        device=torch.device("cpu"),
    )
    return batch.prompt_ids, pointer_targets(batch), offsets


def resolved_batch_size(
    *,
    examples: int,
    sequence_length: int,
    active_heads: int,
    query_chunk_size: int,
    score_element_budget: int,
    maximum_batch_size: int,
) -> int:
    score_elements_per_example = (
        active_heads * min(query_chunk_size, sequence_length) * sequence_length
    )
    return max(
        1,
        min(
            examples,
            maximum_batch_size,
            score_element_budget // max(score_elements_per_example, 1),
        ),
    )


def load_model(
    spec: CheckpointSpec,
    *,
    device: torch.device,
) -> tuple[HardAttentionPointerTransformer, HardAttentionEggrollConfig]:
    checkpoint = torch.load(spec.path, map_location=device)
    config = HardAttentionEggrollConfig(**checkpoint["config"])
    model = make_model(config, device=device)
    model.load_state_dict(checkpoint["model"])
    model.set_attention_top_k(spec.attention_top_k)
    model.set_active_head_indices(spec.active_head_indices)
    model.eval()
    return model, config


def existing_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="") as source:
        return {
            (int(row["generation"]), int(row["length"]))
            for row in csv.DictReader(source)
        }


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=tuple(row),
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_sweep(
    specs: tuple[CheckpointSpec, ...],
    *,
    lengths: tuple[int, ...],
    output_csv: Path,
    device: torch.device,
    seed: int,
    short_examples: int,
    long_examples: int,
    long_min_length: int,
    query_chunk_size: int,
    score_element_budget: int,
    maximum_batch_size: int,
) -> None:
    completed = existing_keys(output_csv)
    started_at = time.monotonic()
    for checkpoint_index, spec in enumerate(specs, start=1):
        model, _ = load_model(spec, device=device)
        for length in lengths:
            if (spec.generation, length) in completed:
                continue
            examples = examples_for_length(
                length,
                short_examples=short_examples,
                long_examples=long_examples,
                long_min_length=long_min_length,
            )
            prompt_ids, targets, offsets = evaluation_batch(
                model,
                length=length,
                examples=examples,
                seed=seed,
            )
            batch_size = resolved_batch_size(
                examples=examples,
                sequence_length=prompt_ids.shape[1],
                active_heads=spec.active_heads,
                query_chunk_size=query_chunk_size,
                score_element_budget=score_element_budget,
                maximum_batch_size=maximum_batch_size,
            )
            loss_sum = 0.0
            correct = 0
            for start in range(0, examples, batch_size):
                end = min(start + batch_size, examples)
                logits = exact_top1_logits(
                    model,
                    prompt_ids[start:end].to(device),
                    offsets[start:end].to(device),
                    query_chunk_size=query_chunk_size,
                )
                batch_targets = targets[start:end].to(device)
                loss_sum += float(
                    F.cross_entropy(logits, batch_targets, reduction="sum")
                )
                correct += int(logits.argmax(dim=-1).eq(batch_targets).sum())
            row = {
                "generation": spec.generation,
                "length": length,
                "examples": examples,
                "loss": loss_sum / examples,
                "accuracy": correct / examples,
                "current_max_length": spec.current_max_length,
                "attention_top_k": spec.attention_top_k or 0,
                "active_heads": spec.active_heads,
                "active_head_indices": json.dumps(spec.active_head_indices),
                "checkpoint": str(spec.path),
            }
            append_result(output_csv, row)
            elapsed = time.monotonic() - started_at
            print(
                f"[{checkpoint_index}/{len(specs)}] "
                f"generation={spec.generation} length={length} "
                f"accuracy={row['accuracy']:.4f} examples={examples} "
                f"batch={batch_size} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def checkpoint_axis_labels(rows: list[dict[str, str]]) -> tuple[list[int], list[str]]:
    metadata: dict[int, tuple[int, int]] = {}
    for row in rows:
        metadata[int(row["generation"])] = (
            int(row["attention_top_k"]),
            int(row["active_heads"]),
        )
    generations = sorted(metadata)
    labels = []
    previous: tuple[int, int] | None = None
    for generation in generations:
        top_k, heads = metadata[generation]
        structure = (top_k, heads)
        if previous is None:
            suffix = f"top-1, {heads} {'head' if heads == 1 else 'heads'}"
        elif heads != previous[1]:
            suffix = f"pruned to {heads} {'head' if heads == 1 else 'heads'}"
        elif top_k != previous[0]:
            suffix = f"top-{top_k}"
        else:
            suffix = ""
        labels.append(
            f"{generation:,}" + (f"  |  {suffix}" if suffix else "")
        )
        previous = structure
    return generations, labels


def plot_sweep(
    input_csv: Path,
    *,
    output_png: Path,
    output_svg: Path,
) -> None:
    with input_csv.open(newline="") as source:
        rows = list(csv.DictReader(source))
    generations, y_labels = checkpoint_axis_labels(rows)
    lengths = sorted({int(row["length"]) for row in rows})
    values = {
        (int(row["generation"]), int(row["length"])): float(row["accuracy"])
        for row in rows
    }
    matrix = torch.tensor(
        [
            [values.get((generation, length), math.nan) for length in lengths]
            for generation in generations
        ]
    ).numpy()

    figure, axis = plt.subplots(figsize=(15.5, 10.5), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(range(len(lengths)), [f"{length:,}" for length in lengths])
    axis.tick_params(axis="x", rotation=45)
    axis.set_yticks(range(len(generations)), y_labels)
    axis.set_xlabel(
        "Evaluation list length (training range: 2-20; right of line is OOD)"
    )
    axis.set_ylabel("Checkpoint generation and routing structure")
    axis.set_title(
        "Hard-attention EGGROLL checkpoint sweep",
        loc="left",
        fontweight="bold",
    )
    train_boundary = max(
        index for index, length in enumerate(lengths) if length <= 20
    )
    axis.axvline(train_boundary + 0.5, color="white", linewidth=1.5, alpha=0.9)
    for row_index in range(len(generations)):
        for column_index in range(len(lengths)):
            accuracy = matrix[row_index, column_index]
            if math.isfinite(float(accuracy)) and accuracy < 0.995:
                axis.text(
                    column_index,
                    row_index,
                    f"{accuracy * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if accuracy < 0.65 else "black",
                )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Exact-match accuracy")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.text(
        1.0,
        -0.12,
        "Cell labels show accuracy (%) when below 99.5%.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=220)
    figure.savefig(output_svg)
    plt.close(figure)
    output_svg.write_text(
        "\n".join(line.rstrip() for line in output_svg.read_text().splitlines())
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", action="append", default=[])
    parser.add_argument("--extra-checkpoint", action="append", default=[])
    parser.add_argument("--lengths", type=parse_integer_tuple, default=DEFAULT_LENGTHS)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=27_007)
    parser.add_argument("--short-examples", type=int, default=256)
    parser.add_argument("--long-examples", type=int, default=64)
    parser.add_argument("--long-min-length", type=int, default=600)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--score-element-budget", type=int, default=32_000_000)
    parser.add_argument("--maximum-batch-size", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    output_svg = Path(args.output_svg)
    if not args.plot_only:
        specs = discover_checkpoints(
            tuple(Path(path) for path in args.checkpoint_dir),
            tuple(Path(path) for path in args.extra_checkpoint),
        )
        evaluate_sweep(
            specs,
            lengths=args.lengths,
            output_csv=output_csv,
            device=torch.device(args.device),
            seed=args.seed,
            short_examples=args.short_examples,
            long_examples=args.long_examples,
            long_min_length=args.long_min_length,
            query_chunk_size=args.query_chunk_size,
            score_element_budget=args.score_element_budget,
            maximum_batch_size=args.maximum_batch_size,
        )
    plot_sweep(output_csv, output_png=output_png, output_svg=output_svg)
    print(f"CSV: {output_csv}", flush=True)
    print(f"PNG: {output_png}", flush=True)
    print(f"SVG: {output_svg}", flush=True)


if __name__ == "__main__":
    main()
