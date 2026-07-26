"""Inspect token-role suppression in learned attention-routing checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
    ShortcutBatch,
    ShortcutPointerVocabulary,
    make_shortcut_batch,
)
from .tokens import SEP


ROLE_LABELS = (
    "hint",
    "leak marker",
    "separator",
    "pointer",
    "pointer value",
    "target value",
    "query self",
    "all query sources",
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL=PATH`` checkpoint argument."""

    try:
        label, raw_path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint must use LABEL=PATH"
        ) from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "checkpoint must use LABEL=PATH"
        )
    return label, Path(raw_path)


def load_attention_router(path: Path) -> AttentionRoutingRule:
    """Restore an attention-routing rule without a forward model."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("backward_rule_type") != "attention_router":
        raise ValueError(f"{path} is not an attention-router checkpoint")
    config = AttentionRoutingRuleConfig(
        **checkpoint["backward_rule_config"]
    )
    rule = AttentionRoutingRule(config)
    rule.load_state_dict(checkpoint["backward_rule_state"])
    # PyTorch 2.0's CPU inference fast path does not support bias-free MHA.
    # This block has no dropout, so training mode leaves its output unchanged.
    rule.train()
    return rule


def query_role_gates(
    attention_gates: Tensor,
    batch: ShortcutBatch,
    vocabulary: ShortcutPointerVocabulary,
) -> dict[str, float]:
    """Summarize final-query gates for semantically meaningful source tokens."""

    if attention_gates.ndim != 4:
        raise ValueError("attention gates must have shape [B, H, T, T]")
    if attention_gates.shape[0] != batch.batch_size:
        raise ValueError("attention gate and batch sizes differ")
    gates = attention_gates.mean(dim=1)
    batch_size, sequence_length, _ = gates.shape
    rows = torch.arange(batch_size, device=gates.device)
    pointer_positions = (
        batch.input_ids.eq(vocabulary.marker_token("PTR"))
        .nonzero(as_tuple=False)[:, 1]
        .to(gates.device)
    )
    if pointer_positions.numel() != batch_size:
        raise ValueError("every prompt must contain exactly one pointer")
    leak_positions = (
        batch.input_ids.eq(vocabulary.leak_token)
        .nonzero(as_tuple=False)[:, 1]
        .to(gates.device)
    )
    separator_positions = (
        batch.input_ids.eq(SEP)
        .nonzero(as_tuple=False)[:, 1]
        .to(gates.device)
    )
    if leak_positions.numel() != batch_size:
        raise ValueError("every prompt must contain exactly one leak marker")
    if separator_positions.numel() != batch_size:
        raise ValueError("every prompt must contain exactly one separator")
    query = sequence_length - 1
    values = (
        gates[rows, query, leak_positions + 1].mean(),
        gates[rows, query, leak_positions].mean(),
        gates[rows, query, separator_positions].mean(),
        gates[rows, query, pointer_positions].mean(),
        gates[rows, query, pointer_positions + 1].mean(),
        gates[rows, query, pointer_positions + 3].mean(),
        gates[:, query, query].mean(),
        gates[:, query, :].mean(),
    )
    return {
        label: float(value)
        for label, value in zip(ROLE_LABELS, values)
    }


def position_matched_role_gates(
    attention_gates: Tensor,
    batch: ShortcutBatch,
    vocabulary: ShortcutPointerVocabulary,
) -> dict[str, float]:
    """Give expected gates at each variable role's absolute positions."""

    if attention_gates.ndim != 4:
        raise ValueError("attention gates must have shape [B, H, T, T]")
    gates = attention_gates.mean(dim=1)
    pointer_positions = (
        batch.input_ids.eq(vocabulary.marker_token("PTR"))
        .nonzero(as_tuple=False)[:, 1]
        .to(gates.device)
    )
    if pointer_positions.numel() != batch.batch_size:
        raise ValueError("every prompt must contain exactly one pointer")
    leak_positions = (
        batch.input_ids.eq(vocabulary.leak_token)
        .nonzero(as_tuple=False)[:, 1]
        .to(gates.device)
    )
    if leak_positions.numel() != batch.batch_size:
        raise ValueError("every prompt must contain exactly one leak marker")
    final_query_gates = gates[:, -1]
    position_means = final_query_gates.mean(dim=0)
    return {
        "hint": float(position_means[leak_positions + 1].mean()),
        "leak marker": float(position_means[leak_positions].mean()),
        "pointer": float(position_means[pointer_positions].mean()),
        "pointer value": float(
            position_means[pointer_positions + 1].mean()
        ),
        "target value": float(
            position_means[pointer_positions + 3].mean()
        ),
    }


@torch.no_grad()
def summarize_rule(
    rule: AttentionRoutingRule,
    batch: ShortcutBatch,
    vocabulary: ShortcutPointerVocabulary,
) -> dict[str, float]:
    """Evaluate one shared map and summarize its first layer."""

    gates = rule.attention_gates(batch.input_ids)[0]
    return query_role_gates(gates, batch, vocabulary)


def plot_routing_roles(
    summaries: Sequence[tuple[str, dict[str, float]]],
    output_path: Path,
) -> None:
    """Plot backward gates by semantic source role."""

    if not summaries:
        raise ValueError("at least one summary is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = torch.arange(len(ROLE_LABELS), dtype=torch.float32).numpy()
    width = 0.8 / len(summaries)
    for index, (label, summary) in enumerate(summaries):
        offset = (index - (len(summaries) - 1) / 2) * width
        axis.bar(
            x + offset,
            [summary[role] for role in ROLE_LABELS],
            width=width,
            label=label,
        )
    axis.axhline(
        1.0,
        color="#666666",
        linestyle=":",
        linewidth=1.0,
        label="unchanged backward edge",
    )
    axis.set_xticks(x, ROLE_LABELS, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Backward gate (lower means stronger suppression)")
    axis.set_title("Final-query routing by source-token role")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=parse_checkpoint,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument(
        "--leak-mode",
        choices=("correct", "masked", "incorrect"),
        default="correct",
    )
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="suffix",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    generator = torch.Generator().manual_seed(args.seed)
    batch = make_shortcut_batch(
        args.examples,
        args.length,
        leak_mode=args.leak_mode,
        generator=generator,
        vocabulary=vocabulary,
        leak_placement=args.leak_placement,
    )
    summaries = []
    matched_summaries = []
    for label, path in args.checkpoint:
        rule = load_attention_router(path)
        gates = rule.attention_gates(batch.input_ids)[0]
        summaries.append(
            (label, query_role_gates(gates, batch, vocabulary))
        )
        matched_summaries.append(
            (
                label,
                position_matched_role_gates(
                    gates,
                    batch,
                    vocabulary,
                ),
            )
        )
    plot_routing_roles(summaries, args.output)
    for (label, summary), (_, matched) in zip(
        summaries,
        matched_summaries,
    ):
        values = " ".join(
            f"{role}={summary[role]:.3f}"
            for role in ROLE_LABELS
        )
        print(f"{label}: {values}")
        matched_values = " ".join(
            f"{role}={matched[role]:.3f} "
            f"(actual/matched={summary[role] / matched[role]:.3f})"
            for role in matched
        )
        print(f"{label} position-matched: {matched_values}")
    print(args.output)


if __name__ == "__main__":
    main()
