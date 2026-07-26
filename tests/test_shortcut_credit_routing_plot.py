from __future__ import annotations

from pathlib import Path

import pytest
import torch

from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_shortcut_batch,
)
from list_sorting_transformer.shortcut_credit_routing_plot import (
    ROLE_LABELS,
    parse_checkpoint,
    plot_routing_roles,
    query_role_gates,
)


def test_parse_checkpoint_requires_label_and_path() -> None:
    assert parse_checkpoint("centre=checkpoint.pt") == (
        "centre",
        Path("checkpoint.pt"),
    )
    with pytest.raises(Exception):
        parse_checkpoint("checkpoint.pt")


def test_query_role_gates_extracts_prompt_positions() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    batch = make_shortcut_batch(
        4,
        5,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(3),
        vocabulary=vocabulary,
    )
    sequence_length = batch.input_ids.shape[1]
    gates = torch.ones(4, 2, sequence_length, sequence_length)
    rows = torch.arange(4)
    pointers = (
        batch.input_ids.eq(vocabulary.marker_token("PTR"))
        .nonzero(as_tuple=False)[:, 1]
    )
    query = sequence_length - 1
    gates[rows, :, query, query - 1] = 0.1
    gates[rows, :, query, pointers] = 0.4
    gates[rows, :, query, pointers + 3] = 0.7

    summary = query_role_gates(gates, batch, vocabulary)

    assert summary["hint"] == pytest.approx(0.1)
    assert summary["pointer"] == pytest.approx(0.4)
    assert summary["target value"] == pytest.approx(0.7)
    assert set(summary) == set(ROLE_LABELS)


def test_plot_routing_roles(tmp_path: Path) -> None:
    output = tmp_path / "routing.png"
    summary = {label: 0.5 for label in ROLE_LABELS}

    plot_routing_roles(
        [("start", summary), ("learned", summary)],
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 1_000
