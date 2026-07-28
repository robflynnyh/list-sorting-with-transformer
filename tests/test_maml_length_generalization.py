from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from list_sorting_transformer.maml_length_generalization import (
    MAMLLengthConfig,
    load_ordinary_reference,
    make_meta_batches,
    make_model,
    make_router,
    one_step_maml_objective,
    one_step_router_maml_objective,
    parse_meta_lengths,
    run,
    select_meta_parameters,
)
from list_sorting_transformer.shortcut_credit import make_clean_pointer_batch
from list_sorting_transformer.tokens import PointerNextVocabulary


def test_one_step_maml_keeps_virtual_weights_temporary() -> None:
    config = MAMLLengthConfig(
        steps=1,
        batch_size=4,
        max_length=4,
        meta_lengths="6",
        meta_examples=4,
        meta_batch_size=4,
        heldout_length=8,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
    )
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    original_state = deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(17)
    short_batch = make_clean_pointer_batch(
        4,
        4,
        generator=generator,
        vocabulary=vocabulary,
    )
    meta_batch = make_clean_pointer_batch(
        4,
        6,
        generator=generator,
        vocabulary=vocabulary,
    )

    objective = one_step_maml_objective(
        model,
        short_batch,
        meta_batch,
        inner_learning_rate=1e-2,
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])
    objective.meta_loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(objective.meta_loss)


def test_maml_meta_gradient_includes_the_virtual_short_step() -> None:
    config = MAMLLengthConfig(
        steps=1,
        batch_size=4,
        max_length=4,
        meta_lengths="6",
        meta_examples=4,
        meta_batch_size=4,
        heldout_length=8,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
    )
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(23)
    short_batch = make_clean_pointer_batch(
        4,
        4,
        generator=generator,
        vocabulary=vocabulary,
    )
    meta_batch = make_clean_pointer_batch(
        4,
        6,
        generator=generator,
        vocabulary=vocabulary,
    )

    objective = one_step_maml_objective(
        model,
        short_batch,
        meta_batch,
        inner_learning_rate=0.1,
    )
    maml_gradients = torch.autograd.grad(
        objective.meta_loss,
        tuple(model.parameters()),
    )
    direct_meta_loss = torch.nn.functional.cross_entropy(
        model(meta_batch.input_ids)[:, -1],
        meta_batch.targets,
    )
    direct_gradients = torch.autograd.grad(
        direct_meta_loss,
        tuple(model.parameters()),
    )

    difference = sum(
        float((maml - direct).square().sum())
        for maml, direct in zip(maml_gradients, direct_gradients)
    )
    assert difference > 1e-8


def test_meta_batches_preserve_repeated_length_weighting() -> None:
    config = MAMLLengthConfig(
        steps=1,
        batch_size=4,
        max_length=4,
        meta_lengths="6,7,7,8",
        meta_examples=4,
        meta_batch_size=2,
        heldout_length=10,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
    )
    batches = make_meta_batches(
        config,
        vocabulary=PointerNextVocabulary("numbers", 10),
        device=torch.device("cpu"),
    )

    assert parse_meta_lengths(config.meta_lengths) == (6, 7, 7, 8)
    assert [batch.length for batch in batches] == [
        6,
        7,
        7,
        8,
        6,
        7,
        7,
        8,
    ]


def test_qkv_meta_scope_selects_only_attention_qkv_weights() -> None:
    config = MAMLLengthConfig(
        d_model=16,
        layers=2,
        heads=1,
    )
    model = make_model(
        config,
        PointerNextVocabulary("numbers", 10),
        device=torch.device("cpu"),
    )

    names, parameters = select_meta_parameters(model, "qkv")

    assert names == (
        "blocks.0.attention.qkv.weight",
        "blocks.1.attention.qkv.weight",
    )
    assert sum(parameter.numel() for parameter in parameters) == 2 * 3 * 16 * 16


@pytest.mark.parametrize("router_credit_mode", ["suppress_renorm", "signed"])
def test_router_meta_gradient_flows_through_virtual_model_step(
    router_credit_mode: str,
) -> None:
    config = MAMLLengthConfig(
        method="router_maml",
        router_credit_mode=router_credit_mode,
        steps=1,
        batch_size=4,
        max_length=4,
        meta_lengths="6",
        meta_examples=4,
        meta_batch_size=4,
        heldout_length=8,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
        router_d_model=16,
        router_heads=1,
        router_initial_gate=0.1,
    )
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    router = make_router(config, vocabulary, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(31)
    short_batch = make_clean_pointer_batch(
        4,
        4,
        generator=generator,
        vocabulary=vocabulary,
    )
    meta_batch = make_clean_pointer_batch(
        4,
        6,
        generator=generator,
        vocabulary=vocabulary,
    )

    ordinary_logits = model(short_batch.input_ids)
    routed_logits = model.forward_with_backward_rule(
        short_batch.input_ids,
        router,
    )
    torch.testing.assert_close(routed_logits, ordinary_logits)
    objective = one_step_router_maml_objective(
        model,
        router,
        short_batch,
        meta_batch,
        inner_learning_rate=1e-2,
    )
    gradients = torch.autograd.grad(
        objective.meta_loss,
        tuple(router.parameters()),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.square().sum()) for gradient in gradients) > 0


def test_ordinary_reference_exposes_length_50_and_400(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "ordinary.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "step": 100,
                "eval/length_50/accuracy": 0.9,
                "eval/length_50/loss": 0.2,
                "eval/length_400/accuracy": 0.15,
                "eval/length_400/loss": 4.0,
            }
        )
        + "\n"
    )

    assert load_ordinary_reference(str(metrics_path)) == {
        100: {
            "ordinary_reference/length_50/accuracy": 0.9,
            "ordinary_reference/length_50/loss": 0.2,
            "ordinary_reference/length_400/accuracy": 0.15,
            "ordinary_reference/length_400/loss": 4.0,
        }
    }


def test_ordinary_mode_reports_length_50_and_heldout(
    tmp_path: Path,
) -> None:
    output_dir = run(
        MAMLLengthConfig(
            run_name="ordinary-smoke",
            output_dir=str(tmp_path),
            method="ordinary",
            steps=1,
            batch_size=4,
            max_length=4,
            meta_lengths="6",
            meta_examples=4,
            meta_batch_size=4,
            heldout_length=8,
            eval_examples=4,
            eval_batch_size=4,
            d_model=16,
            layers=1,
            heads=1,
            log_interval=1,
            eval_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]

    assert "eval/length_50/accuracy" in rows[-1]
    assert "eval/length_8/accuracy" in rows[-1]
    assert not any(key.startswith("train/meta_") for key in rows[-1])


def test_router_maml_mode_persists_model_and_router_checkpoints(
    tmp_path: Path,
) -> None:
    output_dir = run(
        MAMLLengthConfig(
            run_name="router-maml-smoke",
            output_dir=str(tmp_path),
            method="router_maml",
            steps=1,
            batch_size=4,
            max_length=4,
            meta_lengths="6",
            meta_examples=4,
            meta_batch_size=4,
            heldout_length=8,
            eval_examples=4,
            eval_batch_size=4,
            d_model=16,
            layers=1,
            heads=1,
            router_d_model=16,
            router_heads=1,
            router_meta_updates_per_step=2,
            log_interval=1,
            eval_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
    row = json.loads(
        (output_dir / "metrics.jsonl").read_text().splitlines()[-1]
    )

    assert checkpoint["router"] is not None
    assert checkpoint["step"] == 1
    assert {
        int(state["step"])
        for state in checkpoint["meta_optimizer"]["state"].values()
    } == {2}
    assert {
        int(state["step"])
        for state in checkpoint["ordinary_optimizer"]["state"].values()
    } == {1}
    assert row["gradient/meta_parameter_count"] > 0
    assert row["train/router_meta_updates_per_step"] == 2
    assert "router/backward_multiplier_mean" in row
