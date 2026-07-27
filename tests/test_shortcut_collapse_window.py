from dataclasses import replace

import pytest
import torch

from list_sorting_transformer.shortcut_collapse_window import (
    capture_collapse_window,
    clone_state_tree,
    load_collapse_window,
    replay_collapse_window,
    save_collapse_window,
    slice_collapse_window,
)
from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_fitness_batches,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_fresh_backward_rule,
)


def tiny_config() -> ShortcutCreditExperimentConfig:
    return ShortcutCreditExperimentConfig(
        generations=1,
        population_size=2,
        horizon=3,
        max_horizon=3,
        plateau_patience=10,
        batch_size=4,
        fitness_examples=16,
        fitness_batch_size=8,
        correct_eval_examples=8,
        min_length=3,
        max_length=5,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=4,
        backward_rule_type="attention_router",
    )


def test_clone_state_tree_breaks_tensor_aliases() -> None:
    source = {"items": [torch.tensor([1.0]), (torch.tensor([2.0]),)]}
    cloned = clone_state_tree(source, device="cpu")
    source["items"][0].add_(10)
    source["items"][1][0].add_(10)
    assert cloned["items"][0].item() == 1.0
    assert cloned["items"][1][0].item() == 2.0


def test_capture_replay_restores_model_and_adam_state(tmp_path) -> None:
    config = tiny_config()
    device = torch.device("cpu")
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rule = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=device,
    )
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    fitness_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    window = capture_collapse_window(
        config,
        backward_rule=rule,
        generation_seed=1234,
        start_step=5,
        window_steps=3,
        fitness_batches=fitness_batches,
        device=device,
    )
    replay = replay_collapse_window(
        config,
        window=window,
        backward_rule=rule,
        fitness_batches=fitness_batches,
        device=device,
        checkpoint_steps=(1, 3),
    )
    assert replay.end_metrics.loss == pytest.approx(
        window.center_end_metrics.loss,
        abs=1e-7,
    )
    assert replay.end_metrics.mode_accuracy == (
        window.center_end_metrics.mode_accuracy
    )

    path = tmp_path / "window.pt"
    save_collapse_window(path, window)
    loaded = load_collapse_window(path)
    loaded_replay = replay_collapse_window(
        replace(config, horizon=1),
        window=loaded,
        backward_rule=rule,
        fitness_batches=fitness_batches,
        device=device,
    )
    assert loaded_replay.end_metrics.loss == pytest.approx(
        window.center_end_metrics.loss,
        abs=1e-7,
    )


def test_slice_window_preserves_exact_center_trajectory() -> None:
    config = tiny_config()
    device = torch.device("cpu")
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    rule = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=device,
    )
    generator = torch.Generator().manual_seed(config.seed + 10_000)
    fitness_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    window = capture_collapse_window(
        config,
        backward_rule=rule,
        generation_seed=4321,
        start_step=2,
        window_steps=5,
        fitness_batches=fitness_batches,
        device=device,
    )
    full_replay = replay_collapse_window(
        config,
        window=window,
        backward_rule=rule,
        fitness_batches=fitness_batches,
        device=device,
        checkpoint_steps=(2, 5),
    )

    sliced = slice_collapse_window(
        config,
        window=window,
        backward_rule=rule,
        fitness_batches=fitness_batches,
        start_offset=2,
        window_steps=3,
        device=device,
    )
    sliced_replay = replay_collapse_window(
        config,
        window=sliced,
        backward_rule=rule,
        fitness_batches=fitness_batches,
        device=device,
        checkpoint_steps=(3,),
    )

    assert sliced.start_step == 4
    assert sliced.start_metrics.loss == pytest.approx(
        full_replay.checkpoint_metrics[0][1].loss,
        abs=1e-7,
    )
    assert sliced_replay.end_metrics.loss == pytest.approx(
        full_replay.checkpoint_metrics[1][1].loss,
        abs=1e-7,
    )
    assert sliced_replay.end_metrics.mode_accuracy == (
        full_replay.checkpoint_metrics[1][1].mode_accuracy
    )
