"""Capture and replay short optimizer windows around shortcut collapses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .shortcut_credit import (
    BackwardRule,
    ShortcutBatch,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    shortcut_loss,
)
from .shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    make_inner_batches,
)


StateTree = Any


@dataclass(frozen=True)
class CollapseWindow:
    generation_seed: int
    start_step: int
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Any]
    batches: tuple[ShortcutBatch, ...]
    start_metrics: ShortcutMetrics
    center_end_metrics: ShortcutMetrics

    @property
    def end_step(self) -> int:
        return self.start_step + len(self.batches)


@dataclass(frozen=True)
class CollapseWindowReplay:
    end_metrics: ShortcutMetrics
    checkpoint_metrics: tuple[tuple[int, ShortcutMetrics], ...]
    model_state: dict[str, Tensor]


def clone_state_tree(
    value: StateTree,
    *,
    device: torch.device | str,
) -> StateTree:
    """Clone a nested optimizer state without retaining device aliases."""

    if isinstance(value, Tensor):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {
            key: clone_state_tree(item, device=device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [clone_state_tree(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(
            clone_state_tree(item, device=device) for item in value
        )
    return value


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, Tensor):
                parameter_state[name] = value.to(device)


def optimizer_for_model(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    state: dict[str, Any] | None = None,
    device: torch.device,
) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    if state is not None:
        optimizer.load_state_dict(
            clone_state_tree(state, device=device)
        )
        move_optimizer_state(optimizer, device)
    return optimizer


def train_forward_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: ShortcutBatch,
    backward_rule: BackwardRule,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = shortcut_loss(model, batch, backward_rule)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def capture_collapse_window(
    config: ShortcutCreditExperimentConfig,
    *,
    backward_rule: BackwardRule,
    generation_seed: int,
    start_step: int,
    window_steps: int,
    fitness_batches: tuple[ShortcutBatch, ...],
    device: torch.device,
    evaluation_batch_size: int = 64,
) -> CollapseWindow:
    if start_step < 0:
        raise ValueError("collapse window start must be nonnegative")
    if window_steps < 1:
        raise ValueError("collapse window must contain at least one step")

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=generation_seed + 1,
        device=device,
    )
    optimizer = optimizer_for_model(
        model,
        learning_rate=config.forward_learning_rate,
        device=device,
    )
    generator = torch.Generator().manual_seed(generation_seed + 2)
    backward_rule.capture_statistics = False

    for _ in range(start_step):
        batch = make_inner_batches(
            config,
            horizon=1,
            vocabulary=vocabulary,
            generator=generator,
            device=device,
        )[0]
        train_forward_step(model, optimizer, batch, backward_rule)

    start_metrics = evaluate_shortcut_batches(
        model,
        fitness_batches,
        evaluation_batch_size=evaluation_batch_size,
    )
    model_state = {
        name: tensor.detach().clone().cpu()
        for name, tensor in model.state_dict().items()
    }
    optimizer_state = clone_state_tree(
        optimizer.state_dict(),
        device="cpu",
    )

    batches = []
    for _ in range(window_steps):
        batch = make_inner_batches(
            config,
            horizon=1,
            vocabulary=vocabulary,
            generator=generator,
            device=device,
        )[0]
        batches.append(batch.to("cpu"))
        train_forward_step(model, optimizer, batch, backward_rule)
    center_end_metrics = evaluate_shortcut_batches(
        model,
        fitness_batches,
        evaluation_batch_size=evaluation_batch_size,
    )

    return CollapseWindow(
        generation_seed=generation_seed,
        start_step=start_step,
        model_state=model_state,
        optimizer_state=optimizer_state,
        batches=tuple(batches),
        start_metrics=start_metrics,
        center_end_metrics=center_end_metrics,
    )


def replay_collapse_window(
    config: ShortcutCreditExperimentConfig,
    *,
    window: CollapseWindow,
    backward_rule: BackwardRule,
    fitness_batches: tuple[ShortcutBatch, ...],
    device: torch.device,
    checkpoint_steps: tuple[int, ...] = (),
    evaluation_batch_size: int = 64,
) -> CollapseWindowReplay:
    if any(
        step < 1 or step > len(window.batches)
        for step in checkpoint_steps
    ):
        raise ValueError("window checkpoints must index a replay step")

    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(window.model_state)
    optimizer = optimizer_for_model(
        model,
        learning_rate=config.forward_learning_rate,
        state=window.optimizer_state,
        device=device,
    )
    checkpoint_set = set(checkpoint_steps)
    checkpoint_metrics = []
    backward_rule.capture_statistics = False
    for step, batch in enumerate(window.batches, start=1):
        train_forward_step(
            model,
            optimizer,
            batch.to(device),
            backward_rule,
        )
        if step in checkpoint_set:
            checkpoint_metrics.append(
                (
                    step,
                    evaluate_shortcut_batches(
                        model,
                        fitness_batches,
                        evaluation_batch_size=evaluation_batch_size,
                    ),
                )
            )

    end_metrics = (
        checkpoint_metrics[-1][1]
        if checkpoint_metrics
        and checkpoint_metrics[-1][0] == len(window.batches)
        else evaluate_shortcut_batches(
            model,
            fitness_batches,
            evaluation_batch_size=evaluation_batch_size,
        )
    )
    model_state = {
        name: tensor.detach().clone().cpu()
        for name, tensor in model.state_dict().items()
    }
    return CollapseWindowReplay(
        end_metrics=end_metrics,
        checkpoint_metrics=tuple(checkpoint_metrics),
        model_state=model_state,
    )


def metrics_from_dict(values: dict[str, Any]) -> ShortcutMetrics:
    return ShortcutMetrics(**values)


def window_to_dict(window: CollapseWindow) -> dict[str, Any]:
    return {
        "format": "shortcut-collapse-window-v1",
        "generation_seed": window.generation_seed,
        "start_step": window.start_step,
        "model_state": window.model_state,
        "optimizer_state": window.optimizer_state,
        "batches": [
            {
                "input_ids": batch.input_ids,
                "targets": batch.targets,
                "length": batch.length,
                "leak_mode": batch.leak_mode,
                "leak_placement": batch.leak_placement,
            }
            for batch in window.batches
        ],
        "start_metrics": asdict(window.start_metrics),
        "center_end_metrics": asdict(window.center_end_metrics),
    }


def window_from_dict(values: dict[str, Any]) -> CollapseWindow:
    if values.get("format") != "shortcut-collapse-window-v1":
        raise ValueError("unknown collapse window format")
    return CollapseWindow(
        generation_seed=int(values["generation_seed"]),
        start_step=int(values["start_step"]),
        model_state=values["model_state"],
        optimizer_state=values["optimizer_state"],
        batches=tuple(ShortcutBatch(**batch) for batch in values["batches"]),
        start_metrics=metrics_from_dict(values["start_metrics"]),
        center_end_metrics=metrics_from_dict(
            values["center_end_metrics"]
        ),
    )


def save_collapse_window(path: Path, window: CollapseWindow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(window_to_dict(window), path)


def load_collapse_window(path: Path) -> CollapseWindow:
    values = torch.load(path, map_location="cpu", weights_only=False)
    return window_from_dict(values)
