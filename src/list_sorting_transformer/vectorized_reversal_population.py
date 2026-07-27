"""Vectorized forward-model populations for token-reversal experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from .shortcut_credit import (
    ShortcutBatch,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
)
from .shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
)
from .token_gradient_reversal import (
    forward_with_source_gradient_reversal,
)
from .tokens import VALUE_OFFSET


@dataclass(frozen=True)
class VectorizedCandidateResult:
    candidate_index: int
    reward: float
    clean: ShortcutMetrics
    heldout_clean: ShortcutMetrics
    correct: ShortcutMetrics


class _ScoreReversalModel(nn.Module):
    def __init__(self, model: nn.Module, reversal_scale: float) -> None:
        super().__init__()
        self.model = model
        self.reversal_scale = reversal_scale

    def forward(self, token_ids: Tensor, selection: Tensor) -> Tensor:
        return forward_with_source_gradient_reversal(
            self.model,
            token_ids,
            selection,
            reversal_scale=self.reversal_scale,
            reversal_scope="attention_scores",
        )


def _stack_parameters(
    model: nn.Module,
    population_size: int,
) -> dict[str, Tensor]:
    return {
        name: parameter.detach()
        .unsqueeze(0)
        .expand(population_size, *parameter.shape)
        .clone()
        for name, parameter in model.named_parameters()
    }


def functional_adam_step(
    parameters: dict[str, Tensor],
    gradients: dict[str, Tensor],
    first_moments: dict[str, Tensor],
    second_moments: dict[str, Tensor],
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Apply one ordinary Adam update to a stacked parameter population."""

    if step < 1 or learning_rate <= 0:
        raise ValueError("step and learning_rate must be positive")
    first_correction = 1.0 - beta1**step
    second_correction_sqrt = (1.0 - beta2**step) ** 0.5
    updated_parameters = {}
    updated_first = {}
    updated_second = {}
    with torch.no_grad():
        for name, parameter in parameters.items():
            gradient = gradients[name]
            first = (
                beta1 * first_moments[name]
                + (1.0 - beta1) * gradient
            )
            second = (
                beta2 * second_moments[name]
                + (1.0 - beta2) * gradient.square()
            )
            denominator = (
                second.sqrt() / second_correction_sqrt
            ).add(epsilon)
            updated_parameters[name] = parameter - (
                learning_rate / first_correction
            ) * first / denominator
            updated_first[name] = first
            updated_second[name] = second
    return updated_parameters, updated_first, updated_second


def _candidate_loss(
    parameters: dict[str, Tensor],
    *,
    model: nn.Module,
    buffers: dict[str, Tensor],
    input_ids: Tensor,
    selection: Tensor,
    targets: Tensor,
) -> Tensor:
    logits = functional_call(
        model,
        (parameters, buffers),
        (input_ids, selection),
    )
    return F.cross_entropy(logits[:, -1], targets)


def _population_metrics(
    model: nn.Module,
    parameters: dict[str, Tensor],
    buffers: dict[str, Tensor],
    batches: tuple[ShortcutBatch, ...],
    *,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
) -> tuple[ShortcutMetrics, ...]:
    population_size = next(iter(parameters.values())).shape[0]
    total_loss = torch.zeros(population_size, device=device)
    total_correct = torch.zeros(
        population_size,
        dtype=torch.long,
        device=device,
    )
    mode_loss: dict[str, Tensor] = defaultdict(
        lambda: torch.zeros(population_size, device=device)
    )
    mode_correct: dict[str, Tensor] = defaultdict(
        lambda: torch.zeros(
            population_size,
            dtype=torch.long,
            device=device,
        )
    )
    mode_examples: dict[str, int] = defaultdict(int)
    prediction_counts = torch.zeros(
        population_size,
        vocabulary.size,
        dtype=torch.long,
        device=device,
    )
    total_examples = 0

    def candidate_logits(
        candidate_parameters: dict[str, Tensor],
        selection: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        return functional_call(
            model,
            (candidate_parameters, buffers),
            (input_ids, selection),
        )[:, -1]

    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        selections = torch.zeros(
            population_size,
            *batch.input_ids.shape,
            dtype=torch.bool,
            device=device,
        )
        logits = vmap(
            candidate_logits,
            in_dims=(0, 0, None),
        )(parameters, selections, batch.input_ids)
        targets = batch.targets.unsqueeze(0).expand(
            population_size,
            -1,
        )
        losses = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            reduction="none",
        )
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(targets)
        total_loss += losses.sum(dim=1)
        total_correct += correct.sum(dim=1)
        mode_loss[batch.leak_mode] += losses.sum(dim=1)
        mode_correct[batch.leak_mode] += correct.sum(dim=1)
        mode_examples[batch.leak_mode] += batch.batch_size
        prediction_counts += F.one_hot(
            predictions,
            num_classes=vocabulary.size,
        ).sum(dim=1)
        total_examples += batch.batch_size

    if total_examples == 0:
        raise ValueError("evaluation requires at least one example")
    metrics = []
    for candidate_index in range(population_size):
        candidate_counts = prediction_counts[candidate_index]
        metrics.append(
            ShortcutMetrics(
                loss=float(
                    total_loss[candidate_index] / total_examples
                ),
                accuracy=float(
                    total_correct[candidate_index] / total_examples
                ),
                mode_accuracy={
                    mode: float(
                        mode_correct[mode][candidate_index] / count
                    )
                    for mode, count in mode_examples.items()
                },
                mode_loss={
                    mode: float(
                        mode_loss[mode][candidate_index] / count
                    )
                    for mode, count in mode_examples.items()
                },
                unique_prediction_count=int(
                    candidate_counts.count_nonzero()
                ),
                unique_value_prediction_count=int(
                    candidate_counts[
                        VALUE_OFFSET : (
                            VALUE_OFFSET + vocabulary.symbol_count
                        )
                    ].count_nonzero()
                ),
                prediction_mode_fraction=(
                    float(candidate_counts.max()) / total_examples
                ),
            )
        )
    return tuple(metrics)


def train_vectorized_candidate_shard(
    *,
    candidate_indices: tuple[int, ...],
    device_name: str,
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    actions: tuple[tuple[Tensor, ...], ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    heldout_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_loss: float,
    reversal_scale: float,
    vocabulary: ShortcutPointerVocabulary,
) -> list[VectorizedCandidateResult]:
    """Train one same-architecture candidate shard with ``torch.vmap``."""

    if not candidate_indices:
        return []
    if len(inner_batches) != len(actions[0]):
        raise ValueError("one action tensor is required per inner batch")
    device = torch.device(device_name)
    forward_model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    forward_model.load_state_dict(base_state)
    model = _ScoreReversalModel(
        forward_model,
        reversal_scale,
    ).to(device)
    model.train()
    parameters = _stack_parameters(model, len(candidate_indices))
    buffers = {
        name: buffer.detach()
        for name, buffer in model.named_buffers()
    }
    first_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in parameters.items()
    }
    second_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in parameters.items()
    }
    for step, batch_cpu in enumerate(inner_batches, start=1):
        batch = batch_cpu.to(device)
        selections = torch.stack(
            [
                actions[candidate_index][step - 1]
                for candidate_index in candidate_indices
            ]
        ).to(device)
        def batch_loss(
            candidate_parameters: dict[str, Tensor],
            candidate_selection: Tensor,
        ) -> Tensor:
            return _candidate_loss(
                candidate_parameters,
                model=model,
                buffers=buffers,
                input_ids=batch.input_ids,
                selection=candidate_selection,
                targets=batch.targets,
            )

        gradients = vmap(
            grad(batch_loss),
            in_dims=(0, 0),
            randomness="different",
        )(
            parameters,
            selections,
        )
        parameters, first_moments, second_moments = (
            functional_adam_step(
                parameters,
                gradients,
                first_moments,
                second_moments,
                step=step,
                learning_rate=config.forward_learning_rate,
            )
        )

    model.eval()
    clean = _population_metrics(
        model,
        parameters,
        buffers,
        fitness_batches,
        vocabulary=vocabulary,
        device=device,
    )
    heldout = _population_metrics(
        model,
        parameters,
        buffers,
        heldout_batches,
        vocabulary=vocabulary,
        device=device,
    )
    correct = _population_metrics(
        model,
        parameters,
        buffers,
        correct_batches,
        vocabulary=vocabulary,
        device=device,
    )
    return [
        VectorizedCandidateResult(
            candidate_index=candidate_index,
            reward=initial_clean_loss - clean[local_index].loss,
            clean=clean[local_index],
            heldout_clean=heldout[local_index],
            correct=correct[local_index],
        )
        for local_index, candidate_index in enumerate(candidate_indices)
    ]
