"""Vectorized EGGROLL populations for attention-routing rules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from .shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
    EggrollDirection,
    ShortcutBatch,
    ShortcutMetrics,
    ShortcutPointerVocabulary,
)
from .shortcut_credit_experiment import (
    CandidateRankingInput,
    ForwardTrajectoryMetrics,
    ShortcutCreditExperimentConfig,
    candidate_fitness,
    initialize_forward_model,
    parse_fitness_checkpoints,
)
from .tokens import VALUE_OFFSET
from .vectorized_reversal_population import functional_adam_step

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class VectorizedRoutingPopulation:
    """Final parameters and metrics for one candidate chunk."""

    forward_parameters: dict[str, Tensor]
    trajectories: tuple[ForwardTrajectoryMetrics, ...]


VectorizedCandidateOutput = tuple[
    int,
    float,
    ForwardTrajectoryMetrics,
    list[dict[str, float]],
    tuple[float, ...],
]


class _RoutedForwardModel(nn.Module):
    def __init__(
        self,
        forward_model: nn.Module,
        backward_rule: AttentionRoutingRule,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.backward_rule = backward_rule

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.forward_model.forward_with_backward_rule(
            token_ids,
            self.backward_rule,
        )


def stack_candidate_rule_parameters(
    center_parameters: dict[str, Tensor],
    directions: Sequence[EggrollDirection],
    candidate_specs: Sequence[tuple[int, int, int]],
    *,
    sigma: float,
    device: torch.device,
) -> dict[str, Tensor]:
    """Materialize candidate router parameters in population-major form."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return {
        f"backward_rule.{name}": torch.stack(
            tuple(
                center.to(device)
                + sign
                * sigma
                * directions[direction_index].tensors[name].to(device)
                for _, direction_index, sign in candidate_specs
            )
        )
        for name, center in center_parameters.items()
    }


def _stack_forward_parameters(
    model: _RoutedForwardModel,
    population_size: int,
) -> dict[str, Tensor]:
    return {
        name: parameter.detach()
        .unsqueeze(0)
        .expand(population_size, *parameter.shape)
        .clone()
        for name, parameter in model.named_parameters()
        if name.startswith("forward_model.")
    }


def _merged_parameters(
    forward_parameters: dict[str, Tensor],
    rule_parameters: dict[str, Tensor],
) -> dict[str, Tensor]:
    return {**forward_parameters, **rule_parameters}


def _population_metrics(
    model: _RoutedForwardModel,
    forward_parameters: dict[str, Tensor],
    rule_parameters: dict[str, Tensor],
    buffers: dict[str, Tensor],
    batches: tuple[ShortcutBatch, ...],
    *,
    vocabulary: ShortcutPointerVocabulary,
    device: torch.device,
) -> tuple[ShortcutMetrics, ...]:
    population_size = next(iter(forward_parameters.values())).shape[0]
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
        candidate_forward_parameters: dict[str, Tensor],
        candidate_rule_parameters: dict[str, Tensor],
        input_ids: Tensor,
    ) -> Tensor:
        return functional_call(
            model,
            (
                _merged_parameters(
                    candidate_forward_parameters,
                    candidate_rule_parameters,
                ),
                buffers,
            ),
            (input_ids,),
        )[:, -1]

    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        logits = vmap(
            candidate_logits,
            in_dims=(0, 0, None),
            randomness="different",
        )(
            forward_parameters,
            rule_parameters,
            batch.input_ids,
        )
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


def train_vectorized_routing_population(
    *,
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    center_rule: AttentionRoutingRule,
    rule_parameters: dict[str, Tensor],
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    heldout_fitness_batches: tuple[ShortcutBatch, ...] | None,
    heldout_correct_batches: tuple[ShortcutBatch, ...] | None,
    device: torch.device,
) -> VectorizedRoutingPopulation:
    """Train independent forward models for a stack of frozen routers."""

    if center_rule.config.route_output_projection:
        raise ValueError(
            "vectorized routing does not support output-projection routing"
        )
    if center_rule.config.condition_on_forward_state:
        raise ValueError(
            "vectorized routing does not support forward-state conditioning"
        )
    if (heldout_fitness_batches is None) != (
        heldout_correct_batches is None
    ):
        raise ValueError("both held-out batch groups must be provided together")
    population_size = next(iter(rule_parameters.values())).shape[0]
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    forward_model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    forward_model.load_state_dict(base_state)
    worker_rule = AttentionRoutingRule(center_rule.config).to(device)
    worker_rule.capture_statistics = False
    model = _RoutedForwardModel(forward_model, worker_rule).to(device)
    model.train()
    forward_parameters = _stack_forward_parameters(
        model,
        population_size,
    )
    buffers = {
        name: buffer.detach()
        for name, buffer in model.named_buffers()
    }
    first_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in forward_parameters.items()
    }
    second_moments = {
        name: torch.zeros_like(parameter)
        for name, parameter in forward_parameters.items()
    }
    checkpoint_steps = parse_fitness_checkpoints(
        config.fitness_checkpoints
    )
    checkpoint_step_set = set(checkpoint_steps)
    checkpoint_metrics: dict[int, tuple[ShortcutMetrics, ...]] = {}

    for step, cpu_batch in enumerate(inner_batches, start=1):
        batch = cpu_batch.to(device)

        def candidate_loss(
            candidate_forward_parameters: dict[str, Tensor],
            candidate_rule_parameters: dict[str, Tensor],
        ) -> Tensor:
            logits = functional_call(
                model,
                (
                    _merged_parameters(
                        candidate_forward_parameters,
                        candidate_rule_parameters,
                    ),
                    buffers,
                ),
                (batch.input_ids,),
            )
            return F.cross_entropy(logits[:, -1], batch.targets)

        gradients = vmap(
            grad(candidate_loss, argnums=0),
            in_dims=(0, 0),
            randomness="different",
        )(
            forward_parameters,
            rule_parameters,
        )
        forward_parameters, first_moments, second_moments = (
            functional_adam_step(
                forward_parameters,
                gradients,
                first_moments,
                second_moments,
                step=step,
                learning_rate=config.forward_learning_rate,
            )
        )
        if step in checkpoint_step_set:
            checkpoint_metrics[step] = _population_metrics(
                model,
                forward_parameters,
                rule_parameters,
                buffers,
                fitness_batches,
                vocabulary=vocabulary,
                device=device,
            )

    # Keep the wrapper in training mode during functional evaluation. The
    # experiment uses zero dropout, while PyTorch 2.0's eval-only native MHA
    # fast path cannot consume vmap BatchedTensors with bias disabled.
    clean = (
        checkpoint_metrics[len(inner_batches)]
        if len(inner_batches) in checkpoint_metrics
        else _population_metrics(
            model,
            forward_parameters,
            rule_parameters,
            buffers,
            fitness_batches,
            vocabulary=vocabulary,
            device=device,
        )
    )
    correct = _population_metrics(
        model,
        forward_parameters,
        rule_parameters,
        buffers,
        correct_batches,
        vocabulary=vocabulary,
        device=device,
    )
    heldout_clean = (
        None
        if heldout_fitness_batches is None
        else _population_metrics(
            model,
            forward_parameters,
            rule_parameters,
            buffers,
            heldout_fitness_batches,
            vocabulary=vocabulary,
            device=device,
        )
    )
    heldout_correct = (
        None
        if heldout_correct_batches is None
        else _population_metrics(
            model,
            forward_parameters,
            rule_parameters,
            buffers,
            heldout_correct_batches,
            vocabulary=vocabulary,
            device=device,
        )
    )
    trajectories = []
    for index in range(population_size):
        trajectories.append(
            ForwardTrajectoryMetrics(
                clean=clean[index],
                correct=correct[index],
                heldout_clean=(
                    None
                    if heldout_clean is None
                    else heldout_clean[index]
                ),
                heldout_correct=(
                    None
                    if heldout_correct is None
                    else heldout_correct[index]
                ),
                checkpoint_clean=tuple(
                    (step, checkpoint_metrics[step][index])
                    for step in checkpoint_steps
                ),
            )
        )
    return VectorizedRoutingPopulation(
        forward_parameters=forward_parameters,
        trajectories=tuple(trajectories),
    )


def train_vectorized_routing_candidate_chunks(
    *,
    config: ShortcutCreditExperimentConfig,
    candidate_specs: tuple[tuple[int, int, int], ...],
    chunk_size: int,
    device: torch.device,
    base_state: dict[str, Tensor],
    center_rule_config: AttentionRoutingRuleConfig,
    center_parameters: dict[str, Tensor],
    directions: tuple[EggrollDirection, ...],
    inner_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    correct_batches: tuple[ShortcutBatch, ...],
    heldout_fitness_batches: tuple[ShortcutBatch, ...],
    heldout_correct_batches: tuple[ShortcutBatch, ...],
    initial_clean_metrics: ShortcutMetrics,
    perturbation_sigma: float,
    additional_ranking_inputs: tuple[CandidateRankingInput, ...] = (),
) -> list[VectorizedCandidateOutput]:
    """Evaluate bounded candidate chunks with the old runner's output shape."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not candidate_specs:
        return []
    if device.type == "cuda":
        torch.cuda.set_device(device)
    center_rule = AttentionRoutingRule(center_rule_config).to(device)
    worker_base_state = {
        name: tensor.to(device)
        for name, tensor in base_state.items()
    }
    worker_inner_batches = tuple(batch.to(device) for batch in inner_batches)
    worker_fitness_batches = tuple(
        batch.to(device) for batch in fitness_batches
    )
    worker_correct_batches = tuple(
        batch.to(device) for batch in correct_batches
    )
    worker_heldout_fitness_batches = tuple(
        batch.to(device) for batch in heldout_fitness_batches
    )
    worker_heldout_correct_batches = tuple(
        batch.to(device) for batch in heldout_correct_batches
    )
    worker_additional_ranking_inputs = tuple(
        (
            {
                name: tensor.to(device)
                for name, tensor in ranking_base_state.items()
            },
            tuple(batch.to(device) for batch in ranking_inner_batches),
            ranking_initial_clean,
        )
        for (
            ranking_base_state,
            ranking_inner_batches,
            ranking_initial_clean,
        ) in additional_ranking_inputs
    )
    results = []
    for start in range(0, len(candidate_specs), chunk_size):
        chunk_specs = candidate_specs[start : start + chunk_size]
        rule_parameters = stack_candidate_rule_parameters(
            center_parameters,
            directions,
            chunk_specs,
            sigma=perturbation_sigma,
            device=device,
        )
        primary = train_vectorized_routing_population(
            config=config,
            base_state=worker_base_state,
            center_rule=center_rule,
            rule_parameters=rule_parameters,
            inner_batches=worker_inner_batches,
            fitness_batches=worker_fitness_batches,
            correct_batches=worker_correct_batches,
            heldout_fitness_batches=worker_heldout_fitness_batches,
            heldout_correct_batches=worker_heldout_correct_batches,
            device=device,
        )
        ranking_groups = [
            [
                candidate_fitness(
                    config.fitness_objective,
                    initial_clean_metrics,
                    trajectory.clean,
                    checkpoint_clean=trajectory.checkpoint_clean,
                )
            ]
            for trajectory in primary.trajectories
        ]
        for (
            ranking_base_state,
            ranking_inner_batches,
            ranking_initial_clean,
        ) in worker_additional_ranking_inputs:
            ranking = train_vectorized_routing_population(
                config=config,
                base_state=ranking_base_state,
                center_rule=center_rule,
                rule_parameters=rule_parameters,
                inner_batches=ranking_inner_batches,
                fitness_batches=worker_fitness_batches,
                correct_batches=worker_correct_batches,
                heldout_fitness_batches=None,
                heldout_correct_batches=None,
                device=device,
            )
            for local_index, trajectory in enumerate(
                ranking.trajectories
            ):
                ranking_groups[local_index].append(
                    candidate_fitness(
                        config.fitness_objective,
                        ranking_initial_clean,
                        trajectory.clean,
                        checkpoint_clean=trajectory.checkpoint_clean,
                    )
                )
        for local_index, (candidate_index, _, _) in enumerate(chunk_specs):
            ranking_fitnesses = tuple(ranking_groups[local_index])
            results.append(
                (
                    candidate_index,
                    sum(ranking_fitnesses) / len(ranking_fitnesses),
                    primary.trajectories[local_index],
                    [],
                    ranking_fitnesses,
                )
            )
    return results
