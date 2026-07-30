"""Differentiable Adam lookahead for attention-credit routers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call

from .shortcut_credit import (
    AttentionRoutingRule,
    ShortcutBatch,
    ShortcutDecoderTransformer,
)


@dataclass(frozen=True)
class RouterLookaheadObjective:
    short_loss: Tensor
    lookahead_mean_loss: Tensor
    meta_loss: Tensor
    meta_accuracy: Tensor


def clip_virtual_gradients(
    gradients: tuple[Tensor, ...],
    *,
    max_norm: float,
) -> tuple[Tensor, ...]:
    total_norm = torch.sqrt(
        sum(gradient.float().square().sum() for gradient in gradients)
    )
    coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    return tuple(
        gradient * coefficient.to(dtype=gradient.dtype)
        for gradient in gradients
    )


def router_lookahead_objective(
    model: ShortcutDecoderTransformer,
    router: AttentionRoutingRule,
    training_batches: tuple[ShortcutBatch, ...],
    fitness_batches: tuple[ShortcutBatch, ...],
    *,
    ordinary_optimizer: torch.optim.Adam,
    gradient_clip: float,
) -> RouterLookaheadObjective:
    """Evaluate fitness after differentiable routed Adam updates."""

    if not training_batches:
        raise ValueError("lookahead requires at least one training batch")
    if not fitness_batches:
        raise ValueError("lookahead requires at least one fitness batch")
    if len(ordinary_optimizer.param_groups) != 1:
        raise ValueError("lookahead requires one Adam parameter group")
    group = ordinary_optimizer.param_groups[0]
    if group["weight_decay"] != 0 or group["amsgrad"] or group["maximize"]:
        raise ValueError("unsupported Adam configuration for lookahead")

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    adapted_parameters = parameters
    exp_avg = {}
    exp_avg_sq = {}
    optimizer_steps = set()
    for name, parameter in parameters.items():
        state = ordinary_optimizer.state.get(parameter, {})
        exp_avg[name] = state.get(
            "exp_avg",
            torch.zeros_like(parameter),
        ).detach()
        exp_avg_sq[name] = state.get(
            "exp_avg_sq",
            torch.zeros_like(parameter),
        ).detach()
        optimizer_steps.add(
            int(state.get("step", torch.tensor(0)).item())
        )
    if len(optimizer_steps) != 1:
        raise ValueError("Adam parameters have inconsistent step counts")
    initial_step = optimizer_steps.pop()
    beta1, beta2 = group["betas"]
    learning_rate = float(group["lr"])
    epsilon = float(group["eps"])

    short_losses = []
    for lookahead_index, training_batch in enumerate(
        training_batches,
        start=1,
    ):
        logits = functional_call(
            model,
            (adapted_parameters, buffers),
            (training_batch.input_ids,),
            {"backward_rule": router},
        )[:, -1]
        short_loss = F.cross_entropy(logits, training_batch.targets)
        short_losses.append(short_loss)
        gradients = torch.autograd.grad(
            short_loss,
            tuple(adapted_parameters.values()),
            create_graph=True,
        )
        gradients = clip_virtual_gradients(
            gradients,
            max_norm=gradient_clip,
        )
        optimizer_step = initial_step + lookahead_index
        bias_correction1 = 1 - beta1**optimizer_step
        bias_correction2 = 1 - beta2**optimizer_step
        next_parameters = {}
        next_exp_avg = {}
        next_exp_avg_sq = {}
        for (name, parameter), gradient in zip(
            adapted_parameters.items(),
            gradients,
        ):
            mean = beta1 * exp_avg[name] + (1 - beta1) * gradient
            square_mean = (
                beta2 * exp_avg_sq[name]
                + (1 - beta2) * gradient.square()
            )
            denominator = (
                square_mean.sqrt() / bias_correction2**0.5
            ) + epsilon
            next_parameters[name] = parameter - (
                learning_rate / bias_correction1
            ) * mean / denominator
            next_exp_avg[name] = mean
            next_exp_avg_sq[name] = square_mean
        adapted_parameters = next_parameters
        exp_avg = next_exp_avg
        exp_avg_sq = next_exp_avg_sq

    losses = []
    correct = []
    for fitness_batch in fitness_batches:
        logits = functional_call(
            model,
            (adapted_parameters, buffers),
            (fitness_batch.input_ids,),
        )[:, -1]
        losses.append(F.cross_entropy(logits, fitness_batch.targets))
        correct.append(logits.argmax(dim=-1).eq(fitness_batch.targets))
    return RouterLookaheadObjective(
        short_loss=short_losses[0],
        lookahead_mean_loss=torch.stack(short_losses).mean(),
        meta_loss=torch.stack(losses).mean(),
        meta_accuracy=torch.cat(correct).float().mean(),
    )
