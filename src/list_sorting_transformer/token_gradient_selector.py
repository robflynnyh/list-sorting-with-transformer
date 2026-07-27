"""Bidirectional token policy for training-time attention-score reversal."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .shortcut_credit import ShortcutBatch, ShortcutPointerVocabulary
from .token_gradient_reversal import oracle_shortcut_selection


@dataclass(frozen=True)
class TokenGradientSelectorConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    initial_reverse_probability: float = 0.05

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_heads,
        ) < 1:
            raise ValueError("selector dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.initial_reverse_probability < 1:
            raise ValueError(
                "initial_reverse_probability must be in (0, 1)"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sinusoidal_positions(
    length: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create non-learned absolute positions for arbitrary sequence lengths."""

    if length < 1:
        raise ValueError("length must be positive")
    positions = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    )[:, None]
    frequencies = torch.exp(
        torch.arange(
            0,
            d_model,
            2,
            device=device,
            dtype=torch.float32,
        )
        * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros(
        length,
        d_model,
        device=device,
        dtype=torch.float32,
    )
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(
        positions * frequencies[: encoding[:, 1::2].shape[1]]
    )
    return encoding.to(dtype=dtype)


class TokenGradientSelector(nn.Module):
    """Predict keep/reverse actions with bidirectional self-attention."""

    def __init__(self, config: TokenGradientSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=int(
                config.d_model * config.ffn_multiplier
            ),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, 2)
        nn.init.xavier_uniform_(self.action_head.weight, gain=0.1)
        with torch.no_grad():
            self.action_head.bias[0] = 0.0
            self.action_head.bias[1] = math.log(
                config.initial_reverse_probability
                / (1.0 - config.initial_reverse_probability)
            )

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        hidden = self.token_embedding(token_ids)
        hidden = hidden + sinusoidal_positions(
            token_ids.shape[1],
            self.config.d_model,
            device=token_ids.device,
            dtype=hidden.dtype,
        )
        hidden = self.encoder(hidden)
        return self.action_head(self.final_norm(hidden))


@dataclass(frozen=True)
class SelectorTrajectory:
    actions: tuple[Tensor, ...]
    selected_fraction: float
    oracle_selected_fraction: float
    other_selected_fraction: float


@torch.no_grad()
def sample_selector_trajectory(
    selector: TokenGradientSelector,
    batches: tuple[ShortcutBatch, ...],
    *,
    vocabulary: ShortcutPointerVocabulary,
    generator: torch.Generator,
) -> SelectorTrajectory:
    """Sample one binary action for every source token in a trajectory."""

    actions = []
    selected_total = 0
    token_total = 0
    oracle_selected = 0
    oracle_total = 0
    other_selected = 0
    other_total = 0
    was_training = selector.training
    selector.eval()
    for batch in batches:
        logits = selector(batch.input_ids)
        probabilities = logits.softmax(dim=-1)[..., 1]
        selected = torch.rand(
            probabilities.shape,
            generator=generator,
            device=probabilities.device,
        ).lt(probabilities)
        oracle = oracle_shortcut_selection(
            batch.input_ids,
            vocabulary,
        )
        actions.append(selected.cpu())
        selected_total += int(selected.sum())
        token_total += selected.numel()
        oracle_selected += int(selected[oracle].sum())
        oracle_total += int(oracle.sum())
        other_selected += int(selected[~oracle].sum())
        other_total += int((~oracle).sum())
    selector.train(was_training)
    return SelectorTrajectory(
        actions=tuple(actions),
        selected_fraction=selected_total / token_total,
        oracle_selected_fraction=oracle_selected / oracle_total,
        other_selected_fraction=other_selected / other_total,
    )


@torch.no_grad()
def sample_selector_trajectories(
    selector: TokenGradientSelector,
    batches: tuple[ShortcutBatch, ...],
    *,
    group_size: int,
    vocabulary: ShortcutPointerVocabulary,
    generators: tuple[torch.Generator, ...],
) -> tuple[SelectorTrajectory, ...]:
    """Sample a policy group while sharing each selector forward pass."""

    if group_size < 1 or len(generators) != group_size:
        raise ValueError("one generator is required per group member")
    actions: list[list[Tensor]] = [[] for _ in range(group_size)]
    selected_totals = [0] * group_size
    oracle_selected_totals = [0] * group_size
    other_selected_totals = [0] * group_size
    token_total = 0
    oracle_total = 0
    other_total = 0
    was_training = selector.training
    selector.eval()
    for batch in batches:
        probabilities = selector(
            batch.input_ids
        ).softmax(dim=-1)[..., 1]
        oracle = oracle_shortcut_selection(
            batch.input_ids,
            vocabulary,
        )
        token_total += probabilities.numel()
        oracle_total += int(oracle.sum())
        other_total += int((~oracle).sum())
        for member_index, generator in enumerate(generators):
            selected = torch.rand(
                probabilities.shape,
                generator=generator,
                device=probabilities.device,
            ).lt(probabilities)
            actions[member_index].append(selected.cpu())
            selected_totals[member_index] += int(selected.sum())
            oracle_selected_totals[member_index] += int(
                selected[oracle].sum()
            )
            other_selected_totals[member_index] += int(
                selected[~oracle].sum()
            )
    selector.train(was_training)
    return tuple(
        SelectorTrajectory(
            actions=tuple(actions[member_index]),
            selected_fraction=selected_totals[member_index] / token_total,
            oracle_selected_fraction=(
                oracle_selected_totals[member_index] / oracle_total
            ),
            other_selected_fraction=(
                other_selected_totals[member_index] / other_total
            ),
        )
        for member_index in range(group_size)
    )


def trajectory_policy_terms(
    selector: TokenGradientSelector,
    batches: tuple[ShortcutBatch, ...],
    actions: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    """Return mean action log probability and entropy for one trajectory."""

    if len(batches) != len(actions):
        raise ValueError("one action tensor is required per batch")
    log_probability_sum = torch.zeros(
        (),
        device=next(selector.parameters()).device,
    )
    entropy_sum = torch.zeros_like(log_probability_sum)
    token_count = 0
    for batch, selected in zip(batches, actions):
        if selected.shape != batch.input_ids.shape:
            raise ValueError("actions must match their batch token shapes")
        logits = selector(batch.input_ids)
        log_probabilities = logits.log_softmax(dim=-1)
        selected = selected.to(
            device=logits.device,
            dtype=torch.long,
        )
        log_probability_sum = log_probability_sum + log_probabilities.gather(
            -1,
            selected[..., None],
        ).sum()
        probabilities = log_probabilities.exp()
        entropy_sum = entropy_sum - (
            probabilities * log_probabilities
        ).sum()
        token_count += selected.numel()
    return (
        log_probability_sum / token_count,
        entropy_sum / token_count,
    )


def grouped_trajectory_policy_terms(
    selector: TokenGradientSelector,
    batches: tuple[ShortcutBatch, ...],
    trajectories: tuple[SelectorTrajectory, ...],
) -> tuple[Tensor, Tensor]:
    """Return every member log probability with shared policy forwards."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    group_size = len(trajectories)
    log_probability_sums = torch.zeros(
        group_size,
        device=next(selector.parameters()).device,
    )
    entropy_sum = torch.zeros(
        (),
        device=log_probability_sums.device,
    )
    token_count = 0
    for batch_index, batch in enumerate(batches):
        logits = selector(batch.input_ids)
        log_probabilities = logits.log_softmax(dim=-1)
        selected = torch.stack(
            [
                trajectory.actions[batch_index]
                for trajectory in trajectories
            ]
        ).to(device=logits.device, dtype=torch.long)
        if selected.shape[1:] != batch.input_ids.shape:
            raise ValueError("actions must match their batch token shapes")
        expanded = log_probabilities.unsqueeze(0).expand(
            group_size,
            -1,
            -1,
            -1,
        )
        log_probability_sums = log_probability_sums + expanded.gather(
            -1,
            selected[..., None],
        ).squeeze(-1).sum(dim=(1, 2))
        probabilities = log_probabilities.exp()
        entropy_sum = entropy_sum - (
            probabilities * log_probabilities
        ).sum()
        token_count += batch.input_ids.numel()
    return (
        log_probability_sums / token_count,
        entropy_sum / token_count,
    )


@torch.no_grad()
def selector_probability_statistics(
    selector: TokenGradientSelector,
    batches: tuple[ShortcutBatch, ...],
    *,
    vocabulary: ShortcutPointerVocabulary,
) -> dict[str, float]:
    """Measure learned reverse probability at oracle and other positions."""

    was_training = selector.training
    selector.eval()
    oracle_probabilities = []
    other_probabilities = []
    for batch in batches:
        probabilities = selector(batch.input_ids).softmax(dim=-1)[..., 1]
        oracle = oracle_shortcut_selection(
            batch.input_ids,
            vocabulary,
        )
        oracle_probabilities.append(probabilities[oracle])
        other_probabilities.append(probabilities[~oracle])
    selector.train(was_training)
    return {
        "oracle_reverse_probability": float(
            torch.cat(oracle_probabilities).mean()
        ),
        "other_reverse_probability": float(
            torch.cat(other_probabilities).mean()
        ),
    }


def standardize_group_rewards(
    rewards: Tensor,
    *,
    minimum_standard_deviation: float = 0.0,
) -> Tensor:
    """Compute group-relative advantages, or zero when all rewards tie."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must contain at least two group members")
    if minimum_standard_deviation < 0:
        raise ValueError(
            "minimum_standard_deviation must be nonnegative"
        )
    standard_deviation = rewards.std(unbiased=False)
    if float(standard_deviation) < max(
        1e-8,
        minimum_standard_deviation,
    ):
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / standard_deviation
