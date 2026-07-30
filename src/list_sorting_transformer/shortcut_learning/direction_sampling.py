"""Proposal samplers for low-rank EGGROLL directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .shortcut_credit import (
    AttentionRoutingRule,
    EggrollDirection,
    apply_eggroll_direction,
    clone_center_parameters,
    sample_eggroll_direction,
)


@dataclass(frozen=True)
class DirectionSamplingResult:
    directions: tuple[EggrollDirection, ...]
    metrics: dict[str, float]


def scale_direction(
    direction: EggrollDirection,
    scale: float,
) -> EggrollDirection:
    return EggrollDirection(
        {
            name: tensor * scale
            for name, tensor in direction.tensors.items()
        }
    )


def mean_absolute_cosine(signatures: Tensor) -> float:
    if signatures.shape[0] < 2:
        return 0.0
    normalized = torch.nn.functional.normalize(signatures.float(), dim=1)
    similarities = (normalized @ normalized.T).abs()
    mask = ~torch.eye(
        similarities.shape[0],
        dtype=torch.bool,
        device=similarities.device,
    )
    return float(similarities[mask].mean())


def select_diverse_signatures(
    signatures: Tensor,
    count: int,
) -> tuple[int, ...]:
    """Greedily minimize maximum absolute cosine to selected directions."""

    if signatures.ndim != 2:
        raise ValueError("signatures must be a matrix")
    if not 1 <= count <= signatures.shape[0]:
        raise ValueError("selection count is out of range")
    normalized = torch.nn.functional.normalize(signatures.float(), dim=1)
    selected = [int(normalized.square().sum(dim=1).argmax())]
    available = torch.ones(
        signatures.shape[0],
        dtype=torch.bool,
        device=signatures.device,
    )
    available[selected[0]] = False
    maximum_similarity = (normalized @ normalized[selected[0]]).abs()
    while len(selected) < count:
        scores = maximum_similarity.masked_fill(~available, float("inf"))
        next_index = int(scores.argmin())
        selected.append(next_index)
        available[next_index] = False
        similarity = (normalized @ normalized[next_index]).abs()
        maximum_similarity = torch.maximum(maximum_similarity, similarity)
    return tuple(selected)


def _causal_gate_vector(
    router: AttentionRoutingRule,
    token_ids: Tensor,
) -> Tensor:
    gates = torch.stack(router.attention_gates(token_ids), dim=1)
    sequence_length = gates.shape[-1]
    causal = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=gates.device,
    ).tril()
    return gates[..., causal].reshape(-1)


@torch.no_grad()
def _direction_signature(
    router: AttentionRoutingRule,
    center_parameters: dict[str, Tensor],
    direction: EggrollDirection,
    *,
    sigma: float,
    token_ids: Tensor,
    signature_indices: Tensor,
) -> tuple[Tensor, float]:
    apply_eggroll_direction(
        router,
        center_parameters,
        direction,
        sigma=sigma,
        sign=1,
    )
    positive = _causal_gate_vector(router, token_ids)
    apply_eggroll_direction(
        router,
        center_parameters,
        direction,
        sigma=sigma,
        sign=-1,
    )
    negative = _causal_gate_vector(router, token_ids)
    central_change = (positive - negative) * 0.5
    signature = central_change[signature_indices].float()
    rms = float(central_change.float().square().mean().sqrt())
    return signature, rms


@torch.no_grad()
def sample_function_diverse_directions(
    router: AttentionRoutingRule,
    *,
    generator: torch.Generator,
    count: int,
    candidate_multiplier: int,
    sigma: float,
    probe_token_ids: Tensor,
    signature_size: int,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> DirectionSamplingResult:
    """Preselect diverse mutations using cheap router-output changes."""

    if count < 1 or candidate_multiplier < 1 or signature_size < 1:
        raise ValueError("sampling counts must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 < minimum_scale <= maximum_scale:
        raise ValueError("invalid direction scale range")

    pool_size = count * candidate_multiplier
    directions = tuple(
        sample_eggroll_direction(router, generator=generator)
        for _ in range(pool_size)
    )
    center_parameters = clone_center_parameters(router)
    capture_statistics = router.capture_statistics
    router.capture_statistics = False
    try:
        center_vector = _causal_gate_vector(router, probe_token_ids)
        sampled_size = min(signature_size, center_vector.numel())
        signature_indices = torch.linspace(
            0,
            center_vector.numel() - 1,
            steps=sampled_size,
            device=center_vector.device,
        ).long()
        raw = tuple(
            _direction_signature(
                router,
                center_parameters,
                direction,
                sigma=sigma,
                token_ids=probe_token_ids,
                signature_indices=signature_indices,
            )
            for direction in directions
        )
        positive_rms = torch.tensor(
            [rms for _, rms in raw if rms > 1e-12],
            dtype=torch.float32,
        )
        target_rms = (
            float(positive_rms.median())
            if positive_rms.numel()
            else 1.0
        )
        normalized_directions = tuple(
            scale_direction(
                direction,
                max(
                    minimum_scale,
                    min(
                        maximum_scale,
                        target_rms / max(rms, 1e-12),
                    ),
                ),
            )
            for direction, (_, rms) in zip(directions, raw)
        )
        normalized = tuple(
            _direction_signature(
                router,
                center_parameters,
                direction,
                sigma=sigma,
                token_ids=probe_token_ids,
                signature_indices=signature_indices,
            )
            for direction in normalized_directions
        )
        signatures = torch.stack(
            tuple(signature for signature, _ in normalized)
        )
        selected_indices = select_diverse_signatures(signatures, count)
        selected_signatures = signatures[list(selected_indices)]
        selected_rms = [normalized[index][1] for index in selected_indices]
        return DirectionSamplingResult(
            directions=tuple(
                normalized_directions[index]
                for index in selected_indices
            ),
            metrics={
                "direction_sampling/pool_size": float(pool_size),
                "direction_sampling/target_function_rms": target_rms,
                "direction_sampling/selected_function_rms_mean": (
                    sum(selected_rms) / len(selected_rms)
                ),
                "direction_sampling/pool_abs_cosine_mean": (
                    mean_absolute_cosine(signatures)
                ),
                "direction_sampling/selected_abs_cosine_mean": (
                    mean_absolute_cosine(selected_signatures)
                ),
            },
        )
    finally:
        for name, parameter in router.named_parameters():
            parameter.copy_(center_parameters[name])
        router.capture_statistics = capture_statistics
