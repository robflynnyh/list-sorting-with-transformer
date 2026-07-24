"""Absolute input position embeddings for pointer tasks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed absolute position embeddings with Transformer sinusoidal features."""

    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, positions: Tensor) -> Tensor:
        angles = (
            positions.to(dtype=self.inverse_frequency.dtype).unsqueeze(-1)
            * self.inverse_frequency
        )
        embedding = torch.empty(
            *positions.shape,
            self.inverse_frequency.shape[0] * 2,
            device=positions.device,
            dtype=torch.float32,
        )
        embedding[..., 0::2] = angles.sin().to(dtype=torch.float32)
        embedding[..., 1::2] = angles.cos().to(dtype=torch.float32)
        return embedding


class ModularPositionEmbedding(nn.Module):
    """Composable absolute positions represented by categorical residue keys."""

    def __init__(self, dim: int, moduli: tuple[int, ...]) -> None:
        super().__init__()
        if not moduli or any(modulus < 2 for modulus in moduli):
            raise ValueError("position moduli must all be at least two")
        if len(set(moduli)) != len(moduli):
            raise ValueError("position moduli must be distinct")
        if any(
            math.gcd(left, right) != 1
            for index, left in enumerate(moduli)
            for right in moduli[index + 1 :]
        ):
            raise ValueError("position moduli must be pairwise coprime")
        if dim % len(moduli):
            raise ValueError("embedding dimension must be divisible by moduli count")
        self.dim = dim
        self.moduli = moduli
        self.component_dim = dim // len(moduli)
        self.codebooks = nn.ModuleList(
            nn.Embedding(modulus, self.component_dim) for modulus in moduli
        )
        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=0.02)

    @property
    def period(self) -> int:
        return math.prod(self.moduli)

    def residues(self, positions: Tensor) -> tuple[Tensor, ...]:
        return tuple(positions.remainder(modulus) for modulus in self.moduli)

    def forward(self, positions: Tensor) -> Tensor:
        return torch.cat(
            [
                codebook(residue)
                for codebook, residue in zip(
                    self.codebooks,
                    self.residues(positions),
                )
            ],
            dim=-1,
        )

    def component_logits(self, query: Tensor) -> tuple[Tensor, ...]:
        if query.shape[-1] != self.dim:
            raise ValueError("query dimension does not match position embedding")
        components = query.split(self.component_dim, dim=-1)
        scale = math.sqrt(self.component_dim)
        return tuple(
            component @ codebook.weight.T / scale
            for component, codebook in zip(components, self.codebooks)
        )


def sample_position_offsets(
    batch_size: int,
    *,
    minimum: int,
    maximum: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if minimum > maximum:
        raise ValueError("minimum offset must be <= maximum offset")
    offsets = torch.randint(
        minimum,
        maximum + 1,
        (batch_size,),
        generator=generator,
    )
    return offsets.to(device)


def input_position_embeddings(
    position_embedding: SinusoidalPositionEmbedding,
    sequence_length: int,
    *,
    device: torch.device,
    offsets: Tensor | None = None,
) -> Tensor:
    token_offsets = torch.arange(sequence_length, device=device)
    positions = (
        token_offsets
        if offsets is None
        else offsets[:, None] + token_offsets[None, :]
    )
    return position_embedding(positions)
