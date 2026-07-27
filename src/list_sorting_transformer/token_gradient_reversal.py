"""Training-only token-source gradient reversal for the shortcut task."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .shortcut_credit import (
    ShortcutBatch,
    ShortcutDecoderTransformer,
    ShortcutPointerVocabulary,
)


def oracle_shortcut_selection(
    token_ids: Tensor,
    vocabulary: ShortcutPointerVocabulary,
) -> Tensor:
    """Select the leaked answer token immediately following ``<LEAK>``."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, time]")
    leak_matches = token_ids.eq(vocabulary.leak_token)
    if not bool(leak_matches.sum(dim=1).eq(1).all()):
        raise ValueError("every prompt must contain exactly one leak token")
    leak_positions = leak_matches.to(dtype=torch.long).argmax(dim=1)
    shortcut_positions = leak_positions + 1
    if bool(shortcut_positions.ge(token_ids.shape[1]).any()):
        raise ValueError("every leak token must be followed by a shortcut token")
    selection = torch.zeros_like(token_ids, dtype=torch.bool)
    selection.scatter_(1, shortcut_positions[:, None], True)
    return selection


def source_gradient_multipliers(
    selection: Tensor,
    *,
    reversal_scale: float = 1.0,
) -> Tensor:
    """Map selected sources to ``-scale`` and all others to ``+1``."""

    if selection.ndim != 2 or selection.dtype != torch.bool:
        raise ValueError("selection must be a boolean [batch, time] tensor")
    if not reversal_scale > 0:
        raise ValueError("reversal_scale must be positive")
    return torch.where(
        selection,
        torch.full_like(selection, -reversal_scale, dtype=torch.float32),
        torch.ones_like(selection, dtype=torch.float32),
    )


def forward_with_source_gradient_reversal(
    model: ShortcutDecoderTransformer,
    token_ids: Tensor,
    selection: Tensor,
    *,
    reversal_scale: float = 1.0,
) -> Tensor:
    """Run the model with one shared source-reversal mask in every layer."""

    if selection.shape != token_ids.shape:
        raise ValueError("selection must match token_ids")
    multipliers = source_gradient_multipliers(
        selection,
        reversal_scale=reversal_scale,
    ).to(device=token_ids.device)
    hidden = model.embed(token_ids)
    for block in model.blocks:
        hidden = block(
            hidden,
            backward_source_multipliers=multipliers,
        )
    hidden = model.final_norm(hidden)
    return F.linear(hidden, model.token_embedding.weight)


def oracle_reversal_shortcut_loss(
    model: ShortcutDecoderTransformer,
    batch: ShortcutBatch,
    vocabulary: ShortcutPointerVocabulary,
    *,
    reversal_scale: float = 1.0,
) -> Tensor:
    """Cross entropy with oracle reversal of the leaked answer source."""

    selection = oracle_shortcut_selection(batch.input_ids, vocabulary)
    logits = forward_with_source_gradient_reversal(
        model,
        batch.input_ids,
        selection,
        reversal_scale=reversal_scale,
    )
    return F.cross_entropy(logits[:, -1], batch.targets)
