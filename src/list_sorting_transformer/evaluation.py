"""Evaluation utilities shared by training and the standalone evaluator."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import IGNORE_INDEX, make_sorting_batch
from .metrics import generated_sorting_metrics, masked_token_accuracy
from .model import DecoderTransformer, ModelConfig
from .tokens import SymbolVocabulary


def autocast_context(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def output_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@torch.inference_mode()
def evaluate_lengths(
    model: DecoderTransformer,
    vocabulary: SymbolVocabulary,
    lengths: Iterable[int],
    *,
    examples_per_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    use_autocast: bool = True,
) -> dict[int, dict[str, float]]:
    if examples_per_length < 1 or batch_size < 1:
        raise ValueError("evaluation sizes must be positive")
    was_training = model.training
    model.eval()
    results = {}
    for length in lengths:
        generator = torch.Generator().manual_seed(seed + 104_729 * int(length))
        totals: dict[str, float] = {}
        processed = 0
        while processed < examples_per_length:
            current_batch_size = min(batch_size, examples_per_length - processed)
            batch = make_sorting_batch(
                current_batch_size,
                int(length),
                generator=generator,
                symbol_count=vocabulary.symbol_count,
                device=device,
            )
            with autocast_context(device, enabled=use_autocast):
                logits = model(batch.model_inputs)
                loss = output_cross_entropy(logits, batch.labels)
                generated = model.generate(
                    batch.prompt_ids,
                    max_new_tokens=2 * int(length) + 2,
                )
            metrics = generated_sorting_metrics(
                batch.values.cpu(),
                generated.cpu(),
                vocabulary,
            )
            metrics["teacher_forced_loss"] = float(loss.item())
            metrics["teacher_forced_token_accuracy"] = masked_token_accuracy(
                logits,
                batch.labels,
            )
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * current_batch_size
            processed += current_batch_size
        results[int(length)] = {
            name: value / examples_per_length
            for name, value in totals.items()
        }
    model.train(was_training)
    return results


def aggregate_length_ranges(
    per_length: dict[int, dict[str, float]],
    *,
    train_min_length: int,
    train_max_length: int,
) -> dict[str, dict[str, float]]:
    groups = {
        "in_domain": [
            metrics
            for length, metrics in per_length.items()
            if train_min_length <= length <= train_max_length
        ],
        "out_of_domain": [
            metrics
            for length, metrics in per_length.items()
            if length > train_max_length
        ],
    }
    output = {}
    for name, rows in groups.items():
        if not rows:
            continue
        output[name] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in rows[0]
        }
    return output


def load_model_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device,
) -> tuple[DecoderTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_config or model_state")
    model = DecoderTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint
