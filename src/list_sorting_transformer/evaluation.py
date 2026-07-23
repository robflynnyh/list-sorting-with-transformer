"""Evaluation utilities shared by training and the standalone evaluator."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import (
    IGNORE_INDEX,
    PointerQuicksortBatch,
    make_pointer_quicksort_batch,
    make_quicksort_trace_batch,
    make_sorting_batch,
)
from .metrics import (
    generated_pointer_no_tool_metrics,
    generated_pointer_quicksort_metrics,
    generated_quicksort_metrics,
    generated_sorting_metrics,
    masked_token_accuracy,
)
from .model import DecoderTransformer, ModelConfig
from .pointer_quicksort import (
    PointerQuicksortMachine,
    PointerQuicksortRollout,
)
from .quicksort import SnapshotMode
from .recurrent import LSTMConfig, LSTMSorter
from .tokens import (
    PAD,
    PointerQuicksortVocabulary,
    QuicksortTraceVocabulary,
    SymbolVocabulary,
)


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
def generate_pointer_quicksort_rollouts(
    model: DecoderTransformer | LSTMSorter,
    batch: PointerQuicksortBatch,
    vocabulary: PointerQuicksortVocabulary,
    *,
    max_actions: int | None = None,
) -> tuple[PointerQuicksortRollout, ...]:
    """Alternate model actions with observations from batched executors."""

    if max_actions is None:
        max_actions = max(len(trace.action_tokens) for trace in batch.traces)
    if max_actions < 1:
        raise ValueError("max_actions must be positive")

    machines = [
        PointerQuicksortMachine(row, vocabulary)
        for row in batch.values.tolist()
    ]
    generated_actions: list[list[int]] = [[] for _ in machines]
    action_candidates = torch.tensor(
        vocabulary.action_tokens,
        dtype=torch.long,
        device=batch.prompt_ids.device,
    )

    if isinstance(model, DecoderTransformer):
        next_logits, decode_state = model.forward_with_cache(batch.prompt_ids)
        recurrent = False
    else:
        next_logits, decode_state = model.forward_with_state(batch.prompt_ids)
        recurrent = True

    for _ in range(max_actions):
        candidate_logits = next_logits[:, -1].index_select(
            dim=-1,
            index=action_candidates,
        )
        selected = action_candidates[candidate_logits.argmax(dim=-1)]
        selected_actions = selected.tolist()
        action_values = [PAD] * len(machines)
        observation_values = [PAD] * len(machines)

        for row_index, machine in enumerate(machines):
            if machine.finished:
                continue
            action = int(selected_actions[row_index])
            generated_actions[row_index].append(action)
            action_values[row_index] = action
            observation = machine.step(action)
            if observation is not None:
                observation_values[row_index] = observation

        if all(machine.finished for machine in machines):
            break
        action_column = torch.tensor(
            action_values,
            dtype=torch.long,
            device=selected.device,
        )
        observation_column = torch.tensor(
            observation_values,
            dtype=torch.long,
            device=selected.device,
        )
        if recurrent:
            _, decode_state = model.forward_with_state(
                action_column[:, None],
                decode_state,
            )
            next_logits, decode_state = model.forward_with_state(
                observation_column[:, None],
                decode_state,
            )
        else:
            _, decode_state = model.forward_with_cache(
                action_column[:, None],
                caches=decode_state,
            )
            next_logits, decode_state = model.forward_with_cache(
                observation_column[:, None],
                caches=decode_state,
            )

    return tuple(
        PointerQuicksortRollout(
            action_tokens=tuple(actions),
            final_values=tuple(machine.array),
            completed=machine.completed,
            valid_execution=machine.valid,
            timed_out=not machine.finished,
        )
        for actions, machine in zip(generated_actions, machines)
    )


@torch.inference_mode()
def evaluate_lengths(
    model: DecoderTransformer | LSTMSorter,
    vocabulary: SymbolVocabulary,
    lengths: Iterable[int],
    *,
    examples_per_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    use_autocast: bool = True,
    task: str = "direct",
    trace_snapshot_mode: SnapshotMode = "partition",
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
            if task == "direct":
                batch = make_sorting_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    symbol_count=vocabulary.symbol_count,
                    device=device,
                )
                max_new_tokens = 2 * int(length) + 2
            elif task == "quicksort_trace":
                if not isinstance(vocabulary, QuicksortTraceVocabulary):
                    raise TypeError(
                        "quicksort_trace requires QuicksortTraceVocabulary"
                    )
                batch = make_quicksort_trace_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    snapshot_mode=trace_snapshot_mode,
                    device=device,
                )
                max_new_tokens = max(
                    len(trace.target_tokens) for trace in batch.traces
                ) + 8
            elif task in {"pointer_quicksort", "pointer_quicksort_no_tool"}:
                if not isinstance(vocabulary, PointerQuicksortVocabulary):
                    raise TypeError(
                        f"{task} requires PointerQuicksortVocabulary"
                    )
                batch = make_pointer_quicksort_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        task == "pointer_quicksort_no_tool"
                    ),
                    device=device,
                )
                if task == "pointer_quicksort":
                    max_new_tokens = 0
                else:
                    max_new_tokens = max(
                        len(trace.target_tokens) for trace in batch.traces
                    ) + 8
            else:
                raise ValueError(f"unsupported sorting task: {task}")
            with autocast_context(device, enabled=use_autocast):
                logits = model(batch.model_inputs)
                loss = output_cross_entropy(logits, batch.labels)
                if task == "pointer_quicksort":
                    assert isinstance(batch, PointerQuicksortBatch)
                    assert isinstance(vocabulary, PointerQuicksortVocabulary)
                    rollouts = generate_pointer_quicksort_rollouts(
                        model,
                        batch,
                        vocabulary,
                    )
                elif task == "pointer_quicksort_no_tool":
                    assert isinstance(vocabulary, PointerQuicksortVocabulary)
                    generated = model.generate(
                        batch.prompt_ids,
                        max_new_tokens=max_new_tokens,
                        stop_token=vocabulary.action_token("DONE"),
                    )
                else:
                    generated = model.generate(
                        batch.prompt_ids,
                        max_new_tokens=max_new_tokens,
                    )
            if task == "direct":
                metrics = generated_sorting_metrics(
                    batch.values.cpu(),
                    generated.cpu(),
                    vocabulary,
                )
            elif task == "quicksort_trace":
                assert isinstance(vocabulary, QuicksortTraceVocabulary)
                metrics = generated_quicksort_metrics(
                    batch.values.cpu(),
                    generated.cpu(),
                    vocabulary,
                    batch.traces,
                )
            elif task == "pointer_quicksort":
                assert isinstance(batch, PointerQuicksortBatch)
                assert isinstance(vocabulary, PointerQuicksortVocabulary)
                metrics = generated_pointer_quicksort_metrics(
                    batch.values.cpu(),
                    rollouts,
                    vocabulary,
                    batch.traces,
                )
            else:
                assert isinstance(batch, PointerQuicksortBatch)
                assert isinstance(vocabulary, PointerQuicksortVocabulary)
                metrics = generated_pointer_no_tool_metrics(
                    batch.values.cpu(),
                    generated.cpu(),
                    vocabulary,
                    batch.traces,
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
) -> tuple[DecoderTransformer | LSTMSorter, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint is missing model_config or model_state")
    architecture = checkpoint.get("architecture", "transformer")
    if architecture == "transformer":
        model = DecoderTransformer(ModelConfig(**checkpoint["model_config"]))
    elif architecture == "lstm":
        model = LSTMSorter(LSTMConfig(**checkpoint["model_config"]))
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint
