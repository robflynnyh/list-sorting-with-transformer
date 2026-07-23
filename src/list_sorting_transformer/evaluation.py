"""Evaluation utilities shared by training and the standalone evaluator."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .adjacent_sort import (
    AdjacentSortMachine,
    AdjacentSortRollout,
    AutoAdvanceSortMachine,
)
from .data import (
    AdjacentSortBatch,
    IGNORE_INDEX,
    LocalWindowSortBatch,
    PointerNextBatch,
    PointerQuicksortBatch,
    make_adjacent_sort_batch,
    make_auto_advance_sort_batch,
    make_local_window_sort_batch,
    make_pointer_next_batch,
    make_pointer_quicksort_batch,
    make_quicksort_trace_batch,
    make_sorting_batch,
)
from .local_window_sort import (
    WINDOW_TOKEN_LENGTH,
    WINDOW_TOOL_EVENTS,
    LocalWindowSortMachine,
    LocalWindowSortRollout,
)
from .metrics import (
    generated_adjacent_no_tool_metrics,
    generated_adjacent_sort_metrics,
    generated_auto_advance_no_tool_metrics,
    generated_auto_advance_sort_metrics,
    generated_local_window_sort_metrics,
    generated_pointer_next_metrics,
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
    AdjacentSortVocabulary,
    AutoAdvanceSortVocabulary,
    LocalWindowSortVocabulary,
    PointerNextVocabulary,
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


def generate_adjacent_sort_rollouts(
    model: DecoderTransformer | LSTMSorter,
    batch: AdjacentSortBatch,
    vocabulary: AdjacentSortVocabulary,
    *,
    max_actions: int | None = None,
) -> tuple[AdjacentSortRollout, ...]:
    return _generate_adjacent_machine_rollouts(
        model,
        batch,
        vocabulary,
        AdjacentSortMachine,
        max_actions=max_actions,
    )


def generate_auto_advance_sort_rollouts(
    model: DecoderTransformer | LSTMSorter,
    batch: AdjacentSortBatch,
    vocabulary: AutoAdvanceSortVocabulary,
    *,
    max_actions: int | None = None,
) -> tuple[AdjacentSortRollout, ...]:
    return _generate_adjacent_machine_rollouts(
        model,
        batch,
        vocabulary,
        AutoAdvanceSortMachine,
        max_actions=max_actions,
    )


@torch.inference_mode()
def _generate_adjacent_machine_rollouts(
    model: DecoderTransformer | LSTMSorter,
    batch: AdjacentSortBatch,
    vocabulary: AdjacentSortVocabulary | AutoAdvanceSortVocabulary,
    machine_type: type[AdjacentSortMachine] | type[AutoAdvanceSortMachine],
    *,
    max_actions: int | None,
) -> tuple[AdjacentSortRollout, ...]:
    """Alternate model actions with one- or two-token machine observations."""

    if max_actions is None:
        max_actions = max(len(trace.action_tokens) for trace in batch.traces)
    if max_actions < 1:
        raise ValueError("max_actions must be positive")

    machines = [
        machine_type(row, vocabulary)  # type: ignore[arg-type]
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
        observation_rows: list[tuple[int, ...]] = [() for _ in machines]

        for row_index, machine in enumerate(machines):
            if machine.finished:
                continue
            action = int(selected_actions[row_index])
            generated_actions[row_index].append(action)
            action_values[row_index] = action
            observation_rows[row_index] = machine.step(action)

        if all(machine.finished for machine in machines):
            break

        columns = [action_values]
        max_observations = max(len(row) for row in observation_rows)
        columns.extend(
            [
                [
                    row[observation_index]
                    if observation_index < len(row)
                    else PAD
                    for row in observation_rows
                ]
                for observation_index in range(max_observations)
            ]
        )
        for column in columns:
            token_column = torch.tensor(
                column,
                dtype=torch.long,
                device=selected.device,
            )
            if recurrent:
                next_logits, decode_state = model.forward_with_state(
                    token_column[:, None],
                    decode_state,
                )
            else:
                next_logits, decode_state = model.forward_with_cache(
                    token_column[:, None],
                    caches=decode_state,
                )

    return tuple(
        AdjacentSortRollout(
            action_tokens=tuple(actions),
            final_values=tuple(machine.array),
            completed=machine.completed,
            valid_execution=machine.valid,
            timed_out=not machine.finished,
        )
        for actions, machine in zip(generated_actions, machines)
    )


@torch.inference_mode()
def generate_local_window_sort_rollouts(
    model: DecoderTransformer | LSTMSorter,
    batch: LocalWindowSortBatch,
    vocabulary: LocalWindowSortVocabulary,
    *,
    tool_events: Iterable[str] = WINDOW_TOOL_EVENTS,
    max_actions: int | None = None,
) -> tuple[LocalWindowSortRollout, ...]:
    """Decode actions and fixed-width windows in one growing cached context."""

    tool_event_names = frozenset(
        str(event).upper() for event in tool_events
    )
    if not tool_event_names <= set(WINDOW_TOOL_EVENTS):
        allowed = ", ".join(WINDOW_TOOL_EVENTS)
        raise ValueError(f"tool_events may only contain {allowed}")
    if max_actions is None:
        normal_budget = batch.length * (batch.length - 1) // 2 + 2
        max_actions = normal_budget + 2 * batch.length
    if max_actions < 1:
        raise ValueError("max_actions must be positive")

    machines = [
        LocalWindowSortMachine(row, vocabulary)
        for row in batch.values.tolist()
    ]
    generated_actions: list[list[int]] = [[] for _ in machines]
    generated_windows: list[list[tuple[int, ...]]] = [
        [] for _ in machines
    ]
    expected_windows: list[list[tuple[int, ...]]] = [
        [] for _ in machines
    ]
    required_window_counts = [
        sum(
            bool(transition.response_tokens)
            and transition.response_event not in tool_event_names
            for transition in trace.transitions
        )
        for trace in batch.traces
    ]
    tool_window_counts = [0 for _ in machines]
    action_candidates = torch.tensor(
        vocabulary.action_tokens,
        dtype=torch.long,
        device=batch.values.device,
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
        true_windows: list[tuple[int, ...] | None] = [
            None for _ in machines
        ]
        generate_window = [False for _ in machines]

        for row_index, machine in enumerate(machines):
            if machine.halted:
                continue
            action = int(selected_actions[row_index])
            action_values[row_index] = action
            generated_actions[row_index].append(action)
            response_event, response_tokens = machine.step(action)
            if not response_tokens:
                continue
            true_windows[row_index] = response_tokens
            generate_window[row_index] = (
                machine.valid and response_event not in tool_event_names
            )

        if all(machine.halted for machine in machines):
            break

        action_column = torch.tensor(
            action_values,
            dtype=torch.long,
            device=batch.values.device,
        )
        if recurrent:
            next_logits, decode_state = model.forward_with_state(
                action_column[:, None],
                decode_state,
            )
        else:
            next_logits, decode_state = model.forward_with_cache(
                action_column[:, None],
                caches=decode_state,
            )

        generated_rows: list[list[int]] = [[] for _ in machines]
        for row_index, true_window in enumerate(true_windows):
            if true_window is None:
                continue
            if generate_window[row_index]:
                expected_windows[row_index].append(true_window)
            else:
                tool_window_counts[row_index] += 1

        for window_index in range(WINDOW_TOKEN_LENGTH):
            predicted_window_tokens = next_logits[:, -1].argmax(dim=-1)
            window_values = [PAD] * len(machines)
            for row_index, true_window in enumerate(true_windows):
                if true_window is None:
                    continue
                if generate_window[row_index]:
                    token = int(predicted_window_tokens[row_index])
                    generated_rows[row_index].append(token)
                    window_values[row_index] = token
                else:
                    window_values[row_index] = true_window[window_index]
            window_column = torch.tensor(
                window_values,
                dtype=torch.long,
                device=batch.values.device,
            )
            if recurrent:
                next_logits, decode_state = model.forward_with_state(
                    window_column[:, None],
                    decode_state,
                )
            else:
                next_logits, decode_state = model.forward_with_cache(
                    window_column[:, None],
                    caches=decode_state,
                )

        for row_index, should_generate in enumerate(generate_window):
            if should_generate:
                generated_windows[row_index].append(
                    tuple(generated_rows[row_index])
                )

    return tuple(
        LocalWindowSortRollout(
            action_tokens=tuple(actions),
            final_values=tuple(machine.array),
            completed=machine.completed,
            valid_execution=machine.valid,
            timed_out=not machine.halted,
            generated_window_tokens=tuple(generated),
            expected_window_tokens=tuple(expected),
            required_window_count=required_count,
            tool_window_count=tool_count,
        )
        for (
            actions,
            machine,
            generated,
            expected,
            required_count,
            tool_count,
        ) in zip(
            generated_actions,
            machines,
            generated_windows,
            expected_windows,
            required_window_counts,
            tool_window_counts,
        )
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
    window_tool_events: tuple[str, ...] = WINDOW_TOOL_EVENTS,
    train_max_length: int | None = None,
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
            elif task == "pointer_next":
                if not isinstance(vocabulary, PointerNextVocabulary):
                    raise TypeError("pointer_next requires PointerNextVocabulary")
                batch = make_pointer_next_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    device=device,
                )
                max_new_tokens = 2
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
            elif task in {"adjacent_sort", "adjacent_sort_no_tool"}:
                if not isinstance(vocabulary, AdjacentSortVocabulary):
                    raise TypeError(f"{task} requires AdjacentSortVocabulary")
                batch = make_adjacent_sort_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        task == "adjacent_sort_no_tool"
                    ),
                    device=device,
                )
                if task == "adjacent_sort":
                    max_new_tokens = 0
                else:
                    max_new_tokens = max(
                        len(trace.target_tokens) for trace in batch.traces
                    ) + 8
            elif task in {
                "adjacent_sort_auto_advance",
                "adjacent_sort_auto_advance_no_tool",
            }:
                if not isinstance(vocabulary, AutoAdvanceSortVocabulary):
                    raise TypeError(
                        f"{task} requires AutoAdvanceSortVocabulary"
                    )
                batch = make_auto_advance_sort_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    supervise_observations=(
                        task == "adjacent_sort_auto_advance_no_tool"
                    ),
                    device=device,
                )
                if task != "adjacent_sort_auto_advance_no_tool":
                    max_new_tokens = 0
                else:
                    max_new_tokens = max(
                        len(trace.target_tokens) for trace in batch.traces
                    ) + 8
            elif task == "adjacent_sort_local_window":
                if not isinstance(vocabulary, LocalWindowSortVocabulary):
                    raise TypeError(
                        f"{task} requires LocalWindowSortVocabulary"
                    )
                batch = make_local_window_sort_batch(
                    current_batch_size,
                    int(length),
                    generator=generator,
                    vocabulary=vocabulary,
                    tool_events=window_tool_events,
                    device=device,
                )
                max_new_tokens = 0
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
                elif task == "adjacent_sort":
                    assert isinstance(batch, AdjacentSortBatch)
                    assert isinstance(vocabulary, AdjacentSortVocabulary)
                    rollouts = generate_adjacent_sort_rollouts(
                        model,
                        batch,
                        vocabulary,
                    )
                elif task == "adjacent_sort_no_tool":
                    assert isinstance(vocabulary, AdjacentSortVocabulary)
                    generated = model.generate(
                        batch.prompt_ids,
                        max_new_tokens=max_new_tokens,
                        stop_token=vocabulary.action_token("DONE"),
                    )
                elif task == "adjacent_sort_auto_advance":
                    assert isinstance(batch, AdjacentSortBatch)
                    assert isinstance(vocabulary, AutoAdvanceSortVocabulary)
                    rollouts = generate_auto_advance_sort_rollouts(
                        model,
                        batch,
                        vocabulary,
                    )
                elif task == "adjacent_sort_auto_advance_no_tool":
                    assert isinstance(vocabulary, AutoAdvanceSortVocabulary)
                    generated = model.generate(
                        batch.prompt_ids,
                        max_new_tokens=max_new_tokens,
                        stop_token=vocabulary.action_token("DONE"),
                    )
                elif task == "adjacent_sort_local_window":
                    assert isinstance(batch, LocalWindowSortBatch)
                    assert isinstance(vocabulary, LocalWindowSortVocabulary)
                    rollouts = generate_local_window_sort_rollouts(
                        model,
                        batch,
                        vocabulary,
                        tool_events=window_tool_events,
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
            elif task == "pointer_next":
                assert isinstance(batch, PointerNextBatch)
                assert isinstance(vocabulary, PointerNextVocabulary)
                train_max_pointer_index = (
                    None if train_max_length is None else train_max_length - 2
                )
                metrics = generated_pointer_next_metrics(
                    batch.values.cpu(),
                    batch.pointers.cpu(),
                    generated.cpu(),
                    vocabulary,
                    train_max_pointer_index=train_max_pointer_index,
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
            elif task == "pointer_quicksort_no_tool":
                assert isinstance(batch, PointerQuicksortBatch)
                assert isinstance(vocabulary, PointerQuicksortVocabulary)
                metrics = generated_pointer_no_tool_metrics(
                    batch.values.cpu(),
                    generated.cpu(),
                    vocabulary,
                    batch.traces,
                )
            elif task == "adjacent_sort":
                assert isinstance(batch, AdjacentSortBatch)
                assert isinstance(vocabulary, AdjacentSortVocabulary)
                metrics = generated_adjacent_sort_metrics(
                    batch.values.cpu(),
                    rollouts,
                    vocabulary,
                    batch.traces,
                )
            elif task == "adjacent_sort_no_tool":
                assert isinstance(batch, AdjacentSortBatch)
                assert isinstance(vocabulary, AdjacentSortVocabulary)
                metrics = generated_adjacent_no_tool_metrics(
                    batch.values.cpu(),
                    generated.cpu(),
                    vocabulary,
                    batch.traces,
                )
            elif task == "adjacent_sort_auto_advance":
                assert isinstance(batch, AdjacentSortBatch)
                assert isinstance(vocabulary, AutoAdvanceSortVocabulary)
                metrics = generated_auto_advance_sort_metrics(
                    batch.values.cpu(),
                    rollouts,
                    vocabulary,
                    batch.traces,
                )
            elif task == "adjacent_sort_local_window":
                assert isinstance(batch, LocalWindowSortBatch)
                metrics = generated_local_window_sort_metrics(
                    batch.values.cpu(),
                    rollouts,
                    batch.traces,
                )
            else:
                assert isinstance(batch, AdjacentSortBatch)
                assert isinstance(vocabulary, AutoAdvanceSortVocabulary)
                metrics = generated_auto_advance_no_tool_metrics(
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
