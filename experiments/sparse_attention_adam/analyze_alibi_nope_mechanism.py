"""Inspect the learned circuit in a mixed ALiBi/NoPE pointer-next model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor

from list_sorting_transformer.core.data import (
    PointerNextBatch,
    make_pointer_next_batch,
)
from list_sorting_transformer.core.tokens import (
    BOS,
    COMMA,
    SEP,
    VALUE_OFFSET,
)
from list_sorting_transformer.length_generalisation.sparse_attention_adam import (
    AdaptiveEntmaxSelfAttention,
    SparseAttentionAdamConfig,
    SparseAttentionPointerTransformer,
    entmax15,
    pointer_targets,
)


ROLE_NAMES = (
    "bos",
    "comma",
    "ptr",
    "marked_value",
    "target_value",
    "other_value",
    "separator",
)
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}


@dataclass(frozen=True)
class AttentionComponents:
    weights: Tensor
    output: Tensor


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attention_components(
    attention: AdaptiveEntmaxSelfAttention,
    hidden: Tensor,
    *,
    head_mask: Tensor | None = None,
) -> AttentionComponents:
    """Reconstruct attention weights and optionally mask per-head outputs."""

    batch, length, model_dim = hidden.shape
    query, key, value = attention.qkv(hidden).chunk(3, dim=-1)
    query = attention._split_heads(query)
    key = attention._split_heads(key)
    value = attention._split_heads(value)

    if attention.scaling_mode == "adaptive":
        assert attention.beta_projection is not None
        assert attention.gamma_projection is not None
        beta = torch.nn.functional.softplus(
            attention.beta_projection(hidden)
        ).transpose(1, 2)
        gamma = attention.scale_gamma_range * torch.tanh(
            attention.gamma_projection(hidden)
        ).transpose(1, 2)
        log_position = torch.arange(
            2,
            length + 2,
            device=hidden.device,
            dtype=torch.float32,
        ).log()[None, None, :]
        scaler = attention.scale_delta + beta.float() * log_position.pow(
            gamma.float()
        )
        query = torch.cat(
            (
                query[:, : attention.alibi_heads],
                query[:, attention.alibi_heads :]
                * scaler.to(query.dtype).unsqueeze(-1),
            ),
            dim=1,
        )

    scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
    positions = torch.arange(length, device=hidden.device)
    relative_distance = positions[None, :] - positions[:, None]
    scores = scores + (
        attention.slopes.to(scores.dtype)[None, :, None, None]
        * relative_distance[None, None].to(scores.dtype)
    )
    causal_mask = positions[None, :] <= positions[:, None]
    scores = scores.float().masked_fill(
        ~causal_mask[None, None],
        -1e9,
    )
    if attention.attention_normalizer == "entmax15":
        weights = entmax15(scores)
    else:
        weights = scores.softmax(dim=-1)
    weights = weights.masked_fill(~causal_mask[None, None], 0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )

    attended = weights.to(value.dtype) @ value
    if head_mask is not None:
        if head_mask.shape != (attention.heads,):
            raise ValueError("head mask must have shape [heads]")
        attended = attended * head_mask.to(
            device=attended.device,
            dtype=attended.dtype,
        )[None, :, None, None]
    merged = attended.transpose(1, 2).reshape(batch, length, model_dim)
    return AttentionComponents(weights=weights, output=attention.output(merged))


def source_roles(batch: PointerNextBatch) -> Tensor:
    """Classify each prompt position by its role in the pointer-next task."""

    prompt_ids = batch.prompt_ids
    roles = torch.full_like(prompt_ids, ROLE_INDEX["other_value"])
    roles[prompt_ids == BOS] = ROLE_INDEX["bos"]
    roles[prompt_ids == COMMA] = ROLE_INDEX["comma"]
    roles[prompt_ids == SEP] = ROLE_INDEX["separator"]

    pointers = batch.pointers
    rows = torch.arange(batch.values.shape[0])
    ptr_positions = 1 + 2 * pointers
    marked_positions = ptr_positions + 1
    target_positions = ptr_positions + 3
    roles[rows, ptr_positions] = ROLE_INDEX["ptr"]
    roles[rows, marked_positions] = ROLE_INDEX["marked_value"]
    roles[rows, target_positions] = ROLE_INDEX["target_value"]
    return roles


def query_positions(batch: PointerNextBatch) -> dict[str, Tensor]:
    ptr_positions = 1 + 2 * batch.pointers
    return {
        "marked_value": ptr_positions + 1,
        "target_value": ptr_positions + 3,
        "final_separator": torch.full_like(
            ptr_positions,
            batch.prompt_length - 1,
        ),
    }


def load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[SparseAttentionAdamConfig, SparseAttentionPointerTransformer, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("experiment") != "pointer_next_asentmax_adam":
        raise ValueError("checkpoint belongs to another experiment")
    config = SparseAttentionAdamConfig(**checkpoint["config"])
    config = replace(config, device=str(device), wandb=False, resume=None)
    if config.architecture != "standard":
        raise ValueError("mechanism analysis currently requires standard blocks")
    if not 0 < config.alibi_heads < config.heads:
        raise ValueError("mechanism analysis requires mixed ALiBi/NoPE heads")
    model = SparseAttentionPointerTransformer(config)
    model.load_state_dict(checkpoint["model"])
    if config.precision == "bfloat16-true" and device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    model.eval()
    return config, model, int(checkpoint["step"])


def make_fixed_batches(
    model: SparseAttentionPointerTransformer,
    *,
    lengths: Sequence[int],
    examples: int,
    seed: int,
) -> dict[int, PointerNextBatch]:
    generator = torch.Generator().manual_seed(seed)
    return {
        length: make_pointer_next_batch(
            examples,
            length,
            generator=generator,
            vocabulary=model.vocabulary,
        )
        for length in lengths
    }


def _batch_slice(batch: PointerNextBatch, start: int, end: int) -> PointerNextBatch:
    return PointerNextBatch(
        token_ids=batch.token_ids[start:end],
        labels=batch.labels[start:end],
        values=batch.values[start:end],
        pointers=batch.pointers[start:end],
        length=batch.length,
        prompt_length=batch.prompt_length,
    )


@contextmanager
def ablate_heads(
    model: SparseAttentionPointerTransformer,
    masks: dict[int, Tensor],
) -> Iterator[None]:
    with ExitStack() as stack:
        for layer_index, mask in masks.items():
            attention = model.encoder.blocks[layer_index].attention
            if not isinstance(attention, AdaptiveEntmaxSelfAttention):
                raise TypeError("analysis requires AdaptiveEntmaxSelfAttention")

            def replace_output(
                module: AdaptiveEntmaxSelfAttention,
                inputs: tuple[Tensor, ...],
                _output: Tensor,
                *,
                selected_mask: Tensor = mask,
            ) -> Tensor:
                return attention_components(
                    module,
                    inputs[0],
                    head_mask=selected_mask,
                ).output

            handle = attention.register_forward_hook(replace_output)
            stack.callback(handle.remove)
        yield


@torch.inference_mode()
def evaluate_accuracy(
    model: SparseAttentionPointerTransformer,
    batch: PointerNextBatch,
    *,
    device: torch.device,
    batch_size: int,
    masks: dict[int, Tensor] | None = None,
) -> float:
    correct = 0
    context = ablate_heads(model, masks or {}) if masks else nullcontext()
    with context:
        for start in range(0, batch.values.shape[0], batch_size):
            end = min(start + batch_size, batch.values.shape[0])
            chunk = _batch_slice(batch, start, end)
            prompt_ids = chunk.prompt_ids.to(device)
            offsets = torch.zeros(
                end - start,
                dtype=torch.long,
                device=device,
            )
            predictions = model(prompt_ids, offsets=offsets).argmax(dim=-1)
            correct += int(
                predictions.cpu().eq(pointer_targets(chunk)).sum()
            )
    return correct / batch.values.shape[0]


def _keep_mask(heads: int, removed: Sequence[int]) -> Tensor:
    mask = torch.ones(heads)
    mask[list(removed)] = 0
    return mask


def intervention_masks(
    config: SparseAttentionAdamConfig,
) -> dict[str, dict[int, Tensor]]:
    interventions: dict[str, dict[int, Tensor]] = {}
    alibi = tuple(range(config.alibi_heads))
    nope = tuple(range(config.alibi_heads, config.heads))
    for layer in range(config.layers):
        for head in range(config.heads):
            interventions[f"layer_{layer}/remove_head_{head}"] = {
                layer: _keep_mask(config.heads, (head,))
            }
        interventions[f"layer_{layer}/remove_alibi_heads"] = {
            layer: _keep_mask(config.heads, alibi)
        }
        interventions[f"layer_{layer}/remove_nope_heads"] = {
            layer: _keep_mask(config.heads, nope)
        }
    interventions["all_layers/remove_alibi_heads"] = {
        layer: _keep_mask(config.heads, alibi)
        for layer in range(config.layers)
    }
    interventions["all_layers/remove_nope_heads"] = {
        layer: _keep_mask(config.heads, nope)
        for layer in range(config.layers)
    }
    return interventions


@torch.inference_mode()
def verify_reconstruction(
    model: SparseAttentionPointerTransformer,
    batch: PointerNextBatch,
    *,
    device: torch.device,
) -> dict[str, float | bool]:
    chunk = _batch_slice(batch, 0, min(8, batch.values.shape[0]))
    prompt_ids = chunk.prompt_ids.to(device)
    offsets = torch.zeros(prompt_ids.shape[0], dtype=torch.long, device=device)
    baseline = model(prompt_ids, offsets=offsets)
    masks = {
        layer: torch.ones(model.config.heads)
        for layer in range(model.config.layers)
    }
    with ablate_heads(model, masks):
        reconstructed = model(prompt_ids, offsets=offsets)
    difference = (baseline.float() - reconstructed.float()).abs()
    return {
        "max_absolute_logit_difference": float(difference.max().cpu()),
        "mean_absolute_logit_difference": float(difference.mean().cpu()),
        "predictions_identical": bool(
            baseline.argmax(dim=-1).eq(
                reconstructed.argmax(dim=-1)
            ).all()
        ),
    }


def _empty_attention_accumulator(
    *,
    layers: int,
    heads: int,
) -> dict[str, dict[str, dict[str, Tensor]]]:
    return {
        f"layer_{layer}/head_{head}": {
            query: {
                "mass": torch.zeros(len(ROLE_NAMES), dtype=torch.float64),
                "argmax": torch.zeros(len(ROLE_NAMES), dtype=torch.long),
                "examples": torch.zeros((), dtype=torch.long),
            }
            for query in ("marked_value", "target_value", "final_separator")
        }
        for layer in range(layers)
        for head in range(heads)
    }


@torch.inference_mode()
def inspect_attention(
    model: SparseAttentionPointerTransformer,
    batch: PointerNextBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    config = model.config
    accumulator = _empty_attention_accumulator(
        layers=config.layers,
        heads=config.heads,
    )

    for start in range(0, batch.values.shape[0], batch_size):
        end = min(start + batch_size, batch.values.shape[0])
        chunk = _batch_slice(batch, start, end)
        prompt_ids = chunk.prompt_ids.to(device)
        offsets = torch.zeros(end - start, dtype=torch.long, device=device)
        hidden_inputs: dict[int, Tensor] = {}
        handles = []
        for layer, block in enumerate(model.encoder.blocks):
            def capture(
                _module: AdaptiveEntmaxSelfAttention,
                inputs: tuple[Tensor, ...],
                *,
                layer_index: int = layer,
            ) -> None:
                hidden_inputs[layer_index] = inputs[0]

            handles.append(block.attention.register_forward_pre_hook(capture))
        model(prompt_ids, offsets=offsets)
        for handle in handles:
            handle.remove()

        roles = source_roles(chunk).to(device)
        queries = {
            name: positions.to(device)
            for name, positions in query_positions(chunk).items()
        }
        rows = torch.arange(end - start, device=device)
        for layer, block in enumerate(model.encoder.blocks):
            attention = block.attention
            components = attention_components(
                attention,
                hidden_inputs[layer],
            )
            for query_name, positions in queries.items():
                selected = components.weights[
                    rows[:, None],
                    torch.arange(config.heads, device=device)[None, :],
                    positions[:, None],
                    :,
                ]
                argmax_roles = roles.gather(
                    1,
                    selected.argmax(dim=-1),
                )
                for head in range(config.heads):
                    entry = accumulator[f"layer_{layer}/head_{head}"][
                        query_name
                    ]
                    head_weights = selected[:, head]
                    for role_index in range(len(ROLE_NAMES)):
                        role_mask = roles.eq(role_index)
                        entry["mass"][role_index] += (
                            head_weights * role_mask
                        ).sum().double().cpu()
                        entry["argmax"][role_index] += (
                            argmax_roles[:, head].eq(role_index).sum().cpu()
                        )
                    entry["examples"] += end - start

    result: dict[str, object] = {}
    slopes = model.encoder.blocks[0].attention.slopes.cpu()
    for key, queries in accumulator.items():
        head = int(key.rsplit("_", maxsplit=1)[1])
        head_result: dict[str, object] = {
            "head_type": "alibi" if slopes[head] > 0 else "nope",
            "alibi_slope": float(slopes[head]),
            "queries": {},
        }
        for query_name, entry in queries.items():
            examples = int(entry["examples"])
            mass = entry["mass"] / examples
            argmax = entry["argmax"].double() / examples
            head_result["queries"][query_name] = {
                "mean_attention_mass": {
                    role: float(mass[index])
                    for index, role in enumerate(ROLE_NAMES)
                },
                "argmax_fraction": {
                    role: float(argmax[index])
                    for index, role in enumerate(ROLE_NAMES)
                },
            }
        result[key] = head_result
    return result


def render_ablation_plot(
    result: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "alibi-nope-mechanism"
    config = result["model"]
    assert isinstance(config, dict)
    layers = int(config["layers"])
    heads = int(config["heads"])
    alibi_heads = int(config["alibi_heads"])
    evaluations = result["ablation_accuracy"]
    assert isinstance(evaluations, dict)
    lengths = [int(length) for length in result["evaluation"]["lengths"]]

    figure, axes = plt.subplots(
        layers,
        len(lengths),
        figsize=(4.2 * len(lengths), 3.2 * layers),
        squeeze=False,
        sharey=True,
    )
    for layer in range(layers):
        for column, length in enumerate(lengths):
            axis = axes[layer][column]
            baseline = float(evaluations[str(length)]["baseline"])
            drops = [
                baseline
                - float(
                    evaluations[str(length)]["interventions"][
                        f"layer_{layer}/remove_head_{head}"
                    ]
                )
                for head in range(heads)
            ]
            drops.extend(
                (
                    baseline
                    - float(
                        evaluations[str(length)]["interventions"][
                            f"layer_{layer}/remove_alibi_heads"
                        ]
                    ),
                    baseline
                    - float(
                        evaluations[str(length)]["interventions"][
                            f"layer_{layer}/remove_nope_heads"
                        ]
                    ),
                )
            )
            colors = [
                "#007C91" if head < alibi_heads else "#D1495B"
                for head in range(heads)
            ]
            colors.extend(("#007C91", "#D1495B"))
            bars = axis.bar(range(heads + 2), drops, color=colors)
            bars[-2].set_hatch("//")
            bars[-1].set_hatch("//")
            axis.axhline(0, color="#333333", linewidth=0.8)
            axis.set_title(f"Layer {layer + 1}, length {length}")
            axis.set_xlabel("Removed output")
            axis.set_xticks(
                range(heads + 2),
                [*(f"H{head}" for head in range(heads)), "ALiBi", "NoPE"],
                rotation=35,
            )
            if column == 0:
                axis.set_ylabel("Accuracy drop")
    figure.suptitle(
        "Causal effect of zeroing attention-head outputs\n"
        "ALiBi in teal; NoPE in red; hatched bars remove the whole group"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, metadata={"Date": None})
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=(20, 100, 400),
    )
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    if args.examples < 1 or args.batch_size < 1:
        parser.error("examples and batch size must be positive")

    device = torch.device(args.device)
    config, model, step = load_model(args.checkpoint, device=device)
    batches = make_fixed_batches(
        model,
        lengths=args.lengths,
        examples=args.examples,
        seed=args.seed,
    )
    reconstruction = verify_reconstruction(
        model,
        batches[min(args.lengths)],
        device=device,
    )
    interventions = intervention_masks(config)
    ablations: dict[str, object] = {}
    for length in args.lengths:
        batch = batches[length]
        baseline = evaluate_accuracy(
            model,
            batch,
            device=device,
            batch_size=args.batch_size,
        )
        intervention_results = {
            name: evaluate_accuracy(
                model,
                batch,
                device=device,
                batch_size=args.batch_size,
                masks=masks,
            )
            for name, masks in interventions.items()
        }
        ablations[str(length)] = {
            "baseline": baseline,
            "interventions": intervention_results,
        }

    attention: dict[str, object] = {}
    for length in args.lengths:
        attention[str(length)] = inspect_attention(
            model,
            batches[length],
            device=device,
            batch_size=args.batch_size,
        )

    result: dict[str, object] = {
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": _checkpoint_sha256(args.checkpoint),
            "step": step,
            "wandb_run_id": "nxfdvxfw",
            "wandb_url": (
                "https://wandb.ai/wobrob101/"
                "list-sorting-sparse-attention-ablation/runs/nxfdvxfw"
            ),
        },
        "model": {
            "layers": config.layers,
            "heads": config.heads,
            "alibi_heads": config.alibi_heads,
            "d_model": config.d_model,
            "attention_normalizer": config.attention_normalizer,
            "scaling_mode": config.scaling_mode,
            "input_position_mode": config.input_position_mode,
        },
        "evaluation": {
            "lengths": list(args.lengths),
            "examples_per_length": args.examples,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "reconstruction_check": reconstruction,
        "ablation_accuracy": ablations,
        "attention_by_source_role": attention,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_plot is not None:
        render_ablation_plot(result, args.output_plot)
    print(json.dumps(result["reconstruction_check"], indent=2))
    print(f"Wrote {args.output_json}")
    if args.output_plot is not None:
        print(f"Wrote {args.output_plot}")


if __name__ == "__main__":
    main()
