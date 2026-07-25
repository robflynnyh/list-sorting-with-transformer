"""Test whether a compiled routing circuit transfers to byte language modelling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .compiled_pointer_compare import (
    DEFAULT_POSITION_MODULI,
    CompiledPointerCompareConfig,
    CompiledPointerCompareTransformer,
    _set_modular_fourier_codebooks,
)
from .evaluate import resolve_device
from .model import ModelConfig, SplitInputDecoderTransformer
from .positions import ModularPositionEmbedding


INITIALIZATIONS = ("random", "compiled_middle")
DEFAULT_DATA_PATH = Path("/store/store4/data/thepile/00.jsonl")
LENGTH_GENERALIZATION_CONTEXTS = (256, 512, 1_024, 2_048)
DEFAULT_EVALUATION_STEPS = (
    0,
    10,
    50,
    100,
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
)


@dataclass(frozen=True)
class ByteCorpus:
    train: Tensor
    validation: Tensor
    metadata: dict[str, object]


@dataclass(frozen=True)
class LanguageModelTransferConfig:
    initialization: str
    seed: int
    data_path: str = str(DEFAULT_DATA_PATH)
    steps: int = 5_000
    batch_size: int = 64
    sequence_length: int = 256
    train_bytes: int = 32_000_000
    validation_bytes: int = 4_000_000
    validation_document_stride: int = 20
    evaluation_batches: int = 20
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "bf16"
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    ffn_multiplier: float = 4.0
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI

    def __post_init__(self) -> None:
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(
                f"initialization must be one of {INITIALIZATIONS}"
            )
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.train_bytes <= self.sequence_length:
            raise ValueError("train_bytes must exceed sequence_length")
        if self.validation_bytes <= self.sequence_length:
            raise ValueError("validation_bytes must exceed sequence_length")
        if self.validation_document_stride < 2:
            raise ValueError("validation_document_stride must be at least two")
        if self.evaluation_batches < 1:
            raise ValueError("evaluation_batches must be positive")
        if not 0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("invalid learning-rate range")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer configuration")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be float32 or bf16")
        if self.d_model != 128:
            raise ValueError("compiled transfer requires d_model=128")
        if self.n_layers != 6 or self.n_heads != 4:
            raise ValueError("compiled-middle transfer targets six layers and four heads")
        if self.ffn_multiplier != 4.0:
            raise ValueError("compiled transfer requires ffn_multiplier=4")
        if self.position_moduli != DEFAULT_POSITION_MODULI:
            raise ValueError("compiled transfer requires the source position moduli")


def load_byte_corpus(
    path: Path,
    *,
    train_bytes: int,
    validation_bytes: int,
    validation_document_stride: int,
) -> ByteCorpus:
    """Read bounded train/validation byte streams without modifying the source."""

    if not path.is_file():
        raise FileNotFoundError(path)
    train = bytearray()
    validation = bytearray()
    train_documents = 0
    validation_documents = 0
    documents_read = 0

    with path.open("r", encoding="utf-8") as source:
        for document_index, line in enumerate(source):
            if len(train) >= train_bytes and len(validation) >= validation_bytes:
                break
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"document {document_index} has no string 'text' field"
                )
            encoded = text.encode("utf-8") + b"\n\n"
            if document_index % validation_document_stride == 0:
                if len(validation) < validation_bytes:
                    validation.extend(encoded)
                    validation_documents += 1
            elif len(train) < train_bytes:
                train.extend(encoded)
                train_documents += 1
            documents_read += 1
    if len(train) < train_bytes or len(validation) < validation_bytes:
        raise ValueError(
            "source ended before the requested train/validation byte budgets"
        )

    del train[train_bytes:]
    del validation[validation_bytes:]
    train_hash = hashlib.sha256(train).hexdigest()
    validation_hash = hashlib.sha256(validation).hexdigest()
    stat = path.stat()
    metadata: dict[str, object] = {
        "source_path": str(path.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "split_rule": (
            f"document_index % {validation_document_stride} == 0 is validation"
        ),
        "documents_read": documents_read,
        "train_documents": train_documents,
        "validation_documents": validation_documents,
        "train_bytes": train_bytes,
        "validation_bytes": validation_bytes,
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "tokenization": "raw UTF-8 bytes, vocabulary size 256",
    }
    return ByteCorpus(
        train=torch.frombuffer(train, dtype=torch.uint8).clone(),
        validation=torch.frombuffer(validation, dtype=torch.uint8).clone(),
        metadata=metadata,
    )


class ByteLanguageModel(nn.Module):
    """Six-layer causal byte LM with fixed modular absolute positions."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        position_moduli: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.encoder = SplitInputDecoderTransformer(
            model_config,
            content_dim=model_config.d_model // 2,
        )
        self.position_embedding = ModularPositionEmbedding(
            self.encoder.position_dim,
            position_moduli,
        )
        with torch.no_grad():
            _set_modular_fourier_codebooks(self.position_embedding)
        self.position_embedding.requires_grad_(False)
        self.lm_head = nn.Linear(
            model_config.d_model,
            model_config.vocab_size,
            bias=False,
        )
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def input_embeddings(self, token_ids: Tensor) -> Tensor:
        positions = torch.arange(
            token_ids.shape[1],
            device=token_ids.device,
        )
        position_embeddings = self.position_embedding(positions)
        content = self.encoder.embed(token_ids)
        return torch.cat(
            (
                content,
                position_embeddings.unsqueeze(0).expand(
                    token_ids.shape[0],
                    -1,
                    -1,
                ),
            ),
            dim=-1,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.input_embeddings(token_ids)
        for block in self.encoder.blocks:
            hidden = block(hidden)
        return self.lm_head(self.encoder.final_norm(hidden))


def build_language_model(
    config: LanguageModelTransferConfig,
) -> ByteLanguageModel:
    """Build a matched model, replacing only layers 3-4 when requested."""

    torch.manual_seed(config.seed)
    model_config = ModelConfig(
        vocab_size=256,
        representation="alphabet",
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_multiplier=config.ffn_multiplier,
        dropout=0.0,
        position_pattern="none",
    )
    model = ByteLanguageModel(
        model_config,
        position_moduli=config.position_moduli,
    )
    if config.initialization == "compiled_middle":
        compiled = CompiledPointerCompareTransformer(
            CompiledPointerCompareConfig(
                pointer_selection_logit=20.0,
                address_score_scale=5.0,
                pointer_scratch_scale=1.0,
            )
        )
        for source_index, target_index in enumerate((2, 3)):
            model.encoder.blocks[target_index].load_state_dict(
                compiled.encoder.blocks[source_index].state_dict()
            )
    return model


def sample_byte_batch(
    tokens: Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    max_start = tokens.numel() - sequence_length - 1
    if max_start < 1:
        raise ValueError("token stream is too short for the requested sequence")
    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        generator=generator,
    ).to(device)
    offsets = torch.arange(sequence_length + 1, device=device)
    windows = tokens.to(device)[starts[:, None] + offsets[None, :]].long()
    return windows[:, :-1], windows[:, 1:]


def evaluation_batch_size(
    config: LanguageModelTransferConfig,
    sequence_length: int,
) -> int:
    """Keep the number of evaluated bytes per batch constant across lengths."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    reference_bytes = config.batch_size * config.sequence_length
    if reference_bytes % sequence_length:
        raise ValueError(
            "evaluation length must divide the reference bytes per batch"
        )
    return max(1, reference_bytes // sequence_length)


def learning_rate_at_step(
    step: int,
    config: LanguageModelTransferConfig,
) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (
        config.steps - config.warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def evaluation_steps(total_steps: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                step
                for step in (*DEFAULT_EVALUATION_STEPS, total_steps)
                if step <= total_steps
            }
        )
    )


def _parameter_norm(parameters: list[Tensor]) -> float:
    squared = sum(float(parameter.float().square().sum()) for parameter in parameters)
    return math.sqrt(squared)


def _middle_parameters(model: ByteLanguageModel) -> list[Tensor]:
    return [
        parameter
        for index in (2, 3)
        for parameter in model.encoder.blocks[index].parameters()
    ]


def _relative_middle_drift(
    model: ByteLanguageModel,
    initial_parameters: list[Tensor],
) -> float:
    current = _middle_parameters(model)
    difference_norm = math.sqrt(
        sum(
            float(
                (
                    parameter.detach().float().cpu() - initial.float()
                )
                .square()
                .sum()
            )
            for parameter, initial in zip(current, initial_parameters)
        )
    )
    initial_norm = _parameter_norm(initial_parameters)
    return difference_norm / max(initial_norm, 1e-12)


@torch.inference_mode()
def evaluate_language_model(
    model: ByteLanguageModel,
    validation_tokens: Tensor,
    config: LanguageModelTransferConfig,
    *,
    device: torch.device,
    sequence_length: int | None = None,
) -> dict[str, float]:
    model.eval()
    resolved_length = sequence_length or config.sequence_length
    resolved_batch_size = evaluation_batch_size(config, resolved_length)
    generator = torch.Generator().manual_seed(config.seed + 1_000_003)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    use_bf16 = (
        config.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    for _ in range(config.evaluation_batches):
        inputs, targets = sample_byte_batch(
            validation_tokens,
            batch_size=resolved_batch_size,
            sequence_length=resolved_length,
            generator=generator,
            device=device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        total_loss += float(loss)
        total_correct += int(logits.argmax(dim=-1).eq(targets).sum())
        total_tokens += targets.numel()
    cross_entropy = total_loss / total_tokens
    return {
        "sequence_length": resolved_length,
        "batch_size": resolved_batch_size,
        "cross_entropy_nats_per_byte": cross_entropy,
        "bits_per_byte": cross_entropy / math.log(2),
        "byte_perplexity": math.exp(cross_entropy),
        "next_byte_accuracy": total_correct / total_tokens,
        "evaluated_bytes": total_tokens,
    }


def evaluate_length_generalization(
    model: ByteLanguageModel,
    validation_tokens: Tensor,
    config: LanguageModelTransferConfig,
    *,
    device: torch.device,
    lengths: tuple[int, ...] = LENGTH_GENERALIZATION_CONTEXTS,
) -> dict[str, dict[str, float]]:
    return {
        str(length): evaluate_language_model(
            model,
            validation_tokens,
            config,
            device=device,
            sequence_length=length,
        )
        for length in lengths
    }


def run_experiment(
    config: LanguageModelTransferConfig,
    *,
    output_directory: Path,
    device_name: str,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    corpus = load_byte_corpus(
        Path(config.data_path),
        train_bytes=config.train_bytes,
        validation_bytes=config.validation_bytes,
        validation_document_stride=config.validation_document_stride,
    )
    model = build_language_model(config).to(device)
    train_tokens = corpus.train.to(device)
    validation_tokens = corpus.validation.to(device)
    train_generator = torch.Generator().manual_seed(config.seed + 200_003)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_middle = [
        parameter.detach().float().cpu().clone()
        for parameter in _middle_parameters(model)
    ]
    initial_middle_norm = _parameter_norm(initial_middle)
    checks = set(evaluation_steps(config.steps))
    history: list[dict[str, float | int | None]] = []
    recent_losses: list[float] = []
    last_gradient_norm: float | None = None
    started_at = time.monotonic()
    use_bf16 = (
        config.precision == "bf16"
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    def record(step: int) -> None:
        validation = evaluate_language_model(
            model,
            validation_tokens,
            config,
            device=device,
        )
        elapsed = time.monotonic() - started_at
        processed_tokens = step * config.batch_size * config.sequence_length
        row: dict[str, float | int | None] = {
            "step": step,
            "train_cross_entropy": (
                sum(recent_losses) / len(recent_losses)
                if recent_losses
                else None
            ),
            **validation,
            "learning_rate": (
                learning_rate_at_step(max(step, 1), config)
                if step
                else 0.0
            ),
            "gradient_norm": last_gradient_norm,
            "middle_parameter_norm": _parameter_norm(
                [
                    parameter.detach()
                    for parameter in _middle_parameters(model)
                ]
            ),
            "middle_relative_drift": _relative_middle_drift(
                model,
                initial_middle,
            ),
            "elapsed_seconds": elapsed,
            "training_bytes_per_second": (
                processed_tokens / elapsed if elapsed > 0 else 0.0
            ),
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "initialization": config.initialization,
                    "seed": config.seed,
                    **row,
                }
            ),
            flush=True,
        )
        recent_losses.clear()

    record(0)
    model.train()
    for step in range(1, config.steps + 1):
        rate = learning_rate_at_step(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets = sample_byte_batch(
            train_tokens,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            generator=train_generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
            )
        loss.backward()
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                config.gradient_clip,
            )
        )
        optimizer.step()
        recent_losses.append(float(loss.detach()))
        if step in checks:
            record(step)
            model.train()

    length_generalization = evaluate_length_generalization(
        model,
        validation_tokens,
        config,
        device=device,
    )
    checkpoint_path = output_directory / "checkpoint.pt"
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "model_state_dict": {
                name: parameter.detach().cpu()
                for name, parameter in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    result: dict[str, Any] = {
        "config": asdict(config),
        "dataset": corpus.metadata,
        "device": str(device),
        "effective_precision": "bf16" if use_bf16 else "float32",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "compiled_source_blocks": (
            {"source": [1, 2], "target": [3, 4]}
            if config.initialization == "compiled_middle"
            else None
        ),
        "initial_middle_parameter_norm": initial_middle_norm,
        "history": history,
        "final": history[-1],
        "length_generalization": length_generalization,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    output_path = output_directory / "metrics.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_saved_checkpoint(
    checkpoint_path: Path,
    *,
    device_name: str,
    lengths: tuple[int, ...] = LENGTH_GENERALIZATION_CONTEXTS,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config_values = dict(payload["config"])
    config_values["position_moduli"] = tuple(
        config_values["position_moduli"]
    )
    config = LanguageModelTransferConfig(**config_values)
    corpus = load_byte_corpus(
        Path(config.data_path),
        train_bytes=config.train_bytes,
        validation_bytes=config.validation_bytes,
        validation_document_stride=config.validation_document_stride,
    )
    model = build_language_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    validation_tokens = corpus.validation.to(device)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "config": asdict(config),
        "dataset": corpus.metadata,
        "length_generalization": evaluate_length_generalization(
            model,
            validation_tokens,
            config,
            device=device,
            lengths=lengths,
        ),
    }


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(correction=1)) if len(values) > 1 else 0.0


def summarize_results(input_root: Path) -> dict[str, Any]:
    results = [
        json.loads(path.read_text())
        for path in sorted(input_root.glob("*/metrics.json"))
    ]
    if not results:
        raise ValueError(f"no metrics.json files found under {input_root}")
    summary: dict[str, Any] = {}
    for initialization in INITIALIZATIONS:
        matching = [
            result
            for result in results
            if result["config"]["initialization"] == initialization
        ]
        if not matching:
            continue
        by_step: dict[str, dict[str, dict[str, float]]] = {}
        steps = sorted(
            set.intersection(
                *[
                    {int(row["step"]) for row in result["history"]}
                    for result in matching
                ]
            )
        )
        for step in steps:
            rows = [
                next(
                    row
                    for row in result["history"]
                    if int(row["step"]) == step
                )
                for result in matching
            ]
            metrics: dict[str, dict[str, float]] = {}
            for metric in (
                "cross_entropy_nats_per_byte",
                "bits_per_byte",
                "byte_perplexity",
                "next_byte_accuracy",
                "middle_parameter_norm",
                "middle_relative_drift",
                "training_bytes_per_second",
            ):
                mean, std = _mean_and_std(
                    [float(row[metric]) for row in rows]
                )
                metrics[metric] = {"mean": mean, "std": std}
            by_step[str(step)] = metrics
        summary[initialization] = {
            "seeds": sorted(
                int(result["config"]["seed"]) for result in matching
            ),
            "by_step": by_step,
        }
        if all("length_generalization" in result for result in matching):
            by_length: dict[str, dict[str, dict[str, float]]] = {}
            lengths = sorted(
                set.intersection(
                    *[
                        set(result["length_generalization"])
                        for result in matching
                    ]
                ),
                key=int,
            )
            for length in lengths:
                metrics: dict[str, dict[str, float]] = {}
                for metric in (
                    "cross_entropy_nats_per_byte",
                    "bits_per_byte",
                    "byte_perplexity",
                    "next_byte_accuracy",
                ):
                    mean, std = _mean_and_std(
                        [
                            float(
                                result["length_generalization"][length][
                                    metric
                                ]
                            )
                            for result in matching
                        ]
                    )
                    metrics[metric] = {"mean": mean, "std": std}
                metrics["evaluation"] = {
                    "batch_size": int(
                        matching[0]["length_generalization"][length][
                            "batch_size"
                        ]
                    ),
                    "evaluated_bytes": int(
                        matching[0]["length_generalization"][length][
                            "evaluated_bytes"
                        ]
                    ),
                }
                by_length[length] = metrics
            summary[initialization]["by_length"] = by_length
    by_initialization_and_seed = {
        (
            result["config"]["initialization"],
            int(result["config"]["seed"]),
        ): result
        for result in results
    }
    paired_seeds = sorted(
        set(
            seed
            for initialization, seed in by_initialization_and_seed
            if initialization == "random"
        )
        & set(
            seed
            for initialization, seed in by_initialization_and_seed
            if initialization == "compiled_middle"
        )
    )
    paired_final: dict[str, Any] = {"seeds": paired_seeds}
    for metric in ("bits_per_byte", "next_byte_accuracy"):
        deltas = [
            float(
                by_initialization_and_seed[
                    ("compiled_middle", seed)
                ]["final"][metric]
            )
            - float(
                by_initialization_and_seed[("random", seed)]["final"][metric]
            )
            for seed in paired_seeds
        ]
        mean, std = _mean_and_std(deltas)
        paired_final[f"compiled_minus_random_{metric}"] = {
            "by_seed": {
                str(seed): delta
                for seed, delta in zip(paired_seeds, deltas)
            },
            "mean": mean,
            "std": std,
        }

    initialization_diagnostics: dict[str, Any] = {}
    for initialization in INITIALIZATIONS:
        model = build_language_model(
            LanguageModelTransferConfig(
                initialization=initialization,
                seed=7,
            )
        )
        diagnostics: dict[str, Any] = {}
        for module_name, modules in (
            (
                "middle_attention",
                [
                    model.encoder.blocks[index].attention
                    for index in (2, 3)
                ],
            ),
            (
                "middle_ffn",
                [
                    model.encoder.blocks[index].ffn
                    for index in (2, 3)
                ],
            ),
        ):
            parameters = [
                parameter
                for module in modules
                for parameter in module.parameters()
            ]
            parameter_count = sum(
                parameter.numel() for parameter in parameters
            )
            nonzero_count = sum(
                int(parameter.detach().ne(0).sum())
                for parameter in parameters
            )
            diagnostics[module_name] = {
                "parameters": parameter_count,
                "nonzero_at_initialization": nonzero_count,
                "nonzero_fraction": nonzero_count / parameter_count,
            }
        initialization_diagnostics[initialization] = diagnostics

    return {
        "data": results[0]["dataset"],
        "shared_config": {
            key: value
            for key, value in results[0]["config"].items()
            if key not in {"initialization", "seed"}
        },
        "initializations": summary,
        "paired_final": paired_final,
        "initialization_diagnostics": initialization_diagnostics,
        "run_count": len(results),
    }


def render_plots(summary: dict[str, Any], output_directory: Path) -> None:
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    colors = {
        "random": "#3b82b4",
        "compiled_middle": "#d95f02",
    }
    labels = {
        "random": "Random",
        "compiled_middle": "Compiled middle",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for initialization in INITIALIZATIONS:
        result = summary["initializations"].get(initialization)
        if result is None:
            continue
        steps = sorted(int(step) for step in result["by_step"])
        for axis, metric, ylabel in (
            (axes[0], "bits_per_byte", "Validation bits per byte"),
            (axes[1], "next_byte_accuracy", "Validation next-byte accuracy"),
        ):
            means = [
                result["by_step"][str(step)][metric]["mean"]
                for step in steps
            ]
            stds = [
                result["by_step"][str(step)][metric]["std"]
                for step in steps
            ]
            axis.plot(
                steps,
                means,
                color=colors[initialization],
                label=labels[initialization],
                linewidth=2,
            )
            axis.fill_between(
                steps,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=colors[initialization],
                alpha=0.16,
            )
            axis.set_xlabel("Optimizer updates")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Compiled middle-layer transfer to byte language modelling")
    figure.tight_layout()
    figure.savefig(
        output_directory / "language_model_transfer_learning_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    random = summary["initializations"].get("random")
    compiled = summary["initializations"].get("compiled_middle")
    if random is None or compiled is None:
        return
    steps = sorted(
        set(map(int, random["by_step"]))
        & set(map(int, compiled["by_step"]))
    )
    advantages = [
        random["by_step"][str(step)]["bits_per_byte"]["mean"]
        - compiled["by_step"][str(step)]["bits_per_byte"]["mean"]
        for step in steps
    ]
    figure, axis = plt.subplots(figsize=(6.7, 4.2))
    axis.axhline(0.0, color="#222222", linewidth=1)
    axis.plot(steps, advantages, color="#4c956c", linewidth=2)
    axis.fill_between(
        steps,
        0,
        advantages,
        where=[value >= 0 for value in advantages],
        color="#4c956c",
        alpha=0.2,
    )
    axis.fill_between(
        steps,
        0,
        advantages,
        where=[value < 0 for value in advantages],
        color="#c44e52",
        alpha=0.2,
    )
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Random BPC - compiled-middle BPC")
    axis.set_title("Positive values favour compiled transfer")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_directory / "language_model_transfer_advantage.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    if not all(
        "by_length" in summary["initializations"].get(initialization, {})
        for initialization in INITIALIZATIONS
    ):
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for initialization in INITIALIZATIONS:
        result = summary["initializations"][initialization]["by_length"]
        lengths = sorted(map(int, result))
        for axis, metric, ylabel in (
            (axes[0], "bits_per_byte", "Validation bits per byte"),
            (axes[1], "next_byte_accuracy", "Validation next-byte accuracy"),
        ):
            means = [result[str(length)][metric]["mean"] for length in lengths]
            stds = [result[str(length)][metric]["std"] for length in lengths]
            axis.plot(
                lengths,
                means,
                marker="o",
                color=colors[initialization],
                label=labels[initialization],
                linewidth=2,
            )
            axis.fill_between(
                lengths,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=colors[initialization],
                alpha=0.16,
            )
            axis.set_xscale("log", base=2)
            axis.set_xticks(lengths, [str(length) for length in lengths])
            axis.set_xlabel("Evaluation context length (bytes)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Length generalization after training on 256-byte contexts"
    )
    figure.tight_layout()
    figure.savefig(
        output_directory / "language_model_transfer_length_generalization.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _run_command(args: argparse.Namespace) -> None:
    config = LanguageModelTransferConfig(
        initialization=args.initialization,
        seed=args.seed,
        data_path=args.data_path,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        train_bytes=args.train_bytes,
        validation_bytes=args.validation_bytes,
        evaluation_batches=args.evaluation_batches,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        precision=args.precision,
    )
    run_experiment(
        config,
        output_directory=Path(args.output_directory),
        device_name=args.device,
    )


def _summarize_command(args: argparse.Namespace) -> None:
    output_directory = Path(args.output_directory)
    summary = summarize_results(Path(args.input_root))
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    render_plots(summary, output_directory)


def _evaluate_checkpoint_command(args: argparse.Namespace) -> None:
    lengths = tuple(
        int(value.strip())
        for value in args.lengths.split(",")
        if value.strip()
    )
    result = evaluate_saved_checkpoint(
        Path(args.checkpoint),
        device_name=args.device,
        lengths=lengths,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one task/seed cell")
    run.add_argument(
        "--initialization",
        choices=INITIALIZATIONS,
        required=True,
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    run.add_argument("--steps", type=int, default=5_000)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--sequence-length", type=int, default=256)
    run.add_argument("--train-bytes", type=int, default=32_000_000)
    run.add_argument("--validation-bytes", type=int, default=4_000_000)
    run.add_argument("--evaluation-batches", type=int, default=20)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    run.add_argument("--warmup-steps", type=int, default=100)
    run.add_argument("--weight-decay", type=float, default=0.01)
    run.add_argument(
        "--precision",
        choices=("float32", "bf16"),
        default="bf16",
    )
    run.add_argument("--device", default="auto")
    run.add_argument("--output-directory", required=True)
    run.set_defaults(handler=_run_command)

    summarize = subparsers.add_parser(
        "summarize",
        help="aggregate completed runs and render plots",
    )
    summarize.add_argument("--input-root", required=True)
    summarize.add_argument("--output-directory", required=True)
    summarize.set_defaults(handler=_summarize_command)

    evaluate_checkpoint = subparsers.add_parser(
        "evaluate-checkpoint",
        help="evaluate a saved final state at one or more context lengths",
    )
    evaluate_checkpoint.add_argument("--checkpoint", required=True)
    evaluate_checkpoint.add_argument(
        "--lengths",
        default=",".join(map(str, LENGTH_GENERALIZATION_CONTEXTS)),
    )
    evaluate_checkpoint.add_argument("--device", default="auto")
    evaluate_checkpoint.add_argument("--output", required=True)
    evaluate_checkpoint.set_defaults(handler=_evaluate_checkpoint_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
