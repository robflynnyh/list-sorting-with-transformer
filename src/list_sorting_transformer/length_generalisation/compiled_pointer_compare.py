"""Compile the full pointer-comparison algorithm into the project Transformer."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ..core.data import PointerNextBatch, make_pointer_pair_batch
from ..core.evaluate import resolve_device
from ..core.model import ModelConfig, SplitInputDecoderTransformer
from ..core.positions import ModularPositionEmbedding
from ..core.tokens import SEP, PointerCompareVocabulary, VALUE_OFFSET


DEFAULT_POSITION_MODULI = (2, 3, 5, 7, 11, 13, 17, 19)


@dataclass(frozen=True)
class CompiledPointerCompareConfig:
    """Architecture and numerical margins for the compiled circuit."""

    symbol_count: int = 10
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_multiplier: float = 4.0
    position_moduli: tuple[int, ...] = DEFAULT_POSITION_MODULI
    pointer_selection_logit: float = 40.0
    address_score_scale: float = 100.0
    pointer_scratch_scale: float = 0.01
    comparison_margin: float = 0.5

    def __post_init__(self) -> None:
        if self.symbol_count != 10:
            raise ValueError("the compiled comparison currently requires digits")
        if self.d_model != 128:
            raise ValueError("the compiled circuit targets d_model=128")
        if self.n_layers != 4 or self.n_heads != 4:
            raise ValueError("the compiled circuit targets four layers and heads")
        if self.ffn_multiplier != 4.0:
            raise ValueError("the compiled circuit targets ffn_multiplier=4")
        if self.position_moduli != DEFAULT_POSITION_MODULI:
            raise ValueError(
                "the compiled circuit targets the pipeline's modular positions"
            )
        if self.pointer_selection_logit <= 0:
            raise ValueError("pointer_selection_logit must be positive")
        if self.address_score_scale <= 0:
            raise ValueError("address_score_scale must be positive")
        if self.pointer_scratch_scale <= 0:
            raise ValueError("pointer_scratch_scale must be positive")
        if not 0 < self.comparison_margin < 1:
            raise ValueError("comparison_margin must be in (0, 1)")


class CompiledPointerCompareTransformer(nn.Module):
    """A fixed-weight realization of the complete six-stage algorithm.

    Layer 1 copies the modular address attached to ``<PTR>`` into the final
    separator state. Layer 2 shifts that address by one and three token slots
    in separate heads and retrieves the two list values. The output projection
    compares those values. The remaining two blocks are identity blocks so the
    module exactly matches the learned pipeline's four-block architecture.
    """

    token_code_dim = 19
    pointer_scratch_start = 19
    pointer_scratch_dim = 32
    marked_value_index = 51
    marked_value_negative_index = 52
    following_value_index = 53
    following_value_negative_index = 54

    def __init__(self, config: CompiledPointerCompareConfig) -> None:
        super().__init__()
        self.compiled_config = config
        self.vocabulary = PointerCompareVocabulary(
            "numbers",
            config.symbol_count,
        )
        model_config = ModelConfig(
            vocab_size=self.vocabulary.size,
            symbol_count=config.symbol_count,
            representation="numbers",
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ffn_multiplier=config.ffn_multiplier,
            dropout=0.0,
            position_pattern="none",
        )
        self.encoder = SplitInputDecoderTransformer(
            model_config,
            content_dim=config.d_model // 2,
        )
        self.position_embedding = ModularPositionEmbedding(
            self.encoder.position_dim,
            config.position_moduli,
        )
        self.action_head = nn.Linear(config.d_model, 2, bias=False)
        self._compile_weights()
        self.requires_grad_(False)
        self.eval()

    def input_position_embeddings(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        offsets: Tensor,
    ) -> Tensor:
        token_offsets = torch.arange(sequence_length, device=device)
        return self.position_embedding(offsets[:, None] + token_offsets)

    def input_embeddings(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        content = self.encoder.embed(prompt_ids)
        positions = self.input_position_embeddings(
            prompt_ids.shape[1],
            device=prompt_ids.device,
            offsets=offsets,
        )
        return torch.cat((content, positions), dim=-1)

    def hidden_states(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> Tensor:
        hidden = self.input_embeddings(prompt_ids, offsets=offsets)
        for block in self.encoder.blocks:
            hidden = block(hidden)
        return self.encoder.final_norm(hidden)

    def forward(self, prompt_ids: Tensor, *, offsets: Tensor) -> Tensor:
        return self.action_head(
            self.hidden_states(prompt_ids, offsets=offsets)[:, -1]
        )

    @torch.inference_mode()
    def attention_trace(
        self,
        prompt_ids: Tensor,
        *,
        offsets: Tensor,
    ) -> dict[str, Tensor]:
        """Expose the two algorithmic routing decisions for diagnostics."""
        hidden = self.input_embeddings(prompt_ids, offsets=offsets)
        pointer_logits = self.encoder.blocks[0].attention.query_key_logits(
            self.encoder.blocks[0].attention_norm(hidden),
            query_index=-1,
        )
        hidden = self.encoder.blocks[0](hidden)
        address_logits = self.encoder.blocks[1].attention.query_key_logits(
            self.encoder.blocks[1].attention_norm(hidden),
            query_index=-1,
        )
        return {
            "pointer_logits": pointer_logits[:, 0],
            "marked_address_logits": address_logits[:, 0],
            "following_address_logits": address_logits[:, 1],
        }

    def _compile_weights(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            for block in self.encoder.blocks:
                block.attention_norm.weight.fill_(1)
                block.ffn_norm.weight.fill_(1)
            self.encoder.final_norm.weight.fill_(1)

            token_codes = _orthogonal_zero_mean_codes(
                self.vocabulary.size,
                self.token_code_dim,
            )
            self.encoder.token_embedding.weight[
                :, : self.token_code_dim
            ].copy_(token_codes)
            _set_modular_fourier_codebooks(self.position_embedding)

            base_norm_scale = _base_layer_norm_scale(
                d_model=self.compiled_config.d_model,
                token_norm_squared=1.0,
                position_norm_squared=(
                    4.0 * len(self.compiled_config.position_moduli)
                ),
            )
            self._compile_pointer_lookup(token_codes, base_norm_scale)
            self._compile_value_lookups(token_codes, base_norm_scale)
            self._compile_action_head(token_codes)

    def _compile_pointer_lookup(
        self,
        token_codes: Tensor,
        base_norm_scale: float,
    ) -> None:
        block = self.encoder.blocks[0]
        head_dim = block.attention.head_dim
        query_magnitude = math.sqrt(
            self.compiled_config.pointer_selection_logit
            * math.sqrt(head_dim)
        )
        sep_code = token_codes[SEP]
        pointer_code = token_codes[self.vocabulary.marker_token("PTR")]

        q_weight = block.attention.qkv.weight[: self.compiled_config.d_model]
        k_weight = block.attention.qkv.weight[
            self.compiled_config.d_model : 2 * self.compiled_config.d_model
        ]
        v_weight = block.attention.qkv.weight[
            2 * self.compiled_config.d_model :
        ]
        q_weight[0, : self.token_code_dim] = (
            query_magnitude * sep_code / base_norm_scale
        )
        k_weight[0, : self.token_code_dim] = (
            query_magnitude * pointer_code / base_norm_scale
        )

        for modulus_index in range(len(self.compiled_config.position_moduli)):
            source = (
                self.encoder.content_dim + 8 * modulus_index
            )
            target = 4 * modulus_index
            v_weight[
                target : target + 4,
                source : source + 4,
            ] = (
                torch.eye(4)
                * self.compiled_config.pointer_scratch_scale
                / base_norm_scale
            )

        scratch = slice(
            self.pointer_scratch_start,
            self.pointer_scratch_start + self.pointer_scratch_dim,
        )
        block.attention.output.weight[scratch, :head_dim] = torch.eye(
            head_dim
        )

    def _compile_value_lookups(
        self,
        token_codes: Tensor,
        base_norm_scale: float,
    ) -> None:
        block = self.encoder.blocks[1]
        d_model = self.compiled_config.d_model
        head_dim = block.attention.head_dim
        query_key_magnitude = math.sqrt(
            self.compiled_config.address_score_scale
        )
        pointer_state_norm_squared = (
            1.0
            + 4.0 * len(self.compiled_config.position_moduli)
            + 2.0
            * len(self.compiled_config.position_moduli)
            * self.compiled_config.pointer_scratch_scale**2
        )
        pointer_state_norm_scale = _layer_norm_scale(
            d_model,
            pointer_state_norm_squared,
        )

        q_weight = block.attention.qkv.weight[:d_model]
        k_weight = block.attention.qkv.weight[d_model : 2 * d_model]
        v_weight = block.attention.qkv.weight[2 * d_model :]
        for head_index, shift in enumerate((1, 3)):
            head_start = head_index * head_dim
            for modulus_index, modulus in enumerate(
                self.compiled_config.position_moduli
            ):
                source = self.pointer_scratch_start + 4 * modulus_index
                target = head_start + 4 * modulus_index
                q_weight[
                    target : target + 4,
                    source : source + 4,
                ] = (
                    _residue_shift_matrix(modulus, shift)
                    * query_key_magnitude
                    / (
                        pointer_state_norm_scale
                        * self.compiled_config.pointer_scratch_scale
                    )
                )
                position_source = (
                    self.encoder.content_dim + 8 * modulus_index
                )
                k_weight[
                    target : target + 4,
                    position_source : position_source + 4,
                ] = (
                    torch.eye(4)
                    * query_key_magnitude
                    / base_norm_scale
                )

            digit_reader = torch.zeros(self.token_code_dim)
            for value in range(self.compiled_config.symbol_count):
                digit_reader.add_(
                    value
                    * token_codes[VALUE_OFFSET + value]
                    / base_norm_scale
                )
            v_weight[
                head_start,
                : self.token_code_dim,
            ] = digit_reader

        output = block.attention.output.weight
        output[self.marked_value_index, 0] = 1
        output[self.marked_value_negative_index, 0] = -1
        output[self.following_value_index, head_dim] = 1
        output[self.following_value_negative_index, head_dim] = -1

    def _compile_action_head(self, token_codes: Tensor) -> None:
        direction = torch.zeros(self.compiled_config.d_model)
        direction[: self.token_code_dim] = (
            self.compiled_config.comparison_margin
            * token_codes[SEP]
        )
        direction[self.marked_value_index] = -1
        direction[self.following_value_index] = 1
        self.action_head.weight[0] = direction
        self.action_head.weight[1] = -direction


def _orthogonal_zero_mean_codes(count: int, dimension: int) -> Tensor:
    if count >= dimension:
        raise ValueError("code dimension must exceed token count")
    positions = torch.arange(dimension, dtype=torch.float64) + 0.5
    frequencies = torch.arange(1, count + 1, dtype=torch.float64)[:, None]
    codes = math.sqrt(2.0 / dimension) * torch.cos(
        math.pi * frequencies * positions[None, :] / dimension
    )
    return codes.to(dtype=torch.float32)


def _set_modular_fourier_codebooks(
    embedding: ModularPositionEmbedding,
) -> None:
    for modulus, codebook in zip(embedding.moduli, embedding.codebooks):
        residues = torch.arange(modulus, dtype=torch.float64)
        angles = 2 * math.pi * residues / modulus
        second_angles = 2 * angles
        values = torch.stack(
            (
                angles.cos(),
                -angles.cos(),
                angles.sin(),
                -angles.sin(),
                second_angles.cos(),
                -second_angles.cos(),
                second_angles.sin(),
                -second_angles.sin(),
            ),
            dim=-1,
        )
        codebook.weight.copy_(values.to(dtype=codebook.weight.dtype))


def _residue_shift_matrix(modulus: int, shift: int) -> Tensor:
    angle = 2 * math.pi * shift / modulus
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        (
            (cosine, 0, -sine, 0),
            (0, cosine, 0, -sine),
            (sine, 0, cosine, 0),
            (0, sine, 0, cosine),
        ),
        dtype=torch.float32,
    )


def _layer_norm_scale(d_model: int, norm_squared: float) -> float:
    return 1.0 / math.sqrt(norm_squared / d_model + 1e-5)


def _base_layer_norm_scale(
    *,
    d_model: int,
    token_norm_squared: float,
    position_norm_squared: float,
) -> float:
    return _layer_norm_scale(
        d_model,
        token_norm_squared + position_norm_squared,
    )


def target_action_classes(batch: PointerNextBatch) -> Tensor:
    rows = torch.arange(batch.values.shape[0], device=batch.values.device)
    marked = batch.values[rows, batch.pointers]
    following = batch.values[rows, batch.pointers + 1]
    return marked.gt(following).long()


@torch.inference_mode()
def evaluate_compiled_model(
    model: CompiledPointerCompareTransformer,
    *,
    lengths: tuple[int, ...],
    examples: int,
    batch_size: int,
    seed: int,
    offset_min: int,
    offset_max: int,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    generator = torch.Generator().manual_seed(seed)
    results: dict[str, dict[str, float | int]] = {}
    for length in lengths:
        correct = 0
        pointer_routes = 0
        marked_routes = 0
        following_routes = 0
        minimum_action_margin = float("inf")
        minimum_pointer_route_margin = float("inf")
        minimum_marked_route_margin = float("inf")
        minimum_following_route_margin = float("inf")
        completed = 0
        while completed < examples:
            current_batch_size = min(batch_size, examples - completed)
            batch = make_pointer_pair_batch(
                current_batch_size,
                length,
                generator=generator,
                vocabulary=model.vocabulary,
                device=device,
            )
            offsets = torch.randint(
                offset_min,
                offset_max + 1,
                (current_batch_size,),
                generator=generator,
            ).to(device)
            action_logits = model(batch.prompt_ids, offsets=offsets)
            predictions = action_logits.argmax(-1)
            targets = target_action_classes(batch)
            correct += int(predictions.eq(targets).sum())
            action_margins = (
                action_logits.gather(1, targets[:, None]).squeeze(1)
                - action_logits.gather(
                    1,
                    (1 - targets)[:, None],
                ).squeeze(1)
            )
            minimum_action_margin = min(
                minimum_action_margin,
                float(action_margins.min()),
            )

            trace = model.attention_trace(batch.prompt_ids, offsets=offsets)
            pointer_positions = 1 + 2 * batch.pointers
            pointer_routes += int(
                trace["pointer_logits"].argmax(-1).eq(pointer_positions).sum()
            )
            marked_routes += int(
                trace["marked_address_logits"]
                .argmax(-1)
                .eq(pointer_positions + 1)
                .sum()
            )
            following_routes += int(
                trace["following_address_logits"]
                .argmax(-1)
                .eq(pointer_positions + 3)
                .sum()
            )
            minimum_pointer_route_margin = min(
                minimum_pointer_route_margin,
                _minimum_selected_logit_margin(
                    trace["pointer_logits"],
                    pointer_positions,
                ),
            )
            minimum_marked_route_margin = min(
                minimum_marked_route_margin,
                _minimum_selected_logit_margin(
                    trace["marked_address_logits"],
                    pointer_positions + 1,
                ),
            )
            minimum_following_route_margin = min(
                minimum_following_route_margin,
                _minimum_selected_logit_margin(
                    trace["following_address_logits"],
                    pointer_positions + 3,
                ),
            )
            completed += current_batch_size
        results[str(length)] = {
            "examples": examples,
            "action_accuracy": correct / examples,
            "minimum_action_margin": minimum_action_margin,
            "pointer_route_accuracy": pointer_routes / examples,
            "minimum_pointer_route_logit_margin": (
                minimum_pointer_route_margin
            ),
            "marked_route_accuracy": marked_routes / examples,
            "minimum_marked_route_logit_margin": (
                minimum_marked_route_margin
            ),
            "following_route_accuracy": following_routes / examples,
            "minimum_following_route_logit_margin": (
                minimum_following_route_margin
            ),
        }
    return results


def _minimum_selected_logit_margin(
    logits: Tensor,
    selected_indices: Tensor,
) -> float:
    selected = logits.gather(1, selected_indices[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, selected_indices[:, None], float("-inf"))
    margins = selected - alternatives.max(dim=1).values
    return float(margins.min())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="2,11,20,40,400")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--position-offset-min", type=int, default=-1_000_000)
    parser.add_argument("--position-offset-max", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    lengths = tuple(dict.fromkeys(int(item) for item in args.lengths.split(",")))
    if not lengths or min(lengths) < 2:
        raise ValueError("all lengths must be at least two")
    if args.examples < 1 or args.batch_size < 1:
        raise ValueError("examples and batch_size must be positive")
    if args.position_offset_min > args.position_offset_max:
        raise ValueError("position offset bounds are reversed")

    device = resolve_device(args.device)
    config = CompiledPointerCompareConfig()
    model = CompiledPointerCompareTransformer(config).to(device)
    started_at = time.monotonic()
    results = evaluate_compiled_model(
        model,
        lengths=lengths,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
        offset_min=args.position_offset_min,
        offset_max=args.position_offset_max,
        device=device,
    )
    report = {
        "experiment": "compiled_pointer_compare",
        "device": str(device),
        "compiled_config": asdict(config),
        "model_config": model.encoder.config.as_dict(),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "wall_time_seconds": time.monotonic() - started_at,
        "per_length": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
