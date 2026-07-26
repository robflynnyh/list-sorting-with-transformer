"""Test whether direct leak-edge suppression can prevent shortcut learning."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor

from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    AttentionRoutingRuleConfig,
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    make_inner_batches,
    make_mode_batches,
)


class OracleLeakRouter(AttentionRoutingRule):
    """Suppress the final query's direct edge to the leaked answer token."""

    def __init__(
        self,
        config: AttentionRoutingRuleConfig,
        *,
        gate_value: float,
        scope: str,
    ) -> None:
        super().__init__(config)
        self.gate_value = gate_value
        if scope not in {"direct", "hint_source"}:
            raise ValueError("unknown oracle routing scope")
        self.scope = scope
        self.leak_token = ShortcutPointerVocabulary(
            "numbers",
            10,
        ).leak_token

    def attention_gates(self, token_ids: Tensor) -> tuple[Tensor, ...]:
        batch_size, sequence_length = token_ids.shape
        gate = torch.ones(
            batch_size,
            self.config.n_heads,
            sequence_length,
            sequence_length,
            device=token_ids.device,
            dtype=self.token_embedding.weight.dtype,
        )
        leak_positions = token_ids.eq(self.leak_token).nonzero(
            as_tuple=False
        )
        if leak_positions.shape[0] != batch_size:
            raise ValueError("every prompt must contain exactly one leak")
        rows = leak_positions[:, 0]
        hint_positions = leak_positions[:, 1] + 1
        if self.scope == "direct":
            gate[rows, :, -1, hint_positions] = self.gate_value
        else:
            for row, hint_position in zip(
                rows.tolist(),
                hint_positions.tolist(),
            ):
                gate[row, :, :, hint_position] = self.gate_value
        return tuple(gate.clone() for _ in range(self.config.forward_layers))


class UniformAttentionRouter(AttentionRoutingRule):
    """Apply the same backward gate to every attention edge."""

    def __init__(
        self,
        config: AttentionRoutingRuleConfig,
        *,
        gate_value: float,
    ) -> None:
        super().__init__(config)
        self.gate_value = gate_value

    def attention_gates(self, token_ids: Tensor) -> tuple[Tensor, ...]:
        batch_size, sequence_length = token_ids.shape
        gate = torch.full(
            (
                batch_size,
                self.config.n_heads,
                sequence_length,
                sequence_length,
            ),
            self.gate_value,
            device=token_ids.device,
            dtype=self.token_embedding.weight.dtype,
        )
        return tuple(gate.clone() for _ in range(self.config.forward_layers))


def train_condition(
    *,
    config: ShortcutCreditExperimentConfig,
    base_state: dict[str, Tensor],
    inner_batches: tuple,
    fitness_batches: tuple,
    correct_batches: tuple,
    route_output_projection: bool | None,
    gate_value: float,
    oracle_scope: str,
    uniform_routing: bool,
    device: torch.device,
) -> dict[str, float]:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    model = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=None,
        device=device,
    )
    model.load_state_dict(deepcopy(base_state))
    rule = None
    if uniform_routing:
        if route_output_projection is None:
            raise ValueError("uniform routing requires a projection setting")
        rule = UniformAttentionRouter(
            AttentionRoutingRuleConfig(
                vocab_size=vocabulary.size,
                d_model=config.backward_d_model,
                n_heads=config.heads,
                forward_layers=config.forward_layers,
                route_output_projection=route_output_projection,
            ),
            gate_value=gate_value,
        ).to(device)
    elif route_output_projection is not None:
        rule = OracleLeakRouter(
            AttentionRoutingRuleConfig(
                vocab_size=vocabulary.size,
                d_model=config.backward_d_model,
                n_heads=config.heads,
                forward_layers=config.forward_layers,
                route_output_projection=route_output_projection,
            ),
            gate_value=gate_value,
            scope=oracle_scope,
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.forward_learning_rate,
    )
    model.train()
    for batch in inner_batches:
        optimizer.zero_grad(set_to_none=True)
        loss = shortcut_loss(model, batch, rule)
        loss.backward()
        optimizer.step()

    clean = evaluate_shortcut_batches(model, fitness_batches)
    correct = evaluate_shortcut_batches(model, correct_batches)
    return {
        "clean_loss": clean.loss,
        "clean_accuracy": clean.accuracy,
        "masked_accuracy": clean.mode_accuracy["masked"],
        "incorrect_accuracy": clean.mode_accuracy["incorrect"],
        "correct_accuracy": correct.accuracy,
        "unique_value_predictions": float(
            clean.unique_value_prediction_count
        ),
        "prediction_mode_fraction": clean.prediction_mode_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="40,80")
    parser.add_argument("--gate-value", type=float, default=1e-6)
    parser.add_argument(
        "--leak-placement",
        choices=("suffix", "random_list"),
        default="suffix",
    )
    parser.add_argument(
        "--oracle-scope",
        choices=("direct", "hint_source"),
        default="direct",
        help="suppress the final-query edge or every edge sourced at the hint",
    )
    parser.add_argument(
        "--include-masked-training",
        action="store_true",
        help="include ordinary Adam trained on otherwise matched masked hints",
    )
    parser.add_argument(
        "--include-uniform-routing",
        action="store_true",
        help="include Q/K/V routing with one gate shared by every edge",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 < args.gate_value <= 1:
        raise ValueError("gate-value must be in (0, 1]")

    device = torch.device(args.device)
    horizons = tuple(int(value) for value in args.horizons.split(","))
    config = ShortcutCreditExperimentConfig(
        generations=1,
        horizon=max(horizons),
        max_horizon=max(horizons),
        seed=args.seed,
        device=args.device,
        leak_placement=args.leak_placement,
    )
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    fitness_generator = torch.Generator().manual_seed(args.seed + 10_000)
    fitness_batches = make_fitness_batches(
        config.fitness_examples,
        min_length=config.min_length,
        max_length=config.max_length,
        batch_size=config.fitness_batch_size,
        generator=fitness_generator,
        vocabulary=vocabulary,
        leak_placement=config.leak_placement,
        device=device,
    )
    correct_batches = make_mode_batches(
        config.correct_eval_examples,
        leak_mode="correct",
        config=config,
        vocabulary=vocabulary,
        generator=fitness_generator,
        device=device,
    )

    results = []
    for horizon in horizons:
        generation_seed = args.seed * 1_000_003 + 50 * 10_007
        base_model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=generation_seed + 1,
            device=device,
        )
        base_state = deepcopy(base_model.state_dict())
        del base_model
        inner_batches = make_inner_batches(
            config,
            horizon=horizon,
            vocabulary=vocabulary,
            generator=torch.Generator().manual_seed(generation_seed + 2),
            device=device,
        )
        masked_inner_batches = (
            make_inner_batches(
                config,
                horizon=horizon,
                vocabulary=vocabulary,
                generator=torch.Generator().manual_seed(
                    generation_seed + 2
                ),
                device=device,
                leak_mode="masked",
            )
            if args.include_masked_training
            else None
        )
        conditions = [("ordinary", None, False)]
        if args.include_uniform_routing:
            conditions.append(("uniform_qkv", False, True))
        conditions.extend(
            (
                ("qkv_only", False, False),
                ("complete_attention", True, False),
            )
        )
        for name, route_projection, uniform_routing in conditions:
            metrics = train_condition(
                config=config,
                base_state=base_state,
                inner_batches=inner_batches,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                route_output_projection=route_projection,
                gate_value=args.gate_value,
                oracle_scope=args.oracle_scope,
                uniform_routing=uniform_routing,
                device=device,
            )
            row = {
                "horizon": horizon,
                "condition": name,
                "oracle_scope": args.oracle_scope,
                **metrics,
            }
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if name == "ordinary" and masked_inner_batches is not None:
                masked_metrics = train_condition(
                    config=config,
                    base_state=base_state,
                    inner_batches=masked_inner_batches,
                    fitness_batches=fitness_batches,
                    correct_batches=correct_batches,
                    route_output_projection=None,
                    gate_value=args.gate_value,
                    oracle_scope=args.oracle_scope,
                    uniform_routing=False,
                    device=device,
                )
                masked_row = {
                    "horizon": horizon,
                    "condition": "masked_training",
                    "oracle_scope": args.oracle_scope,
                    **masked_metrics,
                }
                results.append(masked_row)
                print(
                    json.dumps(masked_row, sort_keys=True),
                    flush=True,
                )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in results)
            + "\n"
        )


if __name__ == "__main__":
    main()
