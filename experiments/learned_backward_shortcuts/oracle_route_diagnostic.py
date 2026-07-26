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
    ) -> None:
        super().__init__(config)
        self.gate_value = gate_value

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
        gate[:, :, -1, -2] = self.gate_value
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
    if route_output_projection is not None:
        rule = OracleLeakRouter(
            AttentionRoutingRuleConfig(
                vocab_size=vocabulary.size,
                d_model=config.backward_d_model,
                n_heads=config.heads,
                forward_layers=config.forward_layers,
                route_output_projection=route_output_projection,
            ),
            gate_value=gate_value,
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
        for name, route_projection in (
            ("ordinary", None),
            ("qkv_only", False),
            ("complete_attention", True),
        ):
            metrics = train_condition(
                config=config,
                base_state=base_state,
                inner_batches=inner_batches,
                fitness_batches=fitness_batches,
                correct_batches=correct_batches,
                route_output_projection=route_projection,
                gate_value=args.gate_value,
                device=device,
            )
            row = {"horizon": horizon, "condition": name, **metrics}
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in results)
            + "\n"
        )


if __name__ == "__main__":
    main()
