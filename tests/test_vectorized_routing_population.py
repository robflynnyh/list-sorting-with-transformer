from copy import deepcopy

import pytest
import torch

from list_sorting_transformer.shortcut_credit import (
    AttentionRoutingRule,
    ShortcutPointerVocabulary,
    apply_eggroll_direction,
    clone_center_parameters,
    evaluate_shortcut_batches,
    make_fitness_batches,
    make_shortcut_batch,
    sample_eggroll_direction,
    shortcut_loss,
)
from list_sorting_transformer.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
    initialize_fresh_backward_rule,
)
from list_sorting_transformer.vectorized_routing_population import (
    stack_candidate_rule_parameters,
    train_vectorized_routing_population,
)


def test_vectorized_population_rejects_unsupported_rules() -> None:
    with pytest.raises(
        ValueError,
        match="require an unconditioned attention router",
    ):
        ShortcutCreditExperimentConfig(vectorized_population=True)
    with pytest.raises(
        ValueError,
        match="require an unconditioned attention router",
    ):
        ShortcutCreditExperimentConfig(
            backward_rule_type="attention_router",
            vectorized_population=True,
            route_output_projection=True,
        )


def test_vectorized_routing_population_matches_serial_candidates() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    config = ShortcutCreditExperimentConfig(
        horizon=2,
        max_horizon=2,
        batch_size=4,
        fitness_examples=8,
        fitness_batch_size=4,
        correct_eval_examples=4,
        min_length=4,
        max_length=4,
        d_model=16,
        backward_d_model=16,
        forward_layers=1,
        backward_layers=1,
        heads=2,
        forward_learning_rate=1e-4,
        backward_rule_type="attention_router",
        shared_routing_map=True,
        leak_placement="random_list",
        device="cpu",
    )
    base = initialize_forward_model(
        config,
        vocabulary,
        initialization_seed=17,
        device=torch.device("cpu"),
    )
    base_state = deepcopy(base.state_dict())
    center_rule = initialize_fresh_backward_rule(
        config,
        vocabulary,
        device=torch.device("cpu"),
    )
    assert isinstance(center_rule, AttentionRoutingRule)
    with torch.no_grad():
        center_rule.gates.fill_(0.1)
    center_parameters = clone_center_parameters(center_rule)
    directions = tuple(
        sample_eggroll_direction(
            center_rule,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (41, 42)
    )
    candidate_specs = ((0, 0, 1), (1, 1, -1))
    rule_parameters = stack_candidate_rule_parameters(
        center_parameters,
        directions,
        candidate_specs,
        sigma=0.01,
        device=torch.device("cpu"),
    )
    inner_batches = tuple(
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(seed),
            vocabulary=vocabulary,
        )
        for seed in (21, 22)
    )
    fitness_batches = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(31),
        vocabulary=vocabulary,
        leak_placement="random_list",
    )
    heldout_batches = make_fitness_batches(
        8,
        min_length=4,
        max_length=4,
        batch_size=4,
        generator=torch.Generator().manual_seed(32),
        vocabulary=vocabulary,
        leak_placement="random_list",
    )
    correct_batches = (
        make_shortcut_batch(
            4,
            4,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(33),
            vocabulary=vocabulary,
        ),
    )

    vectorized = train_vectorized_routing_population(
        config=config,
        base_state=base_state,
        center_rule=center_rule,
        rule_parameters=rule_parameters,
        inner_batches=inner_batches,
        fitness_batches=fitness_batches,
        correct_batches=correct_batches,
        heldout_fitness_batches=heldout_batches,
        heldout_correct_batches=correct_batches,
        device=torch.device("cpu"),
    )

    for local_index, (_, direction_index, sign) in enumerate(
        candidate_specs
    ):
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=None,
            device=torch.device("cpu"),
        )
        model.load_state_dict(base_state)
        rule = AttentionRoutingRule(center_rule.config)
        apply_eggroll_direction(
            rule,
            center_parameters,
            directions[direction_index],
            sigma=0.01,
            sign=sign,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.forward_learning_rate,
        )
        for batch in inner_batches:
            optimizer.zero_grad(set_to_none=True)
            shortcut_loss(model, batch, rule).backward()
            optimizer.step()

        expected = evaluate_shortcut_batches(model, fitness_batches)
        actual = vectorized.trajectories[local_index].clean
        assert actual.loss == pytest.approx(expected.loss, abs=2e-6)
        assert actual.accuracy == expected.accuracy
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                vectorized.forward_parameters[
                    f"forward_model.{name}"
                ][local_index],
                parameter,
                rtol=2e-5,
                atol=2e-7,
            )
