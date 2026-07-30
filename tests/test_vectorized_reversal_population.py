from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from list_sorting_transformer.shortcut_learning.shortcut_credit import (
    ShortcutPointerVocabulary,
    evaluate_shortcut_batches,
    make_fitness_batches,
    make_shortcut_batch,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
    initialize_forward_model,
)
from list_sorting_transformer.shortcut_learning.token_gradient_reversal import (
    forward_with_source_gradient_reversal,
    oracle_shortcut_selection,
)
from list_sorting_transformer.shortcut_learning.vectorized_reversal_population import (
    functional_adam_step,
    train_vectorized_candidate_shard,
)


def test_functional_adam_matches_torch_adam() -> None:
    ordinary = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = torch.optim.Adam([ordinary], lr=1e-4)
    parameters = {"weight": ordinary.detach().clone()[None]}
    first = {"weight": torch.zeros_like(parameters["weight"])}
    second = {"weight": torch.zeros_like(parameters["weight"])}

    for step in range(1, 4):
        gradient = torch.tensor([0.2 * step, -0.1 * step])
        optimizer.zero_grad(set_to_none=True)
        ordinary.grad = gradient.clone()
        optimizer.step()
        parameters, first, second = functional_adam_step(
            parameters,
            {"weight": gradient[None]},
            first,
            second,
            step=step,
            learning_rate=1e-4,
        )

    torch.testing.assert_close(parameters["weight"][0], ordinary)


def test_vectorized_population_matches_serial_candidates() -> None:
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
        forward_layers=1,
        heads=2,
        forward_learning_rate=1e-4,
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
    actions = (
        tuple(
            torch.zeros_like(batch.input_ids, dtype=torch.bool)
            for batch in inner_batches
        ),
        tuple(
            oracle_shortcut_selection(batch.input_ids, vocabulary)
            for batch in inner_batches
        ),
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
    initial_loss = evaluate_shortcut_batches(
        base,
        fitness_batches,
    ).loss

    vectorized = train_vectorized_candidate_shard(
        candidate_indices=(0, 1),
        device_name="cpu",
        config=config,
        base_state=base_state,
        inner_batches=inner_batches,
        actions=actions,
        fitness_batches=fitness_batches,
        heldout_batches=heldout_batches,
        correct_batches=correct_batches,
        initial_clean_loss=initial_loss,
        reversal_scale=4.0,
        vocabulary=vocabulary,
    )

    for candidate_index, candidate_actions in enumerate(actions):
        model = initialize_forward_model(
            config,
            vocabulary,
            initialization_seed=None,
            device=torch.device("cpu"),
        )
        model.load_state_dict(base_state)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.forward_learning_rate,
        )
        for batch, selection in zip(inner_batches, candidate_actions):
            optimizer.zero_grad(set_to_none=True)
            logits = forward_with_source_gradient_reversal(
                model,
                batch.input_ids,
                selection,
                reversal_scale=4.0,
                reversal_scope="attention_scores",
            )
            F.cross_entropy(logits[:, -1], batch.targets).backward()
            optimizer.step()

        expected = evaluate_shortcut_batches(model, fitness_batches)
        assert vectorized[candidate_index].clean.loss == pytest.approx(
            expected.loss,
            abs=1e-6,
        )
        assert (
            vectorized[candidate_index].clean.accuracy
            == expected.accuracy
        )
