from __future__ import annotations

from copy import deepcopy

import torch

from list_sorting_transformer.maml_length_generalization import (
    MAMLLengthConfig,
    make_model,
    one_step_maml_objective,
)
from list_sorting_transformer.shortcut_credit import make_clean_pointer_batch
from list_sorting_transformer.tokens import PointerNextVocabulary


def test_one_step_maml_keeps_virtual_weights_temporary() -> None:
    config = MAMLLengthConfig(
        steps=1,
        batch_size=4,
        max_length=4,
        meta_length=6,
        meta_examples=4,
        meta_batch_size=4,
        heldout_length=8,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
    )
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    original_state = deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(17)
    short_batch = make_clean_pointer_batch(
        4,
        4,
        generator=generator,
        vocabulary=vocabulary,
    )
    meta_batch = make_clean_pointer_batch(
        4,
        6,
        generator=generator,
        vocabulary=vocabulary,
    )

    objective = one_step_maml_objective(
        model,
        short_batch,
        meta_batch,
        inner_learning_rate=1e-2,
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_state[name])
    objective.meta_loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(objective.meta_loss)


def test_maml_meta_gradient_includes_the_virtual_short_step() -> None:
    config = MAMLLengthConfig(
        steps=1,
        batch_size=4,
        max_length=4,
        meta_length=6,
        meta_examples=4,
        meta_batch_size=4,
        heldout_length=8,
        eval_examples=4,
        eval_batch_size=4,
        d_model=16,
        layers=1,
        heads=1,
    )
    vocabulary = PointerNextVocabulary("numbers", 10)
    model = make_model(config, vocabulary, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(23)
    short_batch = make_clean_pointer_batch(
        4,
        4,
        generator=generator,
        vocabulary=vocabulary,
    )
    meta_batch = make_clean_pointer_batch(
        4,
        6,
        generator=generator,
        vocabulary=vocabulary,
    )

    objective = one_step_maml_objective(
        model,
        short_batch,
        meta_batch,
        inner_learning_rate=0.1,
    )
    maml_gradients = torch.autograd.grad(
        objective.meta_loss,
        tuple(model.parameters()),
    )
    direct_meta_loss = torch.nn.functional.cross_entropy(
        model(meta_batch.input_ids)[:, -1],
        meta_batch.targets,
    )
    direct_gradients = torch.autograd.grad(
        direct_meta_loss,
        tuple(model.parameters()),
    )

    difference = sum(
        float((maml - direct).square().sum())
        for maml, direct in zip(maml_gradients, direct_gradients)
    )
    assert difference > 1e-8
