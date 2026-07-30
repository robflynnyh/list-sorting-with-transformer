import pytest
import torch

from list_sorting_transformer.shortcut_credit import (
    ShortcutPointerVocabulary,
    make_shortcut_batch,
)
from list_sorting_transformer.token_gradient_selector import (
    SelectorTrajectory,
    TokenGradientSelector,
    TokenGradientSelectorConfig,
    grouped_trajectory_policy_terms,
    sample_selector_trajectory,
    sample_selector_trajectories,
    selector_probability_statistics,
    sinusoidal_positions,
    standardize_group_rewards,
    trajectory_policy_terms,
)


def small_selector() -> TokenGradientSelector:
    return TokenGradientSelector(
        TokenGradientSelectorConfig(
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
        )
    )


def test_sinusoidal_positions_support_odd_dimensions() -> None:
    positions = sinusoidal_positions(
        7,
        15,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert positions.shape == (7, 15)
    assert torch.isfinite(positions).all()


def test_selector_is_bidirectional() -> None:
    torch.manual_seed(3)
    selector = small_selector().eval()
    with torch.no_grad():
        selector.action_head.weight.normal_()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 3, 5]])

    first_logits = selector(first)
    second_logits = selector(second)

    assert not torch.equal(first_logits[:, 0], second_logits[:, 0])


def test_selector_starts_at_configured_reverse_probability() -> None:
    selector = TokenGradientSelector(
        TokenGradientSelectorConfig(
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=2,
            initial_reverse_probability=0.05,
        )
    ).eval()

    probabilities = selector(
        torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [5, 6, 7, 8],
            ]
        )
    ).softmax(dim=-1)[..., 1]

    assert float(probabilities.mean()) == pytest.approx(0.05, abs=0.015)
    assert float(probabilities.max()) < 0.1


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_selector_rejects_degenerate_initial_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        TokenGradientSelectorConfig(
            vocab_size=32,
            initial_reverse_probability=probability,
        )


def test_sampling_and_policy_terms_cover_every_token() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    torch.manual_seed(4)
    selector = TokenGradientSelector(
        TokenGradientSelectorConfig(
            vocab_size=vocabulary.size,
            d_model=16,
            n_layers=2,
            n_heads=2,
        )
    )
    batches = (
        make_shortcut_batch(
            4,
            5,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(5),
            vocabulary=vocabulary,
        ),
    )

    trajectory = sample_selector_trajectory(
        selector,
        batches,
        vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(6),
    )
    log_probability, entropy = trajectory_policy_terms(
        selector,
        batches,
        trajectory.actions,
    )

    assert trajectory.actions[0].shape == batches[0].input_ids.shape
    assert 0 <= trajectory.selected_fraction <= 1
    assert torch.isfinite(log_probability)
    assert entropy > 0


def test_grouped_sampling_and_policy_terms_match_individual_terms() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    torch.manual_seed(8)
    selector = TokenGradientSelector(
        TokenGradientSelectorConfig(
            vocab_size=vocabulary.size,
            d_model=16,
            n_layers=2,
            n_heads=2,
        )
    )
    batches = (
        make_shortcut_batch(
            4,
            5,
            leak_mode="correct",
            leak_placement="random_list",
            generator=torch.Generator().manual_seed(9),
            vocabulary=vocabulary,
        ),
    )
    trajectories = sample_selector_trajectories(
        selector,
        batches,
        group_size=2,
        vocabulary=vocabulary,
        generators=(
            torch.Generator().manual_seed(10),
            torch.Generator().manual_seed(11),
        ),
    )

    grouped_log_probabilities, grouped_entropy = (
        grouped_trajectory_policy_terms(
            selector,
            batches,
            trajectories,
        )
    )
    individual_terms = [
        trajectory_policy_terms(
            selector,
            batches,
            trajectory.actions,
        )
        for trajectory in trajectories
    ]

    assert all(
        isinstance(trajectory, SelectorTrajectory)
        for trajectory in trajectories
    )
    torch.testing.assert_close(
        grouped_log_probabilities,
        torch.stack([terms[0] for terms in individual_terms]),
    )
    torch.testing.assert_close(
        grouped_entropy,
        individual_terms[0][1],
    )


def test_selector_probability_statistics_separates_positions() -> None:
    vocabulary = ShortcutPointerVocabulary("numbers", 10)
    selector = TokenGradientSelector(
        TokenGradientSelectorConfig(
            vocab_size=vocabulary.size,
            d_model=16,
            n_layers=2,
            n_heads=2,
        )
    )
    batch = make_shortcut_batch(
        4,
        5,
        leak_mode="correct",
        generator=torch.Generator().manual_seed(7),
        vocabulary=vocabulary,
    )

    statistics = selector_probability_statistics(
        selector,
        (batch,),
        vocabulary=vocabulary,
    )

    assert set(statistics) == {
        "oracle_reverse_probability",
        "other_reverse_probability",
    }
    assert all(0 <= value <= 1 for value in statistics.values())


def test_standardize_group_rewards() -> None:
    advantages = standardize_group_rewards(
        torch.tensor([1.0, 2.0, 3.0])
    )
    torch.testing.assert_close(advantages.mean(), torch.tensor(0.0))
    torch.testing.assert_close(
        advantages.std(unbiased=False),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        standardize_group_rewards(torch.ones(4)),
        torch.zeros(4),
    )
    torch.testing.assert_close(
        standardize_group_rewards(
            torch.tensor([1.0, 1.00001, 0.99999]),
            minimum_standard_deviation=1e-4,
        ),
        torch.zeros(3),
    )
