from experiments.learned_backward_shortcuts.rejected_elite_checkpoint import (
    corrected_rejection_state,
)
from list_sorting_transformer.shortcut_learning.shortcut_credit_experiment import (
    ShortcutCreditExperimentConfig,
)


def test_corrected_rejection_preserves_plateau_but_reduces_search_sigma() -> None:
    config = ShortcutCreditExperimentConfig(
        sigma=0.2,
        elite_min_sigma=0.01,
        elite_rejection_sigma_decay=0.5,
    )
    center = {
        "horizon": 100,
        "plateau_state": {
            "ema_fitness": -0.5,
            "best_ema_fitness": -0.4,
            "stale_generations": 2,
            "search_sigma": 0.08,
            "consecutive_accepted_updates": 2,
        },
    }
    run = {
        "horizon": 200,
        "plateau_state": {
            "ema_fitness": -0.3,
            "best_ema_fitness": -0.3,
            "stale_generations": 0,
            "search_sigma": 0.08,
            "consecutive_accepted_updates": 1,
        },
    }

    corrected = corrected_rejection_state(
        center_checkpoint=center,
        run_checkpoint=run,
        config=config,
    )

    assert corrected.ema_fitness == -0.3
    assert corrected.best_ema_fitness == -0.3
    assert corrected.stale_generations == 0
    assert corrected.search_sigma == 0.04
    assert corrected.consecutive_accepted_updates == 0
