import pytest

from neural_state_machine.action_value_benchmark import (
    ActionValueBenchmarkConfig,
    run_action_value_experiment,
)
from neural_state_machine.phase3b_delayed_benchmark import (
    DelayedCreditConfig,
    DelayedCreditResult,
    run_delayed_credit,
    run_delayed_credit_benchmark,
)


def test_delayed_config_has_fixed_axes_and_phase3a_defaults() -> None:
    config = DelayedCreditConfig()
    assert config.reward_delays == (0, 1, 3, 5)
    assert config.action_value_config == ActionValueBenchmarkConfig()


def test_zero_reward_delay_matches_phase3a_continuity_boundary() -> None:
    config = DelayedCreditConfig(
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )
    legacy = run_action_value_experiment(7, config.action_value_config)
    delayed = run_delayed_credit(7, 0, "td0", config)

    assert delayed.post_training == legacy.post_training
    assert delayed.state_reset == legacy.state_reset
    assert delayed.per_delay == legacy.per_delay
    assert delayed.reset_per_delay == legacy.reset_per_delay
    assert delayed.action_digest == legacy.normal_action_digest
    assert delayed.training_fixture_digest == legacy.training_fixture_digest
    assert delayed.evaluation_fixture_digest == legacy.evaluation_fixture_digest
    assert delayed.parameter_digest == legacy.normal_parameter_digest
    assert delayed.queue_deliveries == config.training_episodes
    assert delayed.repeatable is True


@pytest.mark.parametrize("reward_delay", [1, 3, 5])
def test_delayed_td0_delivers_one_reward_per_training_episode(reward_delay: int) -> None:
    config = DelayedCreditConfig(
        training_episodes=100,
        evaluation_blocks=2,
        checkpoint_interval=50,
    )
    result = run_delayed_credit(7, reward_delay, "td0", config)
    assert isinstance(result, DelayedCreditResult)
    assert result.queue_deliveries == 100
    assert result.pending_feedback is False
    assert result.repeatable is True


def test_delayed_benchmark_rejects_invalid_seed_order_and_arm() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        run_delayed_credit_benchmark((7, 7))
    with pytest.raises(ValueError, match="arm"):
        run_delayed_credit(7, 0, "unknown", DelayedCreditConfig())


def test_delayed_config_rejects_invalid_delay_axes() -> None:
    with pytest.raises(ValueError, match="start"):
        DelayedCreditConfig(reward_delays=(1, 3))
    with pytest.raises(ValueError, match="duplicates"):
        DelayedCreditConfig(reward_delays=(0, 1, 1))
