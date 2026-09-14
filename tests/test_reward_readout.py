from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neural_state_machine.reward_readout import (
    RewardModulatedReadout,
    RewardReadoutDecision,
)


def test_reward_readout_starts_with_independent_zero_parameters() -> None:
    left = RewardModulatedReadout(hidden_size=4, action_count=3)
    right = RewardModulatedReadout(hidden_size=4, action_count=3)

    assert left.hidden_size == 4
    assert left.action_count == 3
    assert left.learning_rate == 0.05
    assert left.temperature == 1.0
    assert left._weights.shape == (3, 4)
    assert left._biases.shape == (3,)
    assert left._weights.dtype == np.float64
    assert left._biases.dtype == np.float64
    np.testing.assert_array_equal(left._weights, np.zeros((3, 4)))
    np.testing.assert_array_equal(left._biases, np.zeros(3))
    assert left._pending_weight_eligibility is None
    assert left._pending_bias_eligibility is None

    left._weights[0, 0] = 1.0
    left._biases[0] = 1.0
    np.testing.assert_array_equal(right._weights, np.zeros((3, 4)))
    np.testing.assert_array_equal(right._biases, np.zeros(3))


def test_reward_decision_is_frozen() -> None:
    decision = RewardReadoutDecision(
        action_index=0,
        logits=np.zeros(2),
        probabilities=np.full(2, 0.5),
    )

    with pytest.raises(FrozenInstanceError):
        decision.action_index = 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("hidden_size", 0),
        ("hidden_size", -1),
        ("hidden_size", True),
        ("hidden_size", 2.5),
        ("hidden_size", None),
        ("action_count", 0),
        ("action_count", 1),
        ("action_count", True),
        ("action_count", 2.5),
        ("action_count", None),
    ],
)
def test_reward_readout_rejects_invalid_sizes(name: str, value: object) -> None:
    kwargs = {
        "hidden_size": 4,
        "action_count": 2,
        "learning_rate": 0.05,
        "temperature": 1.0,
    }
    kwargs[name] = value

    with pytest.raises(ValueError):
        RewardModulatedReadout(**kwargs)


@pytest.mark.parametrize(
    "learning_rate",
    [0.0, -0.1, np.nan, np.inf, -np.inf, True, "0.1", None],
)
def test_reward_readout_rejects_invalid_learning_rate(learning_rate: object) -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        RewardModulatedReadout(
            hidden_size=4,
            action_count=2,
            learning_rate=learning_rate,
        )


@pytest.mark.parametrize(
    "temperature",
    [0.0, -0.1, np.nan, np.inf, -np.inf, True, "1.0", None],
)
def test_reward_readout_rejects_invalid_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        RewardModulatedReadout(
            hidden_size=4,
            action_count=2,
            temperature=temperature,
        )
