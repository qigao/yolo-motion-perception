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


@pytest.mark.parametrize(
    "hidden_state",
    [
        1.0,
        np.zeros((1, 4)),
        np.zeros(3),
        [0.0, 1.0, "bad", 3.0],
        np.array([0.0, np.nan, 2.0, 3.0]),
        np.array([0.0, np.inf, 2.0, 3.0]),
        np.array([0.0, -np.inf, 2.0, 3.0]),
    ],
)
def test_greedy_rejects_invalid_hidden_state_without_mutation(
    hidden_state: object,
) -> None:
    readout = RewardModulatedReadout(4, 3)
    weights_before = readout._weights.copy()
    biases_before = readout._biases.copy()

    with pytest.raises(ValueError, match="hidden_state"):
        readout.select_greedy(hidden_state, (0, 2))

    np.testing.assert_array_equal(readout._weights, weights_before)
    np.testing.assert_array_equal(readout._biases, biases_before)
    assert readout._pending_weight_eligibility is None
    assert readout._pending_bias_eligibility is None


@pytest.mark.parametrize(
    "legal_action_indices",
    [(), (0, 0), (True,), (0.0,), (-1,), (3,), None],
)
def test_greedy_rejects_invalid_legal_action_indices(
    legal_action_indices: object,
) -> None:
    readout = RewardModulatedReadout(4, 3)

    with pytest.raises(ValueError, match="legal_action_indices"):
        readout.select_greedy(np.zeros(4), legal_action_indices)


def test_zero_readout_masks_illegal_actions_and_breaks_ties_by_index() -> None:
    readout = RewardModulatedReadout(4, 3)

    decision = readout.select_greedy(np.zeros(4), (2, 0))

    assert decision.action_index == 0
    np.testing.assert_array_equal(decision.logits, [0.0, -np.inf, 0.0])
    np.testing.assert_array_equal(decision.probabilities, [0.5, 0.0, 0.5])
    assert readout._pending_weight_eligibility is None
    assert readout._pending_bias_eligibility is None


def test_greedy_matches_independent_temperature_scaled_softmax() -> None:
    readout = RewardModulatedReadout(
        hidden_size=2,
        action_count=3,
        temperature=2.0,
    )
    readout._weights[:] = [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]]
    readout._biases[:] = [0.5, -0.5, 1.0]
    hidden = np.array([2.0, -1.0])

    decision = readout.select_greedy(hidden, (2, 0))

    raw = readout._weights @ hidden + readout._biases
    legal_scaled = raw[[0, 2]] / readout.temperature
    legal_exp = np.exp(legal_scaled - np.max(legal_scaled))
    legal_probabilities = legal_exp / np.sum(legal_exp)
    expected_probabilities = np.array(
        [legal_probabilities[0], 0.0, legal_probabilities[1]]
    )
    np.testing.assert_allclose(
        decision.logits,
        [raw[0], -np.inf, raw[2]],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decision.probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=1e-12,
    )
    assert decision.action_index == 0


def test_softmax_is_stable_for_large_legal_logits() -> None:
    readout = RewardModulatedReadout(4, 3)
    readout._weights[:] = [
        [1_000.0, 1_000.0, 1_000.0, 1_000.0],
        [-1_000.0, -1_000.0, -1_000.0, -1_000.0],
        [500.0, 500.0, 500.0, 500.0],
    ]

    with np.errstate(over="raise", invalid="raise"):
        decision = readout.select_greedy(np.ones(4), (0, 2))

    assert np.all(np.isfinite(decision.probabilities[[0, 2]]))
    assert np.sum(decision.probabilities[[0, 2]]) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-15,
    )
    assert decision.probabilities[1] == 0.0
    assert np.isneginf(decision.logits[1])
    assert decision.action_index == 0


def test_greedy_returns_readonly_snapshots_independent_of_parameters() -> None:
    readout = RewardModulatedReadout(2, 2)
    hidden = np.array([0.25, -0.75])

    decision = readout.select_greedy(hidden, (0, 1))
    logits_before = decision.logits.copy()
    probabilities_before = decision.probabilities.copy()
    hidden[:] = 99.0
    readout._weights.fill(3.0)
    readout._biases.fill(-2.0)
    readout.select_greedy(np.zeros(2), (0, 1))

    np.testing.assert_array_equal(decision.logits, logits_before)
    np.testing.assert_array_equal(decision.probabilities, probabilities_before)
    assert decision.logits.dtype == np.float64
    assert decision.probabilities.dtype == np.float64
    with pytest.raises(ValueError):
        decision.logits[0] = 1.0
    with pytest.raises(ValueError):
        decision.probabilities[0] = 1.0
