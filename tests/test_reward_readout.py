import hashlib
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


@pytest.mark.parametrize(
    "rng",
    [None, 7, np.random.RandomState(7), object()],
)
def test_training_selection_rejects_non_generator_without_pending_state(
    rng: object,
) -> None:
    readout = RewardModulatedReadout(4, 2)

    with pytest.raises(ValueError, match="rng"):
        readout.select_for_training(np.zeros(4), (0, 1), rng)

    assert readout.has_pending_feedback is False
    assert readout._pending_weight_eligibility is None
    assert readout._pending_bias_eligibility is None


def test_same_seed_produces_the_same_single_training_selection() -> None:
    left = RewardModulatedReadout(4, 2)
    right = RewardModulatedReadout(4, 2)
    hidden = np.array([0.2, -0.4, 0.6, -0.8])

    left_decision = left.select_for_training(
        hidden,
        (0, 1),
        np.random.default_rng(31),
    )
    right_decision = right.select_for_training(
        hidden,
        (0, 1),
        np.random.default_rng(31),
    )

    assert left_decision.action_index == right_decision.action_index
    assert left_decision.action_index in (0, 1)
    np.testing.assert_array_equal(left_decision.logits, right_decision.logits)
    np.testing.assert_array_equal(
        left_decision.probabilities,
        right_decision.probabilities,
    )
    assert left.has_pending_feedback is True
    assert right.has_pending_feedback is True


def test_training_selection_never_samples_a_masked_action() -> None:
    readout = RewardModulatedReadout(4, 3)

    decision = readout.select_for_training(
        np.ones(4),
        (2, 0),
        np.random.default_rng(37),
    )

    assert decision.action_index in (0, 2)
    assert decision.probabilities[1] == 0.0
    assert np.isneginf(decision.logits[1])


def test_pending_feedback_is_one_shot_and_greedy_does_not_change_it() -> None:
    readout = RewardModulatedReadout(3, 2)
    hidden = np.array([0.25, -0.5, 0.75])

    assert readout.has_pending_feedback is False
    readout.select_for_training(hidden, (0, 1), np.random.default_rng(41))
    assert readout.has_pending_feedback is True
    weight_eligibility = readout._pending_weight_eligibility.copy()
    bias_eligibility = readout._pending_bias_eligibility.copy()

    with pytest.raises(RuntimeError, match="pending"):
        readout.select_for_training(hidden, (0, 1), np.random.default_rng(43))

    np.testing.assert_array_equal(
        readout._pending_weight_eligibility,
        weight_eligibility,
    )
    np.testing.assert_array_equal(
        readout._pending_bias_eligibility,
        bias_eligibility,
    )
    readout.select_greedy(hidden, (0, 1))
    assert readout.has_pending_feedback is True
    np.testing.assert_array_equal(
        readout._pending_weight_eligibility,
        weight_eligibility,
    )
    np.testing.assert_array_equal(
        readout._pending_bias_eligibility,
        bias_eligibility,
    )

    evaluation_only = RewardModulatedReadout(3, 2)
    evaluation_only.select_greedy(hidden, (0, 1))
    assert evaluation_only.has_pending_feedback is False


def test_training_eligibility_does_not_alias_the_hidden_input() -> None:
    readout = RewardModulatedReadout(3, 2)
    hidden = np.array([0.2, -0.4, 0.6])
    hidden_before = hidden.copy()

    decision = readout.select_for_training(
        hidden,
        (0, 1),
        np.random.default_rng(47),
    )
    one_hot = np.zeros(2)
    one_hot[decision.action_index] = 1.0
    delta = one_hot - decision.probabilities
    expected_weights = np.outer(delta, hidden_before)
    expected_biases = delta.copy()
    hidden[:] = 99.0

    np.testing.assert_array_equal(
        readout._pending_weight_eligibility,
        expected_weights,
    )
    np.testing.assert_array_equal(
        readout._pending_bias_eligibility,
        expected_biases,
    )


def test_learning_requires_one_pending_training_decision() -> None:
    readout = RewardModulatedReadout(3, 2)

    with pytest.raises(RuntimeError, match="pending"):
        readout.learn(1.0)

    readout.select_for_training(
        np.ones(3),
        (0, 1),
        np.random.default_rng(53),
    )
    readout.learn(1.0)
    assert readout.has_pending_feedback is False
    with pytest.raises(RuntimeError, match="pending"):
        readout.learn(1.0)


def test_learning_applies_the_literal_softmax_eligibility_update() -> None:
    readout = RewardModulatedReadout(
        hidden_size=3,
        action_count=2,
        learning_rate=0.2,
    )
    hidden = np.array([0.25, -0.5, 0.75])
    decision = readout.select_for_training(
        hidden,
        (0, 1),
        np.random.default_rng(59),
    )
    one_hot = np.zeros(2)
    one_hot[decision.action_index] = 1.0
    delta = one_hot - decision.probabilities

    readout.learn(0.5)

    np.testing.assert_allclose(
        readout._weights,
        0.2 * 0.5 * np.outer(delta, hidden),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        readout._biases,
        0.2 * 0.5 * delta,
        rtol=0.0,
        atol=1e-12,
    )
    assert readout.has_pending_feedback is False


def _selected_margin_after_reward(reward: float) -> tuple[str, float]:
    readout = RewardModulatedReadout(3, 2, learning_rate=0.25)
    hidden = np.array([0.2, -0.4, 0.6])
    decision = readout.select_for_training(
        hidden,
        (0, 1),
        np.random.default_rng(61),
    )
    selected = decision.action_index
    readout.learn(reward)
    replay = readout.select_greedy(hidden, (0, 1))
    alternative = 1 - selected
    margin = float(replay.logits[selected] - replay.logits[alternative])
    return readout.parameter_digest(), margin


def test_positive_reward_increases_the_sampled_action_margin() -> None:
    _, margin = _selected_margin_after_reward(1.0)
    assert margin > 0.0


def test_negative_reward_decreases_the_sampled_action_margin() -> None:
    _, margin = _selected_margin_after_reward(-1.0)
    assert margin < 0.0


@pytest.mark.parametrize("rewards", [(1.0, 100.0), (-1.0, -100.0)])
def test_reward_is_clipped_to_unit_magnitude(
    rewards: tuple[float, float],
) -> None:
    unit_digest, unit_margin = _selected_margin_after_reward(rewards[0])
    large_digest, large_margin = _selected_margin_after_reward(rewards[1])

    assert large_digest == unit_digest
    assert large_margin == unit_margin


def test_zero_reward_consumes_eligibility_without_changing_parameters() -> None:
    readout = RewardModulatedReadout(3, 2)
    before = readout.parameter_digest()
    readout.select_for_training(
        np.ones(3),
        (0, 1),
        np.random.default_rng(67),
    )

    readout.learn(0.0)

    assert readout.parameter_digest() == before
    assert readout.has_pending_feedback is False


@pytest.mark.parametrize("reward", ["bad", None, np.nan, np.inf, -np.inf])
def test_invalid_reward_preserves_pending_eligibility(reward: object) -> None:
    readout = RewardModulatedReadout(3, 2)
    readout.select_for_training(
        np.ones(3),
        (0, 1),
        np.random.default_rng(71),
    )
    weights = readout._pending_weight_eligibility.copy()
    biases = readout._pending_bias_eligibility.copy()

    with pytest.raises(ValueError, match="finite"):
        readout.learn(reward)

    assert readout.has_pending_feedback is True
    np.testing.assert_array_equal(
        readout._pending_weight_eligibility,
        weights,
    )
    np.testing.assert_array_equal(
        readout._pending_bias_eligibility,
        biases,
    )
    readout.learn(1.0)
    assert readout.has_pending_feedback is False


def test_training_sequence_is_repeatable_when_feedback_is_consumed() -> None:
    left = RewardModulatedReadout(3, 2)
    right = RewardModulatedReadout(3, 2)
    left_rng = np.random.default_rng(73)
    right_rng = np.random.default_rng(73)
    hidden = np.array([0.1, -0.2, 0.3])
    left_actions = []
    right_actions = []

    for _ in range(50):
        left_actions.append(
            left.select_for_training(hidden, (0, 1), left_rng).action_index
        )
        right_actions.append(
            right.select_for_training(hidden, (0, 1), right_rng).action_index
        )
        left.learn(0.0)
        right.learn(0.0)

    assert left_actions == right_actions
    assert set(left_actions) == {0, 1}
    assert left.has_pending_feedback is False
    assert right.has_pending_feedback is False


def test_parameter_digest_covers_weight_and_bias_float64_bytes() -> None:
    readout = RewardModulatedReadout(3, 2)
    readout._weights[:] = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
    readout._biases[:] = [0.25, -0.5]
    weights = np.ascontiguousarray(readout._weights, dtype=np.float64)
    biases = np.ascontiguousarray(readout._biases, dtype=np.float64)
    expected = hashlib.sha256()
    expected.update(str(weights.shape).encode("ascii"))
    expected.update(weights.tobytes(order="C"))
    expected.update(str(biases.shape).encode("ascii"))
    expected.update(biases.tobytes(order="C"))

    assert readout.parameter_digest() == expected.hexdigest()


def test_negative_reward_updates_legal_rows_but_not_masked_rows() -> None:
    readout = RewardModulatedReadout(2, 3, learning_rate=0.1)
    hidden = np.array([0.5, -0.25])
    before = readout._weights.copy()
    readout.select_for_training(
        hidden,
        (0, 2),
        np.random.default_rng(79),
    )

    readout.learn(-1.0)

    assert not np.array_equal(readout._weights[0], before[0])
    np.testing.assert_array_equal(readout._weights[1], before[1])
    assert not np.array_equal(readout._weights[2], before[2])
    np.testing.assert_allclose(
        readout._weights[0],
        -readout._weights[2],
        rtol=0.0,
        atol=1e-12,
    )
    assert readout._biases[0] == pytest.approx(
        -readout._biases[2],
        rel=0.0,
        abs=1e-12,
    )


def test_reward_readout_types_are_exported_from_the_package() -> None:
    from neural_state_machine import (
        RewardModulatedReadout as exported_readout,
    )
    from neural_state_machine import (
        RewardReadoutDecision as exported_decision,
    )

    assert exported_readout is RewardModulatedReadout
    assert exported_decision is RewardReadoutDecision
