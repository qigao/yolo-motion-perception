import hashlib

import numpy as np
import pytest

from neural_state_machine.policy import PolicyDecision, RecurrentPolicy


def test_policy_is_constructible_and_exports_decision_type() -> None:
    policy = RecurrentPolicy(input_size=4, action_count=2, hidden_size=8, seed=7)

    assert isinstance(policy, RecurrentPolicy)
    assert PolicyDecision.__name__ == "PolicyDecision"


def test_policy_types_are_exported_from_the_package() -> None:
    from neural_state_machine import PolicyDecision as exported_decision
    from neural_state_machine import RecurrentPolicy as exported_policy

    assert exported_decision is PolicyDecision
    assert exported_policy is RecurrentPolicy


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("input_size", 0),
        ("input_size", True),
        ("action_count", 1),
        ("action_count", 2.5),
        ("hidden_size", -1),
        ("hidden_size", False),
    ],
)
def test_policy_rejects_invalid_sizes(name: str, value: object) -> None:
    kwargs = dict(input_size=4, action_count=2, hidden_size=8, seed=7)  # noqa: C408
    kwargs[name] = value
    with pytest.raises(ValueError):
        RecurrentPolicy(**kwargs)


@pytest.mark.parametrize("radius", [-0.01, 1.0, np.inf, np.nan])
def test_policy_rejects_invalid_recurrent_radius(radius: float) -> None:
    with pytest.raises(ValueError):
        RecurrentPolicy(4, 2, recurrent_radius=radius)


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_policy_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        RecurrentPolicy(4, 2, seed=seed)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, np.inf, np.nan])
def test_policy_rejects_invalid_learning_rate(learning_rate: float) -> None:
    with pytest.raises(ValueError):
        RecurrentPolicy(4, 2, learning_rate=learning_rate)


@pytest.mark.parametrize(
    "stimulus",
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
def test_advance_rejects_invalid_stimuli_before_evolving(stimulus: object) -> None:
    policy = RecurrentPolicy(4, 2, seed=7)

    with pytest.raises(ValueError):
        policy.advance(stimulus)


@pytest.mark.parametrize(
    "legal_action_indices",
    [(), (0, 0), (True,), (0.0,), (-1,), (3,)],
)
def test_decide_rejects_invalid_legal_masks_before_evolving(
    legal_action_indices: tuple[object, ...],
) -> None:
    stimulus = np.array([0.1, -0.2, 0.3, -0.4])
    policy = RecurrentPolicy(4, 3, seed=11)
    reference = RecurrentPolicy(4, 3, seed=11)

    with pytest.raises(ValueError):
        policy.decide(stimulus, legal_action_indices)

    assert np.array_equal(policy.advance(stimulus), reference.advance(stimulus))


def test_decide_breaks_legal_action_ties_by_lowest_numeric_index() -> None:
    policy = RecurrentPolicy(4, 3, seed=5)
    policy._output_weights.fill(0.0)

    decision = policy.decide(np.zeros(4), (1, 0))

    assert decision.action_index == 0


def test_decide_masks_illegal_logits_and_never_selects_an_illegal_action() -> None:
    decision = RecurrentPolicy(4, 4, seed=13).decide(np.ones(4), (1, 3))

    assert np.isneginf(decision.logits[0])
    assert np.isneginf(decision.logits[2])
    assert decision.action_index in (1, 3)


def test_advance_returns_a_readonly_snapshot_independent_of_future_state() -> None:
    policy = RecurrentPolicy(4, 2, seed=17)
    stimulus = np.array([0.1, 0.2, 0.3, 0.4])

    hidden_state = policy.advance(stimulus)
    expected = hidden_state.copy()
    stimulus[:] = 99.0
    policy.advance(np.zeros(4))

    assert np.array_equal(hidden_state, expected)
    with pytest.raises(ValueError):
        hidden_state[0] = 1.0


def test_decide_returns_readonly_snapshots_independent_of_future_state() -> None:
    policy = RecurrentPolicy(4, 3, seed=19)
    stimulus = np.array([-0.1, 0.2, -0.3, 0.4])

    decision = policy.decide(stimulus, (0, 2))
    expected_logits = decision.logits.copy()
    expected_hidden_state = decision.hidden_state.copy()
    stimulus[:] = 99.0
    policy.decide(np.zeros(4), (0, 2))

    assert np.array_equal(decision.logits, expected_logits)
    assert np.array_equal(decision.hidden_state, expected_hidden_state)
    with pytest.raises(ValueError):
        decision.logits[0] = 1.0
    with pytest.raises(ValueError):
        decision.hidden_state[0] = 1.0


def test_reset_state_restores_the_zero_hidden_state() -> None:
    policy = RecurrentPolicy(4, 2, seed=23)
    stimulus = np.array([0.2, -0.4, 0.6, -0.8])

    policy.advance(stimulus)
    policy.reset_state()

    assert np.array_equal(policy.advance(np.zeros(4)), np.zeros(policy.hidden_size))


def test_output_weight_digest_identifies_c_contiguous_float64_output_weights() -> None:
    policy = RecurrentPolicy(4, 3, hidden_size=5, seed=29)
    values = np.ascontiguousarray(policy._output_weights, dtype=np.float64)
    expected = hashlib.sha256(
        str(values.shape).encode("ascii") + values.tobytes(order="C")
    ).hexdigest()

    assert policy.output_weight_digest() == expected


@pytest.mark.parametrize(
    "explore_probability",
    [-0.01, 1.01, np.nan, np.inf, -np.inf],
)
def test_decide_rejects_invalid_exploration_probability(
    explore_probability: float,
) -> None:
    policy = RecurrentPolicy(4, 2, seed=31)

    with pytest.raises(ValueError, match="explore_probability"):
        policy.decide(
            np.zeros(4),
            (0, 1),
            explore_probability=explore_probability,
            rng=np.random.default_rng(31),
        )


def test_decide_requires_rng_for_positive_exploration_probability() -> None:
    policy = RecurrentPolicy(4, 2, seed=37)

    with pytest.raises(ValueError, match="rng"):
        policy.decide(np.zeros(4), (0, 1), explore_probability=0.25)


def test_decide_rejects_non_generator_rng() -> None:
    policy = RecurrentPolicy(4, 2, seed=41)

    with pytest.raises(ValueError, match="rng"):
        policy.decide(
            np.zeros(4),
            (0, 1),
            explore_probability=0.25,
            rng=np.random.RandomState(41),
        )


def test_exploration_is_deterministic_and_respects_legal_indices() -> None:
    left = RecurrentPolicy(4, 3, seed=43)
    right = RecurrentPolicy(4, 3, seed=43)
    left_rng = np.random.default_rng(73)
    right_rng = np.random.default_rng(73)
    stimulus = np.array([0.1, -0.2, 0.3, -0.4])

    left_actions = [
        left.decide(
            stimulus,
            (0, 2),
            explore_probability=1.0,
            rng=left_rng,
        ).action_index
        for _ in range(50)
    ]
    right_actions = [
        right.decide(
            stimulus,
            (0, 2),
            explore_probability=1.0,
            rng=right_rng,
        ).action_index
        for _ in range(50)
    ]

    assert left_actions == right_actions
    assert set(left_actions) <= {0, 2}
    assert set(left_actions) == {0, 2}


def test_learning_requires_one_unconsumed_decision() -> None:
    policy = RecurrentPolicy(4, 2, seed=7)
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)

    policy.decide(np.zeros(4), (0, 1))
    policy.learn(1.0)
    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)


def test_advance_does_not_create_eligibility() -> None:
    policy = RecurrentPolicy(4, 2, seed=47)

    policy.advance(np.ones(4))

    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)


def test_reset_state_consumes_pending_eligibility() -> None:
    policy = RecurrentPolicy(4, 2, seed=53)

    policy.decide(np.ones(4), (0, 1))
    policy.reset_state()

    with pytest.raises(RuntimeError, match="preceding decision"):
        policy.learn(1.0)


def test_nonfinite_reward_does_not_consume_valid_eligibility() -> None:
    policy = RecurrentPolicy(4, 2, seed=59)
    before = policy.output_weight_digest()
    policy.decide(np.array([0.2, -0.4, 0.6, -0.8]), (0, 1))

    with pytest.raises(ValueError, match="finite"):
        policy.learn(np.nan)

    policy.learn(1.0)
    assert policy.output_weight_digest() != before


def _learned_margin(reward: float) -> tuple[str, float]:
    policy = RecurrentPolicy(4, 2, hidden_size=10, seed=61, learning_rate=0.25)
    stimulus = np.array([0.1, -0.2, 0.3, -0.4])
    before = policy.decide(stimulus, (0, 1))
    selected = before.action_index
    policy.learn(reward)
    policy.reset_state()
    after = policy.decide(stimulus, (0, 1))
    alternatives = np.delete(after.logits, selected)
    margin = float(after.logits[selected] - alternatives.max())
    return policy.output_weight_digest(), margin


def test_positive_reward_increases_selected_action_margin() -> None:
    policy = RecurrentPolicy(4, 2, hidden_size=10, seed=61, learning_rate=0.25)
    stimulus = np.array([0.1, -0.2, 0.3, -0.4])
    before = policy.decide(stimulus, (0, 1))
    margin_before = float(before.logits[before.action_index] - np.delete(before.logits, before.action_index).max())
    selected = before.action_index
    policy.learn(1.0)
    policy.reset_state()
    after = policy.decide(stimulus, (0, 1))

    assert float(after.logits[selected] - np.delete(after.logits, selected).max()) > margin_before


def test_negative_reward_decreases_selected_action_margin() -> None:
    policy = RecurrentPolicy(4, 2, hidden_size=10, seed=61, learning_rate=0.25)
    stimulus = np.array([0.1, -0.2, 0.3, -0.4])
    before = policy.decide(stimulus, (0, 1))
    margin_before = float(before.logits[before.action_index] - np.delete(before.logits, before.action_index).max())
    selected = before.action_index
    policy.learn(-1.0)
    policy.reset_state()
    after = policy.decide(stimulus, (0, 1))

    assert float(after.logits[selected] - np.delete(after.logits, selected).max()) < margin_before


@pytest.mark.parametrize("reward_pair", [(1.0, 100.0), (-1.0, -100.0)])
def test_reward_is_clipped_to_unit_magnitude(
    reward_pair: tuple[float, float],
) -> None:
    unit_digest, unit_margin = _learned_margin(reward_pair[0])
    large_digest, large_margin = _learned_margin(reward_pair[1])

    assert large_digest == unit_digest
    assert large_margin == unit_margin


def test_learning_mutates_only_selected_output_row() -> None:
    policy = RecurrentPolicy(4, 3, seed=67)
    stimulus = np.array([0.2, -0.4, 0.6, -0.8])
    input_before = policy._input_weights.copy()
    recurrent_before = policy._recurrent_weights.copy()
    output_before = policy._output_weights.copy()
    decision = policy.decide(stimulus, (0, 2))

    policy.learn(0.5)

    assert np.array_equal(policy._input_weights, input_before)
    assert np.array_equal(policy._recurrent_weights, recurrent_before)
    for index in range(policy.action_count):
        if index == decision.action_index:
            assert not np.array_equal(policy._output_weights[index], output_before[index])
        else:
            assert np.array_equal(policy._output_weights[index], output_before[index])


def test_output_digest_changes_only_for_nonzero_learning() -> None:
    policy = RecurrentPolicy(4, 2, seed=71)
    digest = policy.output_weight_digest()

    policy.advance(np.ones(4))
    assert policy.output_weight_digest() == digest
    policy.decide(np.ones(4), (0, 1))
    assert policy.output_weight_digest() == digest
    policy.reset_state()
    assert policy.output_weight_digest() == digest

    policy.decide(np.ones(4), (0, 1))
    policy.learn(0.5)
    assert policy.output_weight_digest() != digest
