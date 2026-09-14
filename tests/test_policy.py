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
