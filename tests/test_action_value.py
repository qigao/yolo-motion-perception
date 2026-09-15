import numpy as np
import pytest

from neural_state_machine.action_value import ActionValueDecision, NormalizedActionValue


def test_action_value_starts_at_exact_zero() -> None:
    learner = NormalizedActionValue(hidden_size=3, action_count=2, step_size=0.1)
    snapshot = learner.parameter_snapshot()
    assert snapshot.dtype == np.float64
    assert snapshot.shape == (2, 4)
    assert np.array_equal(snapshot, np.zeros((2, 4), dtype=np.float64))
    assert snapshot.flags.writeable is False
    assert learner.has_pending_feedback is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": True}, "hidden_size"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"action_count": True}, "action_count"),
        ({"action_count": 1}, "action_count"),
        ({"step_size": True}, "step_size"),
        ({"step_size": 0.0}, "step_size"),
        ({"step_size": 1.0000001}, "step_size"),
        ({"step_size": float("nan")}, "step_size"),
    ],
)
def test_constructor_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    values = {"hidden_size": 3, "action_count": 2, "step_size": 0.1}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        NormalizedActionValue(**values)


def test_parameter_digest_is_canonical_and_snapshot_is_independent() -> None:
    import hashlib

    learner = NormalizedActionValue(hidden_size=2, action_count=3)
    snapshot = learner.parameter_snapshot()
    expected = hashlib.sha256(
        str(snapshot.shape).encode("ascii") + snapshot.tobytes(order="C")
    ).hexdigest()

    assert learner.parameter_digest() == expected
    with pytest.raises(ValueError):
        snapshot[0, 0] = 1.0

    learner._weights[0, 0] = 2.0
    assert snapshot[0, 0] == 0.0
    assert learner.parameter_digest() != expected


def test_selection_accepts_float64_compatible_hidden_state_and_bias_feature() -> None:
    learner = NormalizedActionValue(2, 2)
    learner._weights[:] = np.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]], dtype=np.float64
    )

    decision = learner.select_greedy(np.array([1, -3], dtype=np.float32), (0, 1))

    assert decision.action_index == 0
    assert np.array_equal(decision.action_values, np.array([2.0, -1.0]))


@pytest.mark.parametrize(
    "hidden_state",
    [
        1.0,
        np.zeros((1, 2)),
        np.zeros(3),
        [0.0, "bad"],
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, -np.inf]),
    ],
)
def test_selection_rejects_invalid_hidden_state(hidden_state: object) -> None:
    learner = NormalizedActionValue(hidden_size=2, action_count=3)

    with pytest.raises(ValueError, match="hidden_state"):
        learner.select_greedy(hidden_state, (0, 1))


@pytest.mark.parametrize(
    "legal_action_indices",
    [
        (),
        (0, 0),
        (True,),
        (0.0,),
        (-1,),
        (3,),
        0,
    ],
)
def test_selection_rejects_invalid_legal_action_indices(
    legal_action_indices: object,
) -> None:
    learner = NormalizedActionValue(hidden_size=2, action_count=3)

    with pytest.raises(ValueError, match="legal_action_indices"):
        learner.select_greedy(np.zeros(2), legal_action_indices)


def test_greedy_selection_masks_illegal_actions_and_breaks_ties_lowest() -> None:
    learner = NormalizedActionValue(2, 3)
    learner._weights[:] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    decision = learner.select_greedy(np.array([2.0, 1.0]), (2, 0))

    assert isinstance(decision, ActionValueDecision)
    assert decision.action_index == 0
    assert decision.action_values.dtype == np.float64
    assert decision.action_values.shape == (3,)
    assert np.isneginf(decision.action_values[1])
    assert decision.action_values.flags.writeable is False
    assert learner.has_pending_feedback is False


def test_greedy_decision_is_readonly_and_does_not_mutate_learner() -> None:
    learner = NormalizedActionValue(2, 3)
    before = learner.parameter_snapshot()
    decision = learner.select_greedy(np.array([0.25, -0.5]), (0, 2))

    with pytest.raises(ValueError):
        decision.action_values[0] = 99.0
    assert np.array_equal(learner.parameter_snapshot(), before)


def _clear_pending_for_test(learner: NormalizedActionValue) -> None:
    learner._pending = None


def test_training_selection_is_uniform_over_legal_actions_and_seeded() -> None:
    left = NormalizedActionValue(2, 4)
    right = NormalizedActionValue(2, 4)
    left._weights[:] = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [99.0, 99.0, 99.0]],
        dtype=np.float64,
    )
    right._weights[:] = np.array(
        [[-10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, -10.0], [-99.0, -99.0, -99.0]],
        dtype=np.float64,
    )
    left_rng = np.random.default_rng(19)
    right_rng = np.random.default_rng(19)
    hidden = np.array([0.3, -0.7])
    legal = (0, 1, 2)

    left_actions = []
    right_actions = []
    for _ in range(1000):
        left_actions.append(left.select_for_training(hidden, legal, left_rng).action_index)
        right_actions.append(right.select_for_training(hidden, legal, right_rng).action_index)
        _clear_pending_for_test(left)
        _clear_pending_for_test(right)

    assert left_actions == right_actions
    counts = np.bincount(left_actions, minlength=4)
    assert all(280 <= counts[index] <= 390 for index in legal)
    assert counts[3] == 0


def test_training_selection_validates_rng_without_advancing_state() -> None:
    learner = NormalizedActionValue(2, 3)
    expected_rng = np.random.default_rng(29)
    actual_rng = np.random.default_rng(29)

    with pytest.raises(ValueError, match="rng"):
        learner.select_for_training(np.zeros(2), (0, 1), np.random.RandomState(29))

    assert actual_rng.integers(100) == expected_rng.integers(100)


def test_training_rejects_invalid_hidden_before_advancing_valid_rng() -> None:
    """Catches moving rng.integers before _feature validation."""
    learner = NormalizedActionValue(2, 3)
    actual_rng = np.random.default_rng(43)
    expected_rng = np.random.default_rng(43)

    with pytest.raises(ValueError, match="hidden_state"):
        learner.select_for_training(np.zeros(1), (0, 1), actual_rng)

    assert actual_rng.integers(100) == expected_rng.integers(100)
    assert learner.has_pending_feedback is False


def test_training_rejects_invalid_legal_indices_before_advancing_valid_rng() -> None:
    """Catches moving rng.integers before _legal_actions validation."""
    learner = NormalizedActionValue(2, 3)
    actual_rng = np.random.default_rng(47)
    expected_rng = np.random.default_rng(47)

    with pytest.raises(ValueError, match="legal_action_indices"):
        learner.select_for_training(np.zeros(2), (0, 0), actual_rng)

    assert actual_rng.integers(100) == expected_rng.integers(100)
    assert learner.has_pending_feedback is False


def test_training_selection_rejects_second_pending_record_without_rng_draw() -> None:
    learner = NormalizedActionValue(2, 3)
    rng = np.random.default_rng(31)
    expected_rng = np.random.default_rng(31)
    learner.select_for_training(np.zeros(2), (0, 1), rng)
    expected_rng.integers(2)
    pending = learner._pending

    with pytest.raises(RuntimeError, match="pending"):
        learner.select_for_training(np.zeros(2), (0, 1), rng)

    assert learner._pending is pending
    assert rng.integers(100) == expected_rng.integers(100)


def test_training_selection_copies_hidden_state_into_pending_record() -> None:
    learner = NormalizedActionValue(2, 3)
    hidden = np.array([0.2, -0.4])
    learner.select_for_training(hidden, (0, 1), np.random.default_rng(37))
    expected = learner._pending.feature.copy()

    hidden[:] = 99.0

    assert np.array_equal(learner._pending.feature, expected)
    assert learner._pending.feature.flags.writeable is False


def test_greedy_selection_neither_creates_nor_replaces_pending_feedback() -> None:
    learner = NormalizedActionValue(2, 3)
    learner.select_greedy(np.zeros(2), (0, 1))
    assert learner.has_pending_feedback is False

    learner.select_for_training(np.zeros(2), (0, 1), np.random.default_rng(41))
    pending = learner._pending
    learner.select_greedy(np.ones(2), (0, 1))

    assert learner._pending is pending
