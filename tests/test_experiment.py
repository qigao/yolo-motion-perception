import numpy as np
import pytest

from neural_state_machine import (
    RecurrentController,
    ToyGame,
    context_separation,
    replay_equal,
    run_episode,
)


def test_episode_is_bounded_and_owns_recorded_neural_arrays() -> None:
    trace = run_episode(RecurrentController(seed=13), ToyGame(), max_steps=4)

    assert 1 <= len(trace.decisions) <= 4
    assert len(trace.decisions) == len(trace.rewards) == len(trace.snapshots)
    assert trace.total_reward == pytest.approx(sum(trace.rewards))
    with pytest.raises(ValueError):
        trace.decisions[0].hidden_state[0] = 12.0


def test_same_seed_closed_loops_replay_exactly() -> None:
    left = run_episode(RecurrentController(seed=17), ToyGame(), max_steps=8)
    right = run_episode(RecurrentController(seed=17), ToyGame(), max_steps=8)

    assert replay_equal(left, right)


def test_different_seed_closed_loops_do_not_replay_exactly() -> None:
    left = run_episode(RecurrentController(seed=17), ToyGame(), max_steps=8)
    right = run_episode(RecurrentController(seed=18), ToyGame(), max_steps=8)

    assert not replay_equal(left, right)


def test_context_separation_matches_hand_calculated_fixture() -> None:
    left = np.array([[0.0, 0.0], [0.0, 2.0]])
    right = np.array([[4.0, 0.0], [4.0, 2.0]])

    assert context_separation(left, right) == pytest.approx(4.0)


def test_context_separation_is_finite_for_zero_spread() -> None:
    left = np.array([[0.0, 0.0], [0.0, 0.0]])
    right = np.array([[3.0, 4.0], [3.0, 4.0]])

    assert context_separation(left, right) == pytest.approx(5.0)
    assert context_separation(left, left) == 0.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array([]), np.ones((1, 2))),
        (np.ones((2, 2, 1)), np.ones((2, 2))),
        (np.ones((2, 2)), np.ones((2, 3))),
        (np.array([[float("nan"), 0.0]]), np.ones((1, 2))),
    ],
)
def test_context_separation_rejects_invalid_state_matrices(
    left: np.ndarray, right: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        context_separation(left, right)


@pytest.mark.parametrize("max_steps", [0, -1, 1.5])
def test_episode_rejects_invalid_step_bound(max_steps: float) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_episode(
            RecurrentController(seed=1),
            ToyGame(),
            max_steps=max_steps,  # type: ignore[arg-type]
        )
