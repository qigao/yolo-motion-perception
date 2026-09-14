import numpy as np
import pytest

from neural_state_machine import Action, GameObservation, RecurrentController


def observation(
    *,
    enemy_distance: float = 0.4,
    enemy_direction: int = 1,
    health: float = 0.8,
    incoming_threat: float = 0.2,
) -> GameObservation:
    return GameObservation(
        enemy_distance=enemy_distance,
        enemy_direction=enemy_direction,
        health=health,
        incoming_threat=incoming_threat,
        healing_distance=0.75,
        healing_direction=-1,
        left_blocked=False,
        right_blocked=False,
    )


def action_margin(logits: np.ndarray, action: Action) -> float:
    selected = int(action)
    alternatives = np.delete(logits, selected)
    return float(logits[selected] - alternatives.max())


def test_same_seed_and_inputs_produce_identical_neural_trajectory() -> None:
    inputs = [
        observation(),
        observation(enemy_distance=0.25, incoming_threat=0.6),
        observation(enemy_distance=0.1, health=0.5, incoming_threat=1.0),
    ]
    left = RecurrentController(hidden_size=12, seed=41, learning_rate=0.2)
    right = RecurrentController(hidden_size=12, seed=41, learning_rate=0.2)

    for stimulus in inputs:
        left_decision = left.step(stimulus)
        right_decision = right.step(stimulus)
        assert left_decision.action is right_decision.action
        assert np.array_equal(left_decision.logits, right_decision.logits)
        assert np.array_equal(left_decision.hidden_state, right_decision.hidden_state)


def test_decision_arrays_cannot_mutate_controller_outputs() -> None:
    decision = RecurrentController(seed=3).step(observation())

    with pytest.raises(ValueError):
        decision.logits[0] = 99.0
    with pytest.raises(ValueError):
        decision.hidden_state[0] = 99.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"hidden_size": -2}, "hidden_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -0.1}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
    ],
)
def test_controller_rejects_invalid_configuration(
    arguments: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RecurrentController(**arguments)


def test_positive_reward_increases_selected_action_margin_for_same_context() -> None:
    controller = RecurrentController(hidden_size=10, seed=9, learning_rate=0.25)
    stimulus = observation(enemy_distance=0.1, incoming_threat=0.9)

    before = controller.step(stimulus)
    margin_before = action_margin(before.logits, before.action)
    controller.learn(1.0)
    controller.reset_state()
    after = controller.step(stimulus)

    assert action_margin(after.logits, before.action) > margin_before


def test_negative_reward_decreases_selected_action_margin_for_same_context() -> None:
    controller = RecurrentController(hidden_size=10, seed=9, learning_rate=0.25)
    stimulus = observation(enemy_distance=0.1, incoming_threat=0.9)

    before = controller.step(stimulus)
    margin_before = action_margin(before.logits, before.action)
    controller.learn(-1.0)
    controller.reset_state()
    after = controller.step(stimulus)

    assert action_margin(after.logits, before.action) < margin_before


def test_reward_is_clipped_to_unit_magnitude() -> None:
    unit = RecurrentController(seed=5)
    large = RecurrentController(seed=5)
    stimulus = observation()

    unit.step(stimulus)
    large.step(stimulus)
    unit.learn(1.0)
    large.learn(100.0)
    unit.reset_state()
    large.reset_state()

    assert np.array_equal(unit.step(stimulus).logits, large.step(stimulus).logits)


def test_learning_requires_a_finite_reward_and_preceding_decision() -> None:
    controller = RecurrentController(seed=2)

    with pytest.raises(RuntimeError, match="decision"):
        controller.learn(0.5)

    controller.step(observation())
    with pytest.raises(ValueError, match="finite"):
        controller.learn(float("nan"))
