import numpy as np
import pytest

from neural_state_machine import GameObservation, encode_observation


def make_observation(**overrides: object) -> GameObservation:
    values: dict[str, object] = {
        "enemy_distance": 0.25,
        "enemy_direction": -1,
        "health": 0.5,
        "incoming_threat": 0.75,
        "healing_distance": 1.0,
        "healing_direction": 0,
        "left_blocked": True,
        "right_blocked": False,
    }
    values.update(overrides)
    return GameObservation(**values)  # type: ignore[arg-type]


def test_encode_observation_uses_stable_literal_order() -> None:
    encoded = encode_observation(make_observation())

    np.testing.assert_array_equal(
        encoded,
        np.array([0.25, -1.0, 0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 1.0]),
    )
    assert encoded.dtype == np.float64
    assert not encoded.flags.writeable


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enemy_distance", -0.01),
        ("enemy_distance", 1.01),
        ("enemy_distance", float("nan")),
        ("health", float("inf")),
        ("incoming_threat", -1.0),
        ("healing_distance", 2.0),
        ("enemy_direction", 0.5),
        ("enemy_direction", 2),
        ("healing_direction", -2),
    ],
)
def test_observation_rejects_invalid_numeric_stimuli(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        make_observation(**{field: value})


@pytest.mark.parametrize("field", ["left_blocked", "right_blocked"])
def test_observation_rejects_non_boolean_boundary_stimulus(field: str) -> None:
    with pytest.raises(ValueError):
        make_observation(**{field: 1})
