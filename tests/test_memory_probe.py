import hashlib

import numpy as np
import pytest

from neural_state_machine.memory_probe import FittedLinearProbe, fit_linear_probe


def test_fitted_probe_defensively_copies_readonly_float64_weights() -> None:
    source = np.array([1, -2], dtype=np.int64)
    probe = FittedLinearProbe(source, 0)
    source[:] = 99

    np.testing.assert_array_equal(probe.weights, [1.0, -2.0])
    assert probe.weights.dtype == np.float64
    assert not probe.weights.flags.writeable
    assert not np.shares_memory(probe.weights, source)


def test_prediction_is_readonly_independent_and_uses_left_on_exact_tie() -> None:
    states = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 8.0]])
    probe = FittedLinearProbe(np.array([1.0, 0.0]), 0.0)
    prediction = probe.predict(states)
    states[:] = 100.0

    np.testing.assert_array_equal(prediction, [1, 0, 0])
    assert prediction.dtype == np.int64
    assert not prediction.flags.writeable


def test_probe_digest_covers_shape_weights_and_float64_bias() -> None:
    probe = FittedLinearProbe(np.array([1.5, -2.0]), 0.25)
    expected = hashlib.sha256()
    expected.update(str((2,)).encode("ascii"))
    expected.update(np.array([1.5, -2.0], dtype=np.float64).tobytes(order="C"))
    expected.update(np.asarray(0.25, dtype=np.float64).tobytes())

    assert probe.digest() == expected.hexdigest()


@pytest.mark.parametrize(
    ("weights", "bias"),
    [
        (1.0, 0.0),
        (np.ones((1, 2)), 0.0),
        (np.array([]), 0.0),
        (np.array([np.nan]), 0.0),
        (np.array([1.0]), np.inf),
    ],
)
def test_fitted_probe_rejects_invalid_parameters(weights: object, bias: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(weights, bias)


@pytest.mark.parametrize(
    "states",
    [1.0, np.ones(2), np.ones((1, 1, 2)), [[np.nan, 0.0]], [[1.0]]],
)
def test_predict_rejects_invalid_state_matrices(states: object) -> None:
    with pytest.raises(ValueError):
        FittedLinearProbe(np.ones(2), 0.0).predict(states)


def test_fit_linear_probe_uses_closed_form_ridge_without_bias_penalty() -> None:
    states = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    probe = fit_linear_probe(states, labels, regularization=1.0)

    np.testing.assert_allclose(probe.weights, [6.0 / 11.0], rtol=0.0, atol=1e-15)
    assert probe.bias == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(probe.predict(states), labels)


@pytest.mark.parametrize(
    ("states", "labels"),
    [
        (np.empty((0, 2)), np.empty(0, dtype=np.int64)),
        (np.empty((2, 0)), np.array([0, 1])),
        (np.ones(2), np.array([0, 1])),
        (np.array([[0.0], [np.inf]]), np.array([0, 1])),
        (np.ones((2, 1)), np.array([[0], [1]])),
        (np.ones((2, 1)), np.array([0])),
        (np.ones((2, 1)), np.array([0.0, 1.0])),
        (np.ones((2, 1)), np.array([False, True])),
        (np.ones((2, 1)), np.array([0, 2])),
        (np.ones((2, 1)), np.array([0, 0])),
    ],
)
def test_fit_linear_probe_rejects_invalid_datasets(states: object, labels: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(states, labels, regularization=1e-6)


@pytest.mark.parametrize("regularization", [0.0, -1.0, np.inf, np.nan, True])
def test_fit_linear_probe_rejects_invalid_regularization(regularization: object) -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=regularization,
        )


def test_fit_wraps_numpy_solve_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = np.linalg.LinAlgError("singular")

    def fail_solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise failure

    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    with pytest.raises(RuntimeError, match="linear probe solve failed") as caught:
        fit_linear_probe(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            regularization=1e-6,
        )
    assert caught.value.__cause__ is failure
