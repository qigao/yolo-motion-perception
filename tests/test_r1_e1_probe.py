from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e1_probe import (
    BinaryRidgeProbe,
    MulticlassRidgeProbe,
    RidgeDiagnostics,
    fit_binary_ridge,
    fit_multiclass_ridge,
    prediction_digest,
)


def _diagnostics(feature_count: int, train_count: int = 4) -> RidgeDiagnostics:
    return RidgeDiagnostics(
        train_count=train_count,
        design_shape=(train_count, feature_count + 1),
        rank=min(train_count, feature_count + 1),
        singular_values=tuple(float(index + 1) for index in range(min(train_count, feature_count + 1))),
    )


def test_binary_probe_copies_coefficients_and_uses_class_zero_on_exact_tie() -> None:
    source = np.array([2.0, -1.0])
    probe = BinaryRidgeProbe(source, 0.0, _diagnostics(2))
    states = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 1.0]])
    source[:] = 99.0

    scores = probe.predict_scores(states)
    choices = probe.predict(states)

    np.testing.assert_array_equal(probe.coefficients, [2.0, -1.0])
    np.testing.assert_array_equal(scores, [2.0, -2.0, 0.0])
    np.testing.assert_array_equal(choices, [1, 0, 0])
    assert probe.coefficients.dtype == np.float64
    assert not probe.coefficients.flags.writeable
    assert not scores.flags.writeable
    assert not choices.flags.writeable


def test_multiclass_probe_copies_parameters_and_lowest_index_wins_ties() -> None:
    coefficients = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    bias = np.array([0.0, 0.0, 0.0])
    probe = MulticlassRidgeProbe(coefficients, bias, _diagnostics(2))
    states = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    coefficients[:] = 50.0
    bias[:] = 50.0

    scores = probe.predict_scores(states)
    choices = probe.predict(states)

    np.testing.assert_array_equal(
        scores,
        [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(choices, [0, 1, 0])
    assert probe.coefficients.shape == (3, 2)
    assert probe.bias.shape == (3,)
    assert not probe.coefficients.flags.writeable
    assert not probe.bias.flags.writeable


@pytest.mark.parametrize(
    ("coefficients", "bias"),
    [
        (1.0, 0.0),
        (np.ones((1, 2)), 0.0),
        (np.array([]), 0.0),
        (np.array([np.nan]), 0.0),
        (np.array([1.0]), np.inf),
    ],
)
def test_binary_probe_rejects_invalid_parameters(coefficients: object, bias: object) -> None:
    with pytest.raises(ValueError):
        BinaryRidgeProbe(coefficients, bias, _diagnostics(1))


@pytest.mark.parametrize(
    ("coefficients", "bias"),
    [
        (np.array([1.0, 2.0]), np.array([0.0, 0.0])),
        (np.ones((1, 2, 3)), np.array([0.0, 0.0])),
        (np.ones((2, 0)), np.array([0.0, 0.0])),
        (np.ones((2, 2)), np.array([0.0])),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.zeros(2)),
        (np.ones((2, 2)), np.array([0.0, np.inf])),
    ],
)
def test_multiclass_probe_rejects_invalid_parameters(coefficients: object, bias: object) -> None:
    with pytest.raises(ValueError):
        MulticlassRidgeProbe(coefficients, bias, _diagnostics(2))


@pytest.mark.parametrize(
    "states",
    [
        1.0,
        np.array([1.0, 2.0]),
        np.ones((1, 3)),
        np.array([[1.0, np.nan]]),
        np.empty((0, 2)),
    ],
)
def test_fit_binary_rejects_invalid_state_matrix(states: object) -> None:
    with pytest.raises(ValueError):
        fit_binary_ridge(states, np.array([0, 1]), regularization=1e-6)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([[0, 1]]),
        np.array([0]),
        np.array([0.0, 1.0]),
        np.array([0, 2]),
        np.array([0, 0]),
    ],
)
def test_fit_binary_rejects_invalid_labels(labels: object) -> None:
    with pytest.raises(ValueError):
        fit_binary_ridge(np.array([[0.0], [1.0]]), labels, regularization=1e-6)


@pytest.mark.parametrize("regularization", [0.0, -1.0, np.inf, np.nan, True, "bad"])
def test_fit_rejects_invalid_regularization(regularization: object) -> None:
    states = np.array([[0.0], [1.0], [2.0], [3.0]])
    binary_labels = np.array([0, 0, 1, 1])
    multiclass_labels = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="regularization"):
        fit_binary_ridge(states, binary_labels, regularization=regularization)
    with pytest.raises(ValueError, match="regularization"):
        fit_multiclass_ridge(
            states,
            multiclass_labels,
            class_count=2,
            regularization=regularization,
        )


@pytest.mark.parametrize("class_count", [0, 1, True, 2.0, "2"])
def test_multiclass_fit_rejects_invalid_class_count(class_count: object) -> None:
    with pytest.raises(ValueError, match="class_count"):
        fit_multiclass_ridge(
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            class_count=class_count,
            regularization=1e-6,
        )


def test_binary_fit_matches_augmented_least_squares_reference() -> None:
    states = np.array([[-2.0, 0.5], [-1.0, -0.5], [1.0, 0.5], [2.0, -0.5]])
    labels = np.array([0, 0, 1, 1])
    regularization = 0.25
    design = np.column_stack((states, np.ones(states.shape[0])))
    penalty = np.zeros((states.shape[1], states.shape[1] + 1))
    penalty[:, : states.shape[1]] = np.sqrt(regularization) * np.eye(states.shape[1])
    augmented_design = np.vstack((design, penalty))
    augmented_target = np.concatenate((np.array([-1.0, -1.0, 1.0, 1.0]), np.zeros(2)))
    expected, *_ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)

    fitted = fit_binary_ridge(states, labels, regularization=regularization)

    np.testing.assert_allclose(fitted.coefficients, expected[:-1], rtol=0.0, atol=1e-14)
    assert fitted.bias == pytest.approx(float(expected[-1]), abs=1e-14)
    assert fitted.diagnostics.train_count == 4
    assert fitted.diagnostics.design_shape == (4, 3)
    expected_singular = np.linalg.svd(design, compute_uv=False)
    np.testing.assert_allclose(fitted.diagnostics.singular_values, expected_singular)
    assert fitted.diagnostics.rank == int(np.linalg.matrix_rank(design))


def test_multiclass_fit_matches_augmented_least_squares_reference() -> None:
    states = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    labels = np.array([0, 1, 2, 1])
    regularization = 0.5
    class_count = 3
    design = np.column_stack((states, np.ones(states.shape[0])))
    penalty = np.zeros((states.shape[1], states.shape[1] + 1))
    penalty[:, : states.shape[1]] = np.sqrt(regularization) * np.eye(states.shape[1])
    augmented_design = np.vstack((design, penalty))
    target = np.zeros((states.shape[0], class_count))
    target[np.arange(states.shape[0]), labels] = 1.0
    augmented_target = np.vstack((target, np.zeros((states.shape[1], class_count))))
    expected, *_ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)

    fitted = fit_multiclass_ridge(
        states,
        labels,
        class_count=class_count,
        regularization=regularization,
    )

    np.testing.assert_allclose(fitted.coefficients, expected[:-1].T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(fitted.bias, expected[-1], rtol=0.0, atol=1e-14)


def test_bias_column_is_unpenalized_even_with_large_regularization() -> None:
    states = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])
    regularization = 1_000.0
    design = np.column_stack((states, np.ones(4)))
    penalty = np.array([[np.sqrt(regularization), 0.0]])
    expected, *_ = np.linalg.lstsq(
        np.vstack((design, penalty)),
        np.concatenate((np.array([-1.0, -1.0, 1.0, 1.0]), [0.0])),
        rcond=None,
    )

    fitted = fit_binary_ridge(states, labels, regularization=regularization)

    np.testing.assert_allclose(fitted.coefficients, expected[:-1], rtol=0.0, atol=1e-14)
    assert fitted.bias == pytest.approx(float(expected[-1]), abs=1e-14)


def test_coefficient_digest_is_repeatable_and_parameter_sensitive() -> None:
    diagnostics = _diagnostics(2)
    first = BinaryRidgeProbe(np.array([1.0, 2.0]), 0.5, diagnostics)
    same = BinaryRidgeProbe(np.array([1.0, 2.0]), 0.5, diagnostics)
    changed_weight = BinaryRidgeProbe(np.array([1.0, 2.5]), 0.5, diagnostics)
    changed_bias = BinaryRidgeProbe(np.array([1.0, 2.0]), 0.25, diagnostics)

    assert first.coefficient_digest() == same.coefficient_digest()
    assert first.coefficient_digest() != changed_weight.coefficient_digest()
    assert first.coefficient_digest() != changed_bias.coefficient_digest()


def test_multiclass_coefficient_digest_is_parameter_sensitive() -> None:
    diagnostics = _diagnostics(2)
    first = MulticlassRidgeProbe(np.eye(2), np.zeros(2), diagnostics)
    same = MulticlassRidgeProbe(np.eye(2), np.zeros(2), diagnostics)
    changed = MulticlassRidgeProbe(np.eye(2), np.array([0.0, 0.1]), diagnostics)

    assert first.coefficient_digest() == same.coefficient_digest()
    assert first.coefficient_digest() != changed.coefficient_digest()


def test_prediction_digest_is_repeatable_and_choice_sensitive() -> None:
    first = prediction_digest(np.array([0, 1, 2, 1], dtype=np.int64))
    same = prediction_digest(np.array([0, 1, 2, 1], dtype=np.int64))
    changed = prediction_digest(np.array([0, 1, 2, 0], dtype=np.int64))

    assert first == same
    assert first != changed


@pytest.mark.parametrize(
    "choices",
    [1, np.array([[0, 1]]), np.array([0.0, 1.0]), np.array([0, -1])],
)
def test_prediction_digest_rejects_invalid_choices(choices: object) -> None:
    with pytest.raises(ValueError):
        prediction_digest(choices)
