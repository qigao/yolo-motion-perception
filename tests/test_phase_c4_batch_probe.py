from __future__ import annotations

import inspect

import numpy as np
import pytest

from neural_state_machine.phase_c4_batch_probe import (
    C4_RIDGE_REGULARIZATION,
    AnonymousBatchRidge,
    fit_anonymous_batch_ridge,
)


def _basis_observations() -> np.ndarray:
    rows = np.zeros((4, 2, 2), dtype=np.float64)
    rows[0, 0, 0] = 1.0
    rows[1, 0, 1] = 1.0
    rows[2, 1, 0] = 1.0
    rows[3, 1, 1] = 1.0
    return rows


def test_fit_surface_exposes_only_anonymous_operator_and_scalar_feedback():
    signature = inspect.signature(fit_anonymous_batch_ridge)
    assert tuple(signature.parameters) == ("expected_features", "aggregate_feedback")
    assert C4_RIDGE_REGULARIZATION == 1e-6


def test_fixed_ridge_matches_closed_form_on_action_blocked_basis():
    observations = _basis_observations()
    feedback = np.array((1.0, -2.0, 3.0, -4.0), dtype=np.float64)

    probe = fit_anonymous_batch_ridge(observations, feedback)

    assert isinstance(probe, AnonymousBatchRidge)
    expected = (feedback / (1.0 + C4_RIDGE_REGULARIZATION)).reshape(2, 2)
    np.testing.assert_allclose(probe.weights, expected, rtol=0.0, atol=1e-12)
    assert probe.weights.shape == (2, 2)
    assert probe.weights.dtype == np.float64
    assert probe.weights.flags.writeable is False
    assert probe.observation_count == 4
    assert probe.rank == 4
    assert probe.regularization == C4_RIDGE_REGULARIZATION


def test_probe_predicts_aggregate_with_same_z_operator():
    observations = _basis_observations()
    feedback = np.array((1.0, -2.0, 3.0, -4.0), dtype=np.float64)
    probe = fit_anonymous_batch_ridge(observations, feedback)
    z = np.array(((2.0, -1.0), (0.5, 3.0)), dtype=np.float64)

    prediction = probe.predict_aggregate(z)

    assert prediction == pytest.approx(float(np.sum(probe.weights * z)), abs=1e-15)


def test_probe_action_values_use_fitted_action_rows():
    observations = _basis_observations()
    feedback = np.array((1.0, -2.0, 3.0, -4.0), dtype=np.float64)
    probe = fit_anonymous_batch_ridge(observations, feedback)
    feature = np.array((0.25, 1.0), dtype=np.float64)

    values = probe.action_values(feature)

    np.testing.assert_allclose(values, probe.weights @ feature, rtol=0.0, atol=1e-15)
    assert values.flags.writeable is False


def test_fit_is_repeatable_and_does_not_alias_inputs():
    observations = _basis_observations()
    feedback = np.array((1.0, -2.0, 3.0, -4.0), dtype=np.float64)
    left = fit_anonymous_batch_ridge(observations, feedback)
    right = fit_anonymous_batch_ridge(observations.copy(), feedback.copy())
    frozen = left.weights.copy()

    observations[:] = 99.0
    feedback[:] = 99.0

    assert left.weights.tobytes() == right.weights.tobytes()
    np.testing.assert_array_equal(left.weights, frozen)


@pytest.mark.parametrize(
    ("observations", "feedback"),
    [
        (np.zeros((0, 2, 2)), np.zeros(0)),
        (np.zeros((3, 2)), np.zeros(3)),
        (np.zeros((3, 1, 2)), np.zeros(3)),
        (np.zeros((3, 2, 0)), np.zeros(3)),
        (np.full((3, 2, 2), np.nan), np.zeros(3)),
        (np.zeros((3, 2, 2)), np.zeros(2)),
        (np.zeros((3, 2, 2)), np.zeros((3, 1))),
        (np.zeros((3, 2, 2)), np.array((0.0, np.inf, 0.0))),
    ],
)
def test_invalid_batch_inputs_fail_closed(observations, feedback):
    with pytest.raises(ValueError):
        fit_anonymous_batch_ridge(observations, feedback)


def test_prediction_validation_fails_closed_without_mutation():
    probe = fit_anonymous_batch_ridge(
        _basis_observations(),
        np.array((1.0, -2.0, 3.0, -4.0), dtype=np.float64),
    )
    before = probe.weights.tobytes()

    with pytest.raises(ValueError):
        probe.predict_aggregate(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        probe.action_values(np.zeros(3))

    assert probe.weights.tobytes() == before
