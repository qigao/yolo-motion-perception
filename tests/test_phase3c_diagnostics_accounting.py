from __future__ import annotations

import numpy as np

from neural_state_machine.phase3c_diagnostics.accounting import (
    cosine_metric,
    expected_drain_delta,
    residual_terms,
)


def test_residual_identity_retains_cross_term() -> None:
    terms = residual_terms(feedback=3.0, prediction_trace=1.0, source_prediction=2.5)
    assert terms.feedback_minus_prediction == 2.0
    assert terms.feedback_minus_source_prediction == 0.5
    assert terms.source_minus_prediction == 1.5
    assert terms.feedback_minus_prediction == (
        terms.feedback_minus_source_prediction + terms.source_minus_prediction
    )
    assert terms.squared_cross_term == 2.0 * 0.5 * 1.5


def test_drain_formula_keeps_prediction_and_eligibility_fixed() -> None:
    eligibility = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)
    delta = expected_drain_delta((0.0, 2.0, -1.0), 0.25, eligibility, 0.1)
    scalar = 0.1 * ((0.0 + 2.0 - 1.0) - 3 * 0.25)
    np.testing.assert_allclose(delta, scalar * eligibility)


def test_zero_norm_cosine_is_explicitly_undefined() -> None:
    metric = cosine_metric(np.zeros((2, 2)), np.ones((2, 2)))
    assert metric.value is None
    assert metric.reason == "zero_norm"
