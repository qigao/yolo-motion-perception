from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase3c_diagnostics import accounting


def _g(action: int, hidden: float) -> np.ndarray:
    feature = np.array([hidden, 1.0], dtype=np.float64)
    feature /= float(feature @ feature)
    result = np.zeros((2, 2), dtype=np.float64)
    result[action] = feature
    return result


def test_eligibility_history_split_reconstructs_trace_and_actual_update() -> None:
    helper = getattr(accounting, "eligibility_history_split", None)
    assert helper is not None, "D3 eligibility history split must be implemented"

    histories = (
        (0, np.array([1.0], dtype=np.float64)),
        (1, np.array([2.0], dtype=np.float64)),
        (0, np.array([-1.0], dtype=np.float64)),
    )
    rho = 0.5
    expected_source = rho**2 * _g(0, 1.0) + _g(0, -1.0)
    expected_other = rho * _g(1, 2.0)
    eligibility = expected_source + expected_other
    feedback = 1.0
    prediction = 0.25
    step_size = 0.1
    actual_update = step_size * (feedback - prediction) * eligibility
    reference = np.array(expected_source, copy=True)

    row = helper(
        history=histories,
        step=2,
        source_steps=(0, 2),
        captured_eligibility=eligibility,
        feedback=feedback,
        prediction_trace=prediction,
        step_size=step_size,
        rho=rho,
        actual_update=actual_update,
        reference_direction=reference,
    )

    np.testing.assert_allclose(row["eligibility_source"], expected_source, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(row["eligibility_other"], expected_other, rtol=0.0, atol=0.0)
    assert row["eligibility_reconstruction_max_residual"] == 0.0
    assert row["update_reconstruction_max_residual"] == 0.0
    assert row["source_update_norm"] > 0.0
    assert row["other_update_norm"] > 0.0
    assert row["reference_norm"] > 0.0
    assert row["actual_vs_reference_cosine"]["value"] is not None


def test_eligibility_history_split_fails_closed_on_trace_mismatch() -> None:
    helper = getattr(accounting, "eligibility_history_split", None)
    assert helper is not None, "D3 eligibility history split must be implemented"
    history = ((0, np.array([1.0], dtype=np.float64)),)
    wrong = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="residual"):
        helper(
            history=history,
            step=0,
            source_steps=(0,),
            captured_eligibility=wrong,
            feedback=1.0,
            prediction_trace=0.0,
            step_size=0.1,
            rho=0.72,
            actual_update=np.zeros((2, 2), dtype=np.float64),
            reference_direction=_g(0, 1.0),
        )
