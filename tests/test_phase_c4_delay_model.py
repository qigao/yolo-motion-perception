from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase_c4_delay_model import (
    DecisionCreditRow,
    DelayLaw,
    build_marginalized_features,
    candidate_indices,
    current_weight_prediction,
)


def _row(index: int, action: int, feature: tuple[float, ...]) -> DecisionCreditRow:
    values = np.array(feature, dtype=np.float64)
    return DecisionCreditRow(
        decision_index=index,
        action_index=action,
        feature=values,
        denominator=float(values @ values),
    )


def test_registered_delay_law_is_fixed_and_normalized():
    law = DelayLaw.registered()
    assert law.support == (1, 3, 5)
    assert law.probabilities == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def test_immediate_delay_law_is_exact_boundary():
    law = DelayLaw.immediate()
    assert law.support == (0,)
    assert law.probabilities == (1.0,)


def test_registered_candidate_indices_are_only_1_3_5_lags():
    law = DelayLaw.registered()
    assert candidate_indices(5, 10, law) == (4, 2, 0)


def test_pre_start_candidates_are_truncated():
    law = DelayLaw.registered()
    assert candidate_indices(2, 10, law) == (1,)


def test_post_training_candidates_are_truncated_to_real_decisions():
    law = DelayLaw.registered()
    assert candidate_indices(8, 4, law) == (3,)
    assert candidate_indices(9, 4, law) == ()


def test_marginalized_features_use_only_candidate_rows():
    rows = (
        _row(0, 1, (2.0, 1.0)),
        _row(1, 1, (99.0, 1.0)),
        _row(2, 0, (3.0, 1.0)),
        _row(3, 1, (88.0, 1.0)),
        _row(4, 0, (4.0, 1.0)),
    )
    result = build_marginalized_features(
        rows,
        feedback_step=5,
        action_count=2,
        law=DelayLaw.registered(),
    )

    expected_z = np.zeros((2, 2), dtype=np.float64)
    expected_z[0] = (np.array((4.0, 1.0)) + np.array((3.0, 1.0))) / 3.0
    expected_z[1] = np.array((2.0, 1.0)) / 3.0

    expected_c = np.zeros((2, 2), dtype=np.float64)
    expected_c[0] = (
        np.array((4.0, 1.0)) / 17.0 + np.array((3.0, 1.0)) / 10.0
    ) / 3.0
    expected_c[1] = (np.array((2.0, 1.0)) / 5.0) / 3.0

    np.testing.assert_allclose(result.expected_feature, expected_z, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.normalized_credit, expected_c, rtol=0.0, atol=1e-15)
    assert not result.expected_feature.flags.writeable
    assert not result.normalized_credit.flags.writeable


def test_current_weight_prediction_uses_supplied_weights():
    z = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
    w1 = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    w2 = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert current_weight_prediction(w1, z) == 5.0
    assert current_weight_prediction(w2, z) == 7.0


@pytest.mark.parametrize(
    ("support", "probabilities"),
    [
        ((1, 1), (0.5, 0.5)),
        ((-1, 1), (0.5, 0.5)),
        ((1, 3), (0.5, -0.5)),
        ((1, 3), (0.5, float("nan"))),
        ((1, 3), (0.4, 0.4)),
    ],
)
def test_invalid_delay_laws_fail_closed(support, probabilities):
    with pytest.raises(ValueError):
        DelayLaw(support=support, probabilities=probabilities)


def test_decision_row_copies_feature_and_rejects_invalid_denominator():
    feature = np.array((2.0, 1.0), dtype=np.float64)
    row = DecisionCreditRow(0, 1, feature, 5.0)
    feature[0] = 999.0
    np.testing.assert_array_equal(row.feature, np.array((2.0, 1.0)))
    assert not row.feature.flags.writeable

    with pytest.raises(ValueError):
        DecisionCreditRow(0, 1, np.array((2.0, 1.0)), 0.0)
