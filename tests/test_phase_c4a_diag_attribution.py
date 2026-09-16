from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.phase_c4a_diagnostics.attribution import (
    associate_target,
    compare_weight_alignment,
    fit_supervised_reference,
    project_target,
)


def test_zero_variance_observer_returns_null_correlation_with_reason():
    target = np.asarray([0.0, 1.0, 2.0, 3.0, 0.5, -0.5], dtype=np.float64)
    association = associate_target(
        target,
        real_decision_count=4,
        actions=np.ones(4, dtype=np.int64),
        correct_actions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        cue_delays=np.asarray([1, 2, 1, 2], dtype=np.int64),
        hidden_delays=np.asarray([1, 3, 5, 1], dtype=np.int64),
        normal_rewards=np.asarray([1.0, -1.0, 1.0, -1.0]),
        donor_rewards=np.asarray([-1.0, 1.0, 1.0, -1.0]),
        arrival_counts=np.asarray([0, 1, 2, 1, 1, 0], dtype=np.int64),
    )

    current_action = association["scalar"]["current_action"]
    assert current_action.count == 4
    assert current_action.correlation is None
    assert current_action.reason is not None
    assert "zero variance" in current_action.reason
    np.testing.assert_array_equal(association["drain_target"], target[4:])
    assert association["drain_target"].flags.writeable is False


def test_association_reports_fixed_categories_and_lag_directions():
    target = np.asarray([0.0, 1.0, -1.0, 2.0, 1.0, -2.0, 0.25], dtype=np.float64)
    association = associate_target(
        target,
        real_decision_count=6,
        actions=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        correct_actions=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
        cue_delays=np.asarray([1, 2, 3, 1, 2, 3], dtype=np.int64),
        hidden_delays=np.asarray([1, 3, 5, 1, 3, 5], dtype=np.int64),
        normal_rewards=np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
        donor_rewards=np.asarray([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]),
        arrival_counts=np.asarray([0, 1, 2, 1, 0, 3, 1], dtype=np.int64),
    )

    assert set(association["scalar"]) == {
        "current_action",
        "correct_action",
        "action_correctness",
        "normal_latent_reward",
        "shuffled_donor_reward",
    }
    assert set(association["categorical"]) == {
        "cue_delay",
        "hidden_delay",
        "block_position",
        "arrival_bucket",
    }
    assert tuple(row.category for row in association["categorical"]["arrival_bucket"]) == (
        "no_arrival",
        "one_source",
        "collision",
    )
    assert set(association["lagged"]) == {
        "correct_action_previous",
        "correct_action_future",
        "action_correctness_previous",
        "action_correctness_future",
    }
    assert tuple(row.lag for row in association["lagged"]["correct_action_previous"]) == tuple(
        range(1, 11)
    )


def test_target_projection_recomposes_and_residual_is_orthogonal_to_design():
    design = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=np.float64,
    )
    target = np.asarray([1.0, 2.0, 4.0, -1.0], dtype=np.float64)
    projection = project_target(design, target)

    np.testing.assert_allclose(
        projection.parallel + projection.residual,
        target,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        design.T @ projection.residual,
        np.zeros(design.shape[1]),
        rtol=0.0,
        atol=1e-10,
    )
    assert projection.parallel.flags.writeable is False
    assert projection.residual.flags.writeable is False
    target_norm = np.linalg.norm(target)
    assert projection.parallel_ratio == pytest.approx(
        float(np.linalg.norm(projection.parallel) / target_norm)
    )
    assert projection.residual_ratio == pytest.approx(
        float(np.linalg.norm(projection.residual) / target_norm)
    )
    assert projection.fit_residual_norm == pytest.approx(float(np.linalg.norm(projection.residual)))


def test_supervised_reference_uses_fixed_frobenius_ridge_and_no_eval_labels():
    features = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    correct_actions = np.asarray([0, 1, 0, 1], dtype=np.int64)
    fit = fit_supervised_reference(features, correct_actions)

    expected_y = np.full((4, 2), -1.0, dtype=np.float64)
    expected_y[np.arange(4), correct_actions] = 1.0
    aug_x = np.vstack((features, np.sqrt(1e-6) * np.eye(3)))
    aug_y = np.vstack((expected_y, np.zeros((3, 2), dtype=np.float64)))
    expected_coef, _, expected_rank, expected_singular = np.linalg.lstsq(
        aug_x,
        aug_y,
        rcond=None,
    )

    assert fit.row_count == 4
    assert fit.feature_size == 3
    assert fit.penalty == 1e-6
    assert fit.augmented_rank == expected_rank
    np.testing.assert_allclose(fit.weights, expected_coef.T, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(fit.singular_values, expected_singular, rtol=0.0, atol=0.0)
    assert fit.weights.flags.writeable is False
    assert fit.singular_values.flags.writeable is False


def test_weight_alignment_is_exact_for_identical_nonzero_readouts():
    weights = np.asarray(
        [
            [1.0, -2.0, 0.5],
            [-1.5, 0.25, 2.0],
        ],
        dtype=np.float64,
    )
    alignment = compare_weight_alignment(weights, weights.copy())

    assert alignment.global_cosine == pytest.approx(1.0)
    assert alignment.action_cosines == pytest.approx((1.0, 1.0))
    assert alignment.norm_ratio == pytest.approx(1.0)
    assert alignment.action_norm_ratios == pytest.approx((1.0, 1.0))
    assert alignment.supervised_projection_magnitude == pytest.approx(float(np.linalg.norm(weights)))
    assert alignment.reason is None
