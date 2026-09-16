from __future__ import annotations

import math

import numpy as np
import pytest

from neural_state_machine.phase_c4a_diagnostics.geometry import analyze_design_geometry


def _full_rank_design() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_full_rank_geometry_uses_registered_svd_formulas():
    design = _full_rank_design()
    metrics = analyze_design_geometry(design)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    expected_tol = max(design.shape) * np.finfo(np.float64).eps * singular[0]
    expected_shrink = singular * singular / (singular * singular + 1e-6)
    expected_leverage = np.sum((u * u) * expected_shrink[np.newaxis, :], axis=1)

    assert metrics.row_count == 6
    assert metrics.column_count == 4
    assert metrics.feature_size == 2
    assert metrics.tolerance == expected_tol
    assert metrics.rank == int(np.count_nonzero(singular > expected_tol)) == 4
    assert metrics.nullity == 0
    np.testing.assert_array_equal(metrics.singular_values, singular)
    assert metrics.singular_values.flags.writeable is False
    assert metrics.sigma_max == singular[0]
    assert metrics.sigma_min_retained == singular[-1]
    assert metrics.condition_number == singular[0] / singular[-1]
    assert metrics.frobenius_norm == np.linalg.norm(design, ord="fro")
    assert metrics.stable_rank == metrics.frobenius_norm**2 / metrics.sigma_max**2
    assert metrics.ridge_effective_dof == pytest.approx(float(expected_shrink.sum()))
    assert metrics.ridge_effective_dof == pytest.approx(float(expected_leverage.sum()))
    assert metrics.leverage_min == pytest.approx(float(expected_leverage.min()))
    assert metrics.leverage_median == pytest.approx(float(np.median(expected_leverage)))
    assert metrics.leverage_max == pytest.approx(float(expected_leverage.max()))
    assert metrics.action_block_column_norms == pytest.approx(
        (
            float(np.linalg.norm(design[:, :2], ord="fro")),
            float(np.linalg.norm(design[:, 2:], ord="fro")),
        )
    )
    assert metrics.cross_block_gram_frobenius == pytest.approx(
        float(np.linalg.norm(design[:, :2].T @ design[:, 2:], ord="fro"))
    )
    assert metrics.bias_column_norms == pytest.approx(
        (
            float(np.linalg.norm(design[:, 1])),
            float(np.linalg.norm(design[:, 3])),
        )
    )


def test_rank_deficient_geometry_reports_nullity_and_infinite_condition():
    design = np.asarray(
        [
            [1.0, 2.0, 1.0, 2.0],
            [2.0, 4.0, 2.0, 4.0],
            [3.0, 6.0, 3.0, 6.0],
            [4.0, 8.0, 4.0, 8.0],
        ],
        dtype=np.float64,
    )
    metrics = analyze_design_geometry(design)
    singular = np.linalg.svd(design, compute_uv=False)
    expected_tol = max(design.shape) * np.finfo(np.float64).eps * singular[0]

    assert metrics.rank == int(np.count_nonzero(singular > expected_tol)) == 1
    assert metrics.nullity == 3
    assert metrics.sigma_min_retained == pytest.approx(singular[0])
    assert math.isinf(metrics.condition_number)
    assert 0.0 < metrics.ridge_effective_dof < 1.0


def test_leverage_percentiles_use_nearest_rank():
    design = _full_rank_design()
    metrics = analyze_design_geometry(design)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    shrink = singular * singular / (singular * singular + 1e-6)
    leverage = np.sort(np.sum((u * u) * shrink[np.newaxis, :], axis=1))

    p10 = leverage[math.ceil(0.10 * leverage.size) - 1]
    p90 = leverage[math.ceil(0.90 * leverage.size) - 1]
    assert metrics.leverage_p10 == pytest.approx(float(p10))
    assert metrics.leverage_p90 == pytest.approx(float(p90))


def test_geometry_rejects_nonfinite_odd_block_or_nonregistered_penalty():
    with pytest.raises(ValueError, match="two equal action blocks"):
        analyze_design_geometry(np.ones((4, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        analyze_design_geometry(np.asarray([[1.0, 0.0], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="1e-6"):
        analyze_design_geometry(np.eye(4, dtype=np.float64), penalty=1e-3)
