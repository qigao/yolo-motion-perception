import numpy as np
import pytest

from neural_state_machine.r1_e3m_regression import (
    RIDGE_REGULARIZATION,
    fit_multioutput_ridge,
    r2_summary,
)


def test_multioutput_ridge_recovers_affine_targets():
    x = np.linspace(-1.0, 1.0, 80, dtype=np.float64).reshape(-1, 1)
    features = np.column_stack((x[:, 0], x[:, 0] ** 2))
    targets = np.column_stack(
        (
            2.0 * features[:, 0] - 0.5 * features[:, 1] + 0.25,
            -1.5 * features[:, 0] + 0.75,
        )
    )

    probe = fit_multioutput_ridge(features, targets)
    predicted = probe.predict(features)

    assert RIDGE_REGULARIZATION == 1e-6
    assert predicted.shape == targets.shape
    assert np.max(np.abs(predicted - targets)) < 1e-5


def test_bias_is_unpenalized():
    features = np.zeros((12, 3), dtype=np.float64)
    targets = np.full((12, 2), (3.5, -2.0), dtype=np.float64)

    probe = fit_multioutput_ridge(features, targets)
    predicted = probe.predict(features)

    assert np.allclose(probe.bias, np.array([3.5, -2.0]))
    assert np.allclose(predicted, targets)


def test_probe_digest_is_deterministic():
    features = np.arange(24, dtype=np.float64).reshape(8, 3) / 24.0
    targets = np.column_stack((features[:, 0], features[:, 1]))

    first = fit_multioutput_ridge(features, targets)
    second = fit_multioutput_ridge(features, targets)

    assert first.coefficient_digest() == second.coefficient_digest()


def test_nonfinite_or_wrong_shapes_fail_closed():
    features = np.ones((4, 2), dtype=np.float64)
    targets = np.ones((4, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="rank two"):
        fit_multioutput_ridge(features[:, 0], targets)
    with pytest.raises(ValueError, match="sample count"):
        fit_multioutput_ridge(features, targets[:3])
    broken = features.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_multioutput_ridge(broken, targets)


def test_r2_summary_excludes_degenerate_channels():
    actual = np.column_stack(
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.ones(4),
        )
    )
    predicted = np.column_stack(
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.zeros(4),
        )
    )

    summary = r2_summary(actual, predicted)

    assert summary.valid_channel_count == 1
    assert summary.degenerate_channel_count == 1
    assert summary.macro_r2 == pytest.approx(1.0)
    assert summary.per_channel_r2[0] == pytest.approx(1.0)
    assert np.isnan(summary.per_channel_r2[1])
