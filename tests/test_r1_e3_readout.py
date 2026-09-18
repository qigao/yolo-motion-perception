from __future__ import annotations

import numpy as np
import pytest

from neural_state_machine.r1_e1_probe import fit_multiclass_ridge


def _api():
    from neural_state_machine.r1_e3_readout import (
        fit_frame_only,
        fit_reservoir_instantaneous,
        fit_reservoir_temporal_mean,
        temporal_mean_last_four,
    )

    return (
        fit_frame_only,
        fit_reservoir_instantaneous,
        fit_reservoir_temporal_mean,
        temporal_mean_last_four,
    )


def _labels() -> np.ndarray:
    return np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)


def _observations() -> np.ndarray:
    values = np.arange(8 * 20 * 14, dtype=np.float64).reshape(8, 20, 14)
    return values / float(values.max())


def _trajectories() -> np.ndarray:
    values = np.arange(8 * 20 * 6, dtype=np.float64).reshape(8, 20, 6)
    return np.tanh(values / 100.0)


def test_temporal_mean_is_exact_mean_of_bins_16_to_19() -> None:
    *_, pool = _api()
    trajectories = _trajectories()

    pooled = pool(trajectories)

    np.testing.assert_array_equal(
        pooled,
        np.mean(trajectories[:, 16:20, :], axis=1, dtype=np.float64),
    )
    assert pooled.dtype == np.float64
    assert not pooled.flags.writeable


def test_frame_only_is_exactly_bin_19_with_frozen_ridge() -> None:
    fit_frame, _, _, _ = _api()
    observations = _observations()
    labels = _labels()

    probe = fit_frame(observations, labels, class_count=4)
    reference = fit_multiclass_ridge(
        observations[:, 19, :],
        labels,
        class_count=4,
        regularization=1e-6,
    )

    assert probe.coefficient_digest() == reference.coefficient_digest()


def test_reservoir_instantaneous_is_exactly_state_19_with_frozen_ridge() -> None:
    _, fit_instant, _, _ = _api()
    trajectories = _trajectories()
    labels = _labels()

    probe = fit_instant(trajectories, labels, class_count=4)
    reference = fit_multiclass_ridge(
        trajectories[:, 19, :],
        labels,
        class_count=4,
        regularization=1e-6,
    )

    assert probe.coefficient_digest() == reference.coefficient_digest()


def test_temporal_readout_uses_fixed_last_four_and_frozen_ridge() -> None:
    _, _, fit_temporal, _ = _api()
    trajectories = _trajectories()
    labels = _labels()

    probe = fit_temporal(trajectories, labels, class_count=4)
    reference = fit_multiclass_ridge(
        np.mean(trajectories[:, 16:20, :], axis=1, dtype=np.float64),
        labels,
        class_count=4,
        regularization=1e-6,
    )

    assert probe.coefficient_digest() == reference.coefficient_digest()


def test_readouts_require_exactly_20_registered_bins() -> None:
    fit_frame, fit_instant, fit_temporal, pool = _api()
    labels = _labels()
    observations = _observations()[:, :19, :]
    trajectories = _trajectories()[:, :19, :]

    with pytest.raises(ValueError, match="20"):
        fit_frame(observations, labels, class_count=4)
    with pytest.raises(ValueError, match="20"):
        fit_instant(trajectories, labels, class_count=4)
    with pytest.raises(ValueError, match="20"):
        fit_temporal(trajectories, labels, class_count=4)
    with pytest.raises(ValueError, match="20"):
        pool(trajectories)


def test_temporal_api_exposes_no_window_override() -> None:
    _, _, fit_temporal, pool = _api()

    with pytest.raises(TypeError):
        fit_temporal(_trajectories(), _labels(), class_count=4, window=3)
    with pytest.raises(TypeError):
        pool(_trajectories(), window=3)
