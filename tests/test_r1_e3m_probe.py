from __future__ import annotations

import numpy as np
import pytest


def _api():
    from neural_state_machine.r1_e3m_probe import (
        DELAYS,
        RIDGE_REGULARIZATION,
        delay_valid_mask,
        delayed_geometry_targets,
        evaluate_regression,
        fit_instantaneous_delay_probe,
        fit_reservoir_delay_probe,
    )

    return (
        DELAYS,
        RIDGE_REGULARIZATION,
        delay_valid_mask,
        delayed_geometry_targets,
        evaluate_regression,
        fit_instantaneous_delay_probe,
        fit_reservoir_delay_probe,
    )


def _tensors(sample_count: int = 12) -> np.ndarray:
    values = np.zeros((sample_count, 20, 6), dtype=np.float64)
    for sample in range(sample_count):
        for bin_index in range(20):
            values[sample, bin_index, 0] = 0.01 * sample + 0.001 * bin_index
            values[sample, bin_index, 1] = 0.02 * sample + 0.002 * bin_index
            values[sample, bin_index, 2] = 0.1 + 0.001 * sample
            values[sample, bin_index, 3] = 0.2 + 0.0015 * sample
            values[sample, bin_index, 4] = 0.9
            values[sample, bin_index, 5] = 1.0
    return values


def test_delays_are_exactly_frozen_values() -> None:
    DELAYS, *_ = _api()

    assert DELAYS == (1, 2, 5, 10, 15)


def test_delayed_target_is_exact_geometry_from_bin_19_minus_delay() -> None:
    _, _, _, delayed_targets, *_ = _api()
    tensors = _tensors()

    targets = delayed_targets(tensors, 5)

    np.testing.assert_array_equal(targets, tensors[:, 14, :4])
    assert targets.flags.writeable is False


def test_unregistered_delay_fails_closed() -> None:
    _, _, _, delayed_targets, *_ = _api()

    with pytest.raises(ValueError, match="registered delays"):
        delayed_targets(_tensors(), 3)


def test_instantaneous_probe_uses_bin_19_and_frozen_regularization() -> None:
    _, regularization, _, _, _, fit_instantaneous, _ = _api()
    tensors = _tensors()

    probe = fit_instantaneous(tensors, 5)

    assert regularization == 1e-6
    assert probe.feature_count == 6
    predictions = probe.predict(tensors[:, 19, :])
    assert predictions.shape == (tensors.shape[0], 4)


def test_reservoir_probe_accepts_final_states_and_same_delayed_target() -> None:
    _, _, _, delayed_targets, _, _, fit_reservoir = _api()
    tensors = _tensors()
    targets = delayed_targets(tensors, 10)
    states = np.column_stack(
        (
            tensors[:, 19, 0],
            tensors[:, 19, 1],
            tensors[:, 9, 0],
            tensors[:, 9, 1],
            np.arange(tensors.shape[0], dtype=np.float64),
        )
    )

    probe = fit_reservoir(states, tensors, 10)

    assert probe.feature_count == states.shape[1]
    predictions = probe.predict(states)
    assert predictions.shape == targets.shape


def test_regression_metrics_are_deterministic_and_per_channel() -> None:
    _, _, _, delayed_targets, evaluate, fit_instantaneous, _ = _api()
    tensors = _tensors()
    targets = delayed_targets(tensors, 1)
    probe = fit_instantaneous(tensors, 1)
    predictions = probe.predict(tensors[:, 19, :])

    first = evaluate(targets, predictions)
    second = evaluate(targets, predictions)

    assert first == second
    assert len(first.r2_per_channel) == 4
    assert first.sample_count == tensors.shape[0]
    assert np.isfinite(first.mean_r2)
    assert np.isfinite(first.mse)


def test_coefficient_digest_is_repeatable() -> None:
    _, _, _, _, _, fit_instantaneous, _ = _api()
    tensors = _tensors()

    first = fit_instantaneous(tensors, 2)
    second = fit_instantaneous(tensors, 2)

    assert first.coefficient_digest() == second.coefficient_digest()


def test_delay_valid_mask_requires_observed_target_bin() -> None:
    _, _, valid_mask, _, _, _, _ = _api()
    tensors = _tensors(4)
    target_bin = 19 - 10
    tensors[1, target_bin, :5] = 0.0
    tensors[1, target_bin, 5] = 0.0

    mask = valid_mask(tensors, 10)

    assert mask.tolist() == [True, False, True, True]
    assert mask.dtype == np.bool_
    assert mask.flags.writeable is False


def test_delay_mask_does_not_require_current_bin_presence() -> None:
    _, _, valid_mask, _, _, _, _ = _api()
    tensors = _tensors(3)
    tensors[0, 19, :5] = 0.0
    tensors[0, 19, 5] = 0.0

    mask = valid_mask(tensors, 15)

    assert mask.tolist() == [True, True, True]
