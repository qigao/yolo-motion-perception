from __future__ import annotations

import numpy as np

from neural_state_machine.r1_e1_probe import fit_multiclass_ridge


def _api():
    from neural_state_machine.r1_e2_readout import (
        RIDGE_REGULARIZATION,
        causal_mean_pool,
        causal_mean_pool_batch,
        fit_instant_multiclass,
        fit_temporal_mean_multiclass,
    )

    return (
        RIDGE_REGULARIZATION,
        causal_mean_pool,
        causal_mean_pool_batch,
        fit_instant_multiclass,
        fit_temporal_mean_multiclass,
    )


def test_causal_mean_pool_uses_only_trailing_history_at_decision() -> None:
    _, pool, _, _, _ = _api()
    states = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [999.0, 999.0],
        ],
        dtype=np.float64,
    )

    pooled = pool(states, window=3, decision_index=3)

    assert np.array_equal(pooled, np.array([3.0, 30.0], dtype=np.float64))
    assert pooled.dtype == np.float64
    assert not pooled.flags.writeable


def test_batch_pooling_matches_individual_causal_pooling() -> None:
    _, pool, pool_batch, _, _ = _api()
    trajectories = np.arange(2 * 6 * 3, dtype=np.float64).reshape(2, 6, 3)

    batched = pool_batch(trajectories, window=4, decision_index=4)
    expected = np.vstack(
        [pool(trajectory, window=4, decision_index=4) for trajectory in trajectories]
    )

    assert np.array_equal(batched, expected)
    assert batched.shape == (2, 3)
    assert not batched.flags.writeable


def test_instant_readout_reuses_frozen_r1_e1_ridge_law() -> None:
    regularization, _, _, fit_instant, _ = _api()
    assert regularization == 1e-6

    states = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.5],
            [1.0, -0.5],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    e2_probe = fit_instant(states, labels, class_count=2)
    reference = fit_multiclass_ridge(
        states,
        labels,
        class_count=2,
        regularization=1e-6,
    )

    assert e2_probe.coefficient_digest() == reference.coefficient_digest()
    assert np.array_equal(e2_probe.predict(states), reference.predict(states))


def test_temporal_readout_is_exactly_ridge_on_causal_mean_states() -> None:
    _, _, pool_batch, _, fit_temporal = _api()
    trajectories = np.array(
        [
            [[-3.0], [-2.0], [-1.0], [50.0]],
            [[-2.0], [-1.0], [-0.5], [50.0]],
            [[0.5], [1.0], [2.0], [-50.0]],
            [[1.0], [2.0], [3.0], [-50.0]],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    pooled = pool_batch(trajectories, window=3, decision_index=2)

    e2_probe = fit_temporal(
        trajectories,
        labels,
        class_count=2,
        window=3,
        decision_index=2,
    )
    reference = fit_multiclass_ridge(
        pooled,
        labels,
        class_count=2,
        regularization=1e-6,
    )

    assert e2_probe.coefficient_digest() == reference.coefficient_digest()
    assert np.array_equal(e2_probe.predict(pooled), reference.predict(pooled))


def test_temporal_pooling_invalid_windows_fail_closed() -> None:
    _, pool, pool_batch, _, _ = _api()
    states = np.ones((4, 2), dtype=np.float64)
    batch = np.ones((2, 4, 2), dtype=np.float64)

    invalid_calls = (
        lambda: pool(states, window=0),
        lambda: pool(states, window=5),
        lambda: pool(states, window=2, decision_index=-1),
        lambda: pool(states, window=3, decision_index=1),
        lambda: pool_batch(batch, window=5),
        lambda: pool_batch(batch, window=2, decision_index=4),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid causal window must fail")
