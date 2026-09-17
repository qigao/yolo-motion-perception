from __future__ import annotations

import numpy as np

from .r1_e1_probe import MulticlassRidgeProbe, fit_multiclass_ridge


RIDGE_REGULARIZATION = 1e-6


def causal_mean_pool(
    states: np.ndarray,
    *,
    window: int,
    decision_index: int | None = None,
) -> np.ndarray:
    values = _validated_states(states)
    decision = _validated_window(
        time_count=values.shape[0],
        window=window,
        decision_index=decision_index,
    )
    start = decision - window + 1
    pooled = np.mean(values[start : decision + 1], axis=0, dtype=np.float64)
    return _readonly_copy(pooled)


def causal_mean_pool_batch(
    trajectories: np.ndarray,
    *,
    window: int,
    decision_index: int | None = None,
) -> np.ndarray:
    values = _validated_trajectories(trajectories)
    decision = _validated_window(
        time_count=values.shape[1],
        window=window,
        decision_index=decision_index,
    )
    start = decision - window + 1
    pooled = np.mean(values[:, start : decision + 1, :], axis=1, dtype=np.float64)
    return _readonly_copy(pooled)


def fit_instant_multiclass(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> MulticlassRidgeProbe:
    return fit_multiclass_ridge(
        states,
        labels,
        class_count=class_count,
        regularization=RIDGE_REGULARIZATION,
    )


def fit_temporal_mean_multiclass(
    trajectories: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    window: int,
    decision_index: int | None = None,
) -> MulticlassRidgeProbe:
    pooled = causal_mean_pool_batch(
        trajectories,
        window=window,
        decision_index=decision_index,
    )
    return fit_multiclass_ridge(
        pooled,
        labels,
        class_count=class_count,
        regularization=RIDGE_REGULARIZATION,
    )


def _validated_states(states: object) -> np.ndarray:
    try:
        values = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be float64-compatible") from exc
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("states must have shape (time > 0, features > 0)")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must contain only finite values")
    return values


def _validated_trajectories(trajectories: object) -> np.ndarray:
    try:
        values = np.asarray(trajectories, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectories must be float64-compatible") from exc
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError(
            "trajectories must have shape (samples > 0, time > 0, features > 0)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectories must contain only finite values")
    return values


def _validated_window(
    *,
    time_count: int,
    window: object,
    decision_index: object | None,
) -> int:
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive Python integer")
    decision = time_count - 1 if decision_index is None else decision_index
    if type(decision) is not int or decision < 0 or decision >= time_count:
        raise ValueError("decision_index must identify an existing timestep")
    if window > decision + 1:
        raise ValueError("window cannot extend before the start of the trajectory")
    return decision


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied
