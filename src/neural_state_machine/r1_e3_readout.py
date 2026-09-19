from __future__ import annotations

import numpy as np

from .r1_e1_probe import MulticlassRidgeProbe, fit_multiclass_ridge


RIDGE_REGULARIZATION = 1e-6
REGISTERED_BIN_COUNT = 20
TEMPORAL_START_BIN = 16
TEMPORAL_STOP_BIN = 20


def fit_frame_only(
    observations: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> MulticlassRidgeProbe:
    values = _validated_sequences(observations, name="observations")
    return fit_multiclass_ridge(
        values[:, 19, :],
        labels,
        class_count=class_count,
        regularization=RIDGE_REGULARIZATION,
    )


def fit_reservoir_instantaneous(
    trajectories: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> MulticlassRidgeProbe:
    values = _validated_sequences(trajectories, name="trajectories")
    return fit_multiclass_ridge(
        values[:, 19, :],
        labels,
        class_count=class_count,
        regularization=RIDGE_REGULARIZATION,
    )


def fit_reservoir_temporal_mean(
    trajectories: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
) -> MulticlassRidgeProbe:
    pooled = temporal_mean_last_four(trajectories)
    return fit_multiclass_ridge(
        pooled,
        labels,
        class_count=class_count,
        regularization=RIDGE_REGULARIZATION,
    )


def temporal_mean_last_four(trajectories: np.ndarray) -> np.ndarray:
    values = _validated_sequences(trajectories, name="trajectories")
    pooled = np.mean(
        values[:, TEMPORAL_START_BIN:TEMPORAL_STOP_BIN, :],
        axis=1,
        dtype=np.float64,
    )
    return _readonly_copy(pooled)


def _validated_sequences(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (samples, {REGISTERED_BIN_COUNT}, features)"
        )
    if array.shape[0] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} must contain samples and features")
    if array.shape[1] != REGISTERED_BIN_COUNT:
        raise ValueError(
            f"{name} must contain exactly {REGISTERED_BIN_COUNT} registered bins"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied
