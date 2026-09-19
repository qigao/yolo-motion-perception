from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


DELAYS = (1, 2, 5, 10, 15)
RIDGE_REGULARIZATION = 1e-6
TARGET_CHANNELS = (0, 1, 2, 3)
_COEFFICIENT_DIGEST_VERSION = b"r1-e3m-ridge-v1\0"
_PREDICTION_DIGEST_VERSION = b"r1-e3m-regression-prediction-v1\0"


@dataclass(frozen=True)
class RegressionMetrics:
    sample_count: int
    mean_r2: float
    r2_per_channel: tuple[float, float, float, float]
    mse: float

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        values = (self.mean_r2, self.mse, *self.r2_per_channel)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("regression metrics must be finite")


@dataclass(frozen=True)
class MultiTargetRidgeProbe:
    coefficients: np.ndarray
    bias: np.ndarray
    regularization: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[0] != 4:
            raise ValueError(
                "coefficients must have shape (4, feature_count)"
            )
        if coefficients.shape[1] == 0:
            raise ValueError("probe must contain at least one feature")
        if bias.shape != (4,):
            raise ValueError("bias must have shape (4,)")
        if not np.isfinite(coefficients).all() or not np.isfinite(bias).all():
            raise ValueError("probe parameters must be finite")
        if (
            not math.isfinite(float(self.regularization))
            or float(self.regularization) != RIDGE_REGULARIZATION
        ):
            raise ValueError("probe regularization must be exactly 1e-6")

        coefficients = np.array(
            coefficients,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        bias = np.array(bias, dtype=np.float64, copy=True, order="C")
        coefficients.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(
            self,
            "regularization",
            float(self.regularization),
        )

    @property
    def feature_count(self) -> int:
        return int(self.coefficients.shape[1])

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _feature_matrix(
            features,
            expected_features=self.feature_count,
        )
        predictions = values @ self.coefficients.T + self.bias
        return _readonly(predictions)

    def coefficient_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_COEFFICIENT_DIGEST_VERSION)
        _digest_array(digest, self.coefficients)
        _digest_array(digest, self.bias)
        digest.update(repr(self.regularization).encode("ascii"))
        return digest.hexdigest()


def delayed_geometry_targets(
    tensors: np.ndarray,
    delay: int,
) -> np.ndarray:
    values = _window_tensor_batch(tensors)
    _registered_delay(delay)
    target_bin = 19 - delay
    targets = values[:, target_bin, :4]
    return _readonly(targets)


def fit_instantaneous_delay_probe(
    tensors: np.ndarray,
    delay: int,
) -> MultiTargetRidgeProbe:
    values = _window_tensor_batch(tensors)
    targets = delayed_geometry_targets(values, delay)
    return _fit_multi_target_ridge(values[:, 19, :], targets)


def fit_reservoir_delay_probe(
    final_states: np.ndarray,
    tensors: np.ndarray,
    delay: int,
) -> MultiTargetRidgeProbe:
    values = _window_tensor_batch(tensors)
    states = _feature_matrix(final_states)
    if states.shape[0] != values.shape[0]:
        raise ValueError(
            "final_states and tensors must have the same sample count"
        )
    targets = delayed_geometry_targets(values, delay)
    return _fit_multi_target_ridge(states, targets)


def evaluate_regression(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> RegressionMetrics:
    expected = _target_matrix(targets)
    observed = _target_matrix(predictions)
    if expected.shape != observed.shape:
        raise ValueError("targets and predictions must have matching shape")

    errors = expected - observed
    mse = float(np.mean(np.square(errors), dtype=np.float64))
    r2_values: list[float] = []
    for channel in range(4):
        truth = expected[:, channel]
        pred = observed[:, channel]
        residual = float(np.sum(np.square(truth - pred), dtype=np.float64))
        centered = truth - float(np.mean(truth, dtype=np.float64))
        total = float(np.sum(np.square(centered), dtype=np.float64))
        if total == 0.0:
            r2 = 1.0 if residual == 0.0 else 0.0
        else:
            r2 = 1.0 - residual / total
        r2_values.append(float(r2))

    return RegressionMetrics(
        sample_count=int(expected.shape[0]),
        mean_r2=float(np.mean(np.asarray(r2_values), dtype=np.float64)),
        r2_per_channel=tuple(r2_values),
        mse=mse,
    )


def regression_prediction_digest(predictions: np.ndarray) -> str:
    values = _target_matrix(predictions)
    digest = hashlib.sha256()
    digest.update(_PREDICTION_DIGEST_VERSION)
    _digest_array(digest, values)
    return digest.hexdigest()


def _fit_multi_target_ridge(
    features: np.ndarray,
    targets: np.ndarray,
) -> MultiTargetRidgeProbe:
    values = _feature_matrix(features)
    expected = _target_matrix(targets)
    if values.shape[0] != expected.shape[0]:
        raise ValueError("features and targets must have the same sample count")

    design = np.column_stack(
        (values, np.ones(values.shape[0], dtype=np.float64))
    )
    feature_count = values.shape[1]
    penalty = np.zeros(
        (feature_count, feature_count + 1),
        dtype=np.float64,
    )
    penalty[:, :feature_count] = math.sqrt(
        RIDGE_REGULARIZATION
    ) * np.eye(feature_count, dtype=np.float64)
    augmented_design = np.vstack((design, penalty))
    augmented_target = np.vstack(
        (
            expected,
            np.zeros((feature_count, 4), dtype=np.float64),
        )
    )
    try:
        parameters, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_target,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("R1-E3M Ridge fit failed") from exc

    return MultiTargetRidgeProbe(
        coefficients=parameters[:-1].T,
        bias=parameters[-1],
        regularization=RIDGE_REGULARIZATION,
    )


def _registered_delay(delay: object) -> int:
    if type(delay) is not int or delay not in DELAYS:
        raise ValueError(
            f"delay must be one of the registered delays {DELAYS}"
        )
    return delay


def _window_tensor_batch(values: object) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("tensors must be NumPy-compatible") from exc
    if array.dtype != np.float64:
        raise ValueError("tensors must use float64")
    if array.ndim != 3 or array.shape[1:] != (20, 6):
        raise ValueError("tensors must have shape (samples, 20, 6)")
    if array.shape[0] == 0:
        raise ValueError("tensors must contain samples")
    if not np.isfinite(array).all():
        raise ValueError("tensors must contain only finite values")
    return array


def _feature_matrix(
    values: object,
    *,
    expected_features: int | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be float64-compatible") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a non-empty rank-two matrix")
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError(
            f"features must have {expected_features} columns"
        )
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def _target_matrix(values: object) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets must be float64-compatible") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 4:
        raise ValueError("targets must have shape (samples, 4)")
    if not np.isfinite(array).all():
        raise ValueError("targets must contain only finite values")
    return array


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _digest_array(digest: object, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
