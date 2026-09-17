from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


_BINARY_DIGEST_VERSION = b"r1-e1-binary-ridge-v1\0"
_MULTICLASS_DIGEST_VERSION = b"r1-e1-multiclass-ridge-v1\0"
_PREDICTION_DIGEST_VERSION = b"r1-e1-prediction-v1\0"


@dataclass(frozen=True)
class RidgeDiagnostics:
    train_count: int
    design_shape: tuple[int, int]
    rank: int
    singular_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.train_count) is not int or self.train_count <= 0:
            raise ValueError("train_count must be a positive Python integer")
        if (
            type(self.design_shape) is not tuple
            or len(self.design_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.design_shape)
        ):
            raise ValueError("design_shape must contain two positive Python integers")
        if self.design_shape[0] != self.train_count:
            raise ValueError("design_shape sample count must match train_count")
        if (
            type(self.rank) is not int
            or self.rank < 0
            or self.rank > min(self.design_shape)
        ):
            raise ValueError("rank must be valid for design_shape")
        try:
            singular_values = tuple(float(value) for value in self.singular_values)
        except (TypeError, ValueError) as exc:
            raise ValueError("singular_values must be finite non-negative values") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in singular_values):
            raise ValueError("singular_values must be finite non-negative values")
        object.__setattr__(self, "singular_values", singular_values)


@dataclass(frozen=True)
class BinaryRidgeProbe:
    coefficients: np.ndarray
    bias: float
    diagnostics: RidgeDiagnostics

    def __post_init__(self) -> None:
        coefficients = _validated_vector(self.coefficients, "coefficients")
        if isinstance(self.bias, bool):
            raise ValueError("bias must be a finite float64-compatible value")
        try:
            bias = float(self.bias)
        except (TypeError, ValueError) as exc:
            raise ValueError("bias must be a finite float64-compatible value") from exc
        if not math.isfinite(bias):
            raise ValueError("bias must be a finite float64-compatible value")
        if not isinstance(self.diagnostics, RidgeDiagnostics):
            raise ValueError("diagnostics must be RidgeDiagnostics")
        object.__setattr__(self, "coefficients", _readonly_copy(coefficients, dtype=np.float64))
        object.__setattr__(self, "bias", bias)

    def predict_scores(self, states: np.ndarray) -> np.ndarray:
        values = _validated_state_matrix(
            states,
            expected_features=self.coefficients.size,
            require_samples=False,
        )
        return _readonly_copy(values @ self.coefficients + self.bias, dtype=np.float64)

    def predict(self, states: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(states)
        choices = np.where(scores > 0.0, 1, 0)
        return _readonly_copy(choices, dtype=np.int64)

    def coefficient_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_BINARY_DIGEST_VERSION)
        _digest_array(digest, self.coefficients)
        _digest_array(digest, np.asarray(self.bias, dtype=np.float64))
        return digest.hexdigest()


@dataclass(frozen=True)
class MulticlassRidgeProbe:
    coefficients: np.ndarray
    bias: np.ndarray
    diagnostics: RidgeDiagnostics

    def __post_init__(self) -> None:
        try:
            coefficients = np.asarray(self.coefficients, dtype=np.float64)
            bias = np.asarray(self.bias, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe parameters must be float64-compatible") from exc
        if (
            coefficients.ndim != 2
            or coefficients.shape[0] < 2
            or coefficients.shape[1] == 0
        ):
            raise ValueError("coefficients must have shape (class_count >= 2, features > 0)")
        if bias.ndim != 1 or bias.shape != (coefficients.shape[0],):
            raise ValueError("bias must have one value per class")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(bias)):
            raise ValueError("probe parameters must contain only finite values")
        if not isinstance(self.diagnostics, RidgeDiagnostics):
            raise ValueError("diagnostics must be RidgeDiagnostics")
        object.__setattr__(
            self,
            "coefficients",
            _readonly_copy(coefficients, dtype=np.float64),
        )
        object.__setattr__(self, "bias", _readonly_copy(bias, dtype=np.float64))

    def predict_scores(self, states: np.ndarray) -> np.ndarray:
        values = _validated_state_matrix(
            states,
            expected_features=self.coefficients.shape[1],
            require_samples=False,
        )
        scores = values @ self.coefficients.T + self.bias
        return _readonly_copy(scores, dtype=np.float64)

    def predict(self, states: np.ndarray) -> np.ndarray:
        choices = np.argmax(self.predict_scores(states), axis=1)
        return _readonly_copy(choices, dtype=np.int64)

    def coefficient_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_MULTICLASS_DIGEST_VERSION)
        _digest_array(digest, self.coefficients)
        _digest_array(digest, self.bias)
        return digest.hexdigest()


def fit_binary_ridge(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float = 1e-6,
) -> BinaryRidgeProbe:
    values = _validated_state_matrix(states, require_samples=True)
    binary_labels = _validated_integer_labels(
        labels,
        sample_count=values.shape[0],
        class_count=2,
        require_all_classes=True,
    )
    strength = _validated_regularization(regularization)
    design, augmented_design = _augmented_design(values, strength)
    target = np.where(binary_labels == 0, -1.0, 1.0)
    augmented_target = np.concatenate((target, np.zeros(values.shape[1], dtype=np.float64)))
    try:
        parameters, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_target,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("binary ridge least-squares fit failed") from exc
    return BinaryRidgeProbe(
        coefficients=parameters[:-1],
        bias=float(parameters[-1]),
        diagnostics=_ridge_diagnostics(design),
    )


def fit_multiclass_ridge(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    regularization: float = 1e-6,
) -> MulticlassRidgeProbe:
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class_count must be a Python integer of at least two")
    values = _validated_state_matrix(states, require_samples=True)
    class_labels = _validated_integer_labels(
        labels,
        sample_count=values.shape[0],
        class_count=class_count,
        require_all_classes=True,
    )
    strength = _validated_regularization(regularization)
    design, augmented_design = _augmented_design(values, strength)
    target = np.zeros((values.shape[0], class_count), dtype=np.float64)
    target[np.arange(values.shape[0]), class_labels] = 1.0
    augmented_target = np.vstack(
        (target, np.zeros((values.shape[1], class_count), dtype=np.float64))
    )
    try:
        parameters, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_target,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("multiclass ridge least-squares fit failed") from exc
    return MulticlassRidgeProbe(
        coefficients=parameters[:-1].T,
        bias=parameters[-1],
        diagnostics=_ridge_diagnostics(design),
    )


def prediction_digest(predictions: np.ndarray) -> str:
    raw = np.asarray(predictions)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("predictions must be a non-empty rank-one array")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("predictions must contain integer class indices")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("predictions must contain non-negative class indices")
    digest = hashlib.sha256()
    digest.update(_PREDICTION_DIGEST_VERSION)
    _digest_array(digest, values)
    return digest.hexdigest()


def _validated_vector(values: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty rank-one array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_state_matrix(
    states: object,
    *,
    expected_features: int | None = None,
    require_samples: bool,
) -> np.ndarray:
    try:
        values = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be float64-compatible") from exc
    if values.ndim != 2:
        raise ValueError("states must be rank two")
    if values.shape[1] == 0 or (require_samples and values.shape[0] == 0):
        raise ValueError("states must contain samples and features")
    if expected_features is not None and values.shape[1] != expected_features:
        raise ValueError(f"states must have {expected_features} features")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must contain only finite values")
    return values


def _validated_integer_labels(
    labels: object,
    *,
    sample_count: int,
    class_count: int,
    require_all_classes: bool,
) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.shape[0] != sample_count:
        raise ValueError("labels must be rank one and match the sample count")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError("labels must contain integer class indices")
    values = np.asarray(raw, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= class_count):
        raise ValueError("labels contain an out-of-range class index")
    if require_all_classes and set(values.tolist()) != set(range(class_count)):
        raise ValueError("labels must contain every class")
    return values


def _validated_regularization(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("regularization must be finite and positive")
    try:
        regularization = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regularization must be finite and positive") from exc
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    return regularization


def _augmented_design(
    states: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((states, np.ones(states.shape[0], dtype=np.float64)))
    feature_count = states.shape[1]
    penalty = np.zeros((feature_count, feature_count + 1), dtype=np.float64)
    penalty[:, :feature_count] = math.sqrt(regularization) * np.eye(
        feature_count,
        dtype=np.float64,
    )
    return design, np.vstack((design, penalty))


def _ridge_diagnostics(design: np.ndarray) -> RidgeDiagnostics:
    singular_values = np.linalg.svd(design, compute_uv=False)
    return RidgeDiagnostics(
        train_count=design.shape[0],
        design_shape=(design.shape[0], design.shape[1]),
        rank=int(np.linalg.matrix_rank(design)),
        singular_values=tuple(float(value) for value in singular_values),
    )


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _digest_array(digest: object, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
