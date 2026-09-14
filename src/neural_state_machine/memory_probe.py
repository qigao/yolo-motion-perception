from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True, order="C")
    copied.flags.writeable = False
    return copied


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


@dataclass(frozen=True)
class FittedLinearProbe:
    weights: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        try:
            weights = np.asarray(self.weights, dtype=np.float64)
            bias = float(self.bias)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe parameters must be float64-compatible") from exc
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a non-empty rank-one array")
        if not np.all(np.isfinite(weights)) or not math.isfinite(bias):
            raise ValueError("probe parameters must contain only finite values")
        object.__setattr__(self, "weights", _readonly_copy(weights, dtype=np.float64))
        object.__setattr__(self, "bias", bias)

    def predict(self, states: np.ndarray) -> np.ndarray:
        values = _validated_state_matrix(
            states,
            expected_features=self.weights.size,
            require_samples=False,
        )
        choices = np.where(values @ self.weights + self.bias > 0.0, 1, 0)
        return _readonly_copy(choices, dtype=np.int64)

    def digest(self) -> str:
        values = np.ascontiguousarray(self.weights, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
        digest.update(np.asarray(self.bias, dtype=np.float64).tobytes())
        return digest.hexdigest()


def _validated_regularization(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("regularization must be finite and positive")  # noqa: TRY004
    try:
        regularization = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regularization must be finite and positive") from exc
    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    return regularization


def fit_linear_probe(
    states: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
) -> FittedLinearProbe:
    values = _validated_state_matrix(states, require_samples=True)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != values.shape[0]:
        raise ValueError("labels must be rank one and match the sample count")
    if raw_labels.dtype.kind not in "iu" or raw_labels.dtype.kind == "b":
        raise ValueError("labels must contain integer action indices")
    action_indices = np.asarray(raw_labels, dtype=np.int64)
    if not np.all(np.isin(action_indices, (0, 1))):
        raise ValueError("labels must contain only zero and one")
    if set(action_indices.tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes")
    strength = _validated_regularization(regularization)
    design = np.column_stack((values, np.ones(values.shape[0])))
    penalty = np.diag([strength] * values.shape[1] + [0.0])
    target = np.where(action_indices == 0, -1.0, 1.0)
    try:
        parameters = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("linear probe solve failed") from exc
    return FittedLinearProbe(parameters[:-1], float(parameters[-1]))
