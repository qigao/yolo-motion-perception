from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .r1_e1_probe import RidgeDiagnostics


RIDGE_REGULARIZATION = 1e-6
_PROBE_DIGEST_VERSION = b"r1-e3m-multioutput-ridge-v1\0"
_R2_DEGENERATE_SST = 1e-12


@dataclass(frozen=True)
class MultiOutputRidgeProbe:
    coefficients: np.ndarray
    bias: np.ndarray
    diagnostics: RidgeDiagnostics

    def __post_init__(self) -> None:
        coefficients = _matrix(
            self.coefficients,
            name="coefficients",
            require_samples=True,
        )
        bias = _vector(self.bias, name="bias")
        if bias.shape != (coefficients.shape[0],):
            raise ValueError("bias must have one value per output channel")
        if not isinstance(self.diagnostics, RidgeDiagnostics):
            raise ValueError("diagnostics must be RidgeDiagnostics")
        object.__setattr__(self, "coefficients", _readonly(coefficients))
        object.__setattr__(self, "bias", _readonly(bias))

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = _matrix(
            features,
            name="features",
            require_samples=False,
        )
        if values.shape[1] != self.coefficients.shape[1]:
            raise ValueError(
                f"features must have {self.coefficients.shape[1]} columns"
            )
        return _readonly(values @ self.coefficients.T + self.bias)

    def coefficient_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(_PROBE_DIGEST_VERSION)
        _digest_array(digest, self.coefficients)
        _digest_array(digest, self.bias)
        return digest.hexdigest()


@dataclass(frozen=True)
class R2Summary:
    per_channel_r2: np.ndarray
    macro_r2: float
    valid_channel_count: int
    degenerate_channel_count: int

    def __post_init__(self) -> None:
        values = np.asarray(self.per_channel_r2, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("per_channel_r2 must be a non-empty vector")
        if (
            type(self.valid_channel_count) is not int
            or type(self.degenerate_channel_count) is not int
            or self.valid_channel_count < 0
            or self.degenerate_channel_count < 0
            or self.valid_channel_count + self.degenerate_channel_count
            != values.size
        ):
            raise ValueError("R2 channel counts are inconsistent")
        if not math.isfinite(float(self.macro_r2)):
            raise ValueError("macro_r2 must be finite")
        object.__setattr__(self, "per_channel_r2", _readonly(values))
        object.__setattr__(self, "macro_r2", float(self.macro_r2))


def fit_multioutput_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    regularization: float = RIDGE_REGULARIZATION,
) -> MultiOutputRidgeProbe:
    values = _matrix(
        features,
        name="features",
        require_samples=True,
    )
    expected = _matrix(
        targets,
        name="targets",
        require_samples=True,
    )
    if values.shape[0] != expected.shape[0]:
        raise ValueError("features and targets must have the same sample count")

    strength = _regularization(regularization)
    design = np.column_stack(
        (values, np.ones(values.shape[0], dtype=np.float64))
    )
    feature_count = values.shape[1]

    penalty = np.zeros(
        (feature_count, feature_count + 1),
        dtype=np.float64,
    )
    penalty[:, :feature_count] = math.sqrt(strength) * np.eye(
        feature_count,
        dtype=np.float64,
    )
    augmented_design = np.vstack((design, penalty))
    augmented_targets = np.vstack(
        (
            expected,
            np.zeros(
                (feature_count, expected.shape[1]),
                dtype=np.float64,
            ),
        )
    )

    try:
        parameters, *_ = np.linalg.lstsq(
            augmented_design,
            augmented_targets,
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("multi-output ridge least-squares fit failed") from exc

    singular_values = np.linalg.svd(design, compute_uv=False)
    diagnostics = RidgeDiagnostics(
        train_count=design.shape[0],
        design_shape=(design.shape[0], design.shape[1]),
        rank=int(np.linalg.matrix_rank(design)),
        singular_values=tuple(float(value) for value in singular_values),
    )
    return MultiOutputRidgeProbe(
        coefficients=parameters[:-1].T,
        bias=parameters[-1],
        diagnostics=diagnostics,
    )


def r2_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> R2Summary:
    observed = _matrix(
        actual,
        name="actual",
        require_samples=True,
    )
    estimate = _matrix(
        predicted,
        name="predicted",
        require_samples=True,
    )
    if observed.shape != estimate.shape:
        raise ValueError("actual and predicted must have the same shape")

    centered = observed - np.mean(observed, axis=0, dtype=np.float64)
    sst = np.sum(centered * centered, axis=0, dtype=np.float64)
    residual = observed - estimate
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)

    scores = np.full(observed.shape[1], np.nan, dtype=np.float64)
    valid = sst > _R2_DEGENERATE_SST
    scores[valid] = 1.0 - sse[valid] / sst[valid]
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("all R2 output channels are degenerate")
    macro = float(np.mean(scores[valid], dtype=np.float64))
    return R2Summary(
        per_channel_r2=scores,
        macro_r2=macro,
        valid_channel_count=valid_count,
        degenerate_channel_count=int(scores.size - valid_count),
    )


def _matrix(
    values: object,
    *,
    name: str,
    require_samples: bool,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank two")
    if array.shape[1] == 0 or (require_samples and array.shape[0] == 0):
        raise ValueError(f"{name} must contain samples and features")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _vector(values: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _regularization(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("regularization must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("regularization must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("regularization must be finite and positive")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=np.float64, copy=True, order="C")
    copied.flags.writeable = False
    return copied


def _digest_array(digest: object, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
