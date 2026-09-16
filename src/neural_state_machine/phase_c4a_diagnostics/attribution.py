"""D2 target structure, projection, and supervised-reference diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import freeze_float64


_REGISTERED_PENALTY = 1e-6


@dataclass(frozen=True, slots=True)
class CorrelationStat:
    count: int
    observer_mean: float | None
    target_mean: float | None
    correlation: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CategoryMean:
    category: int | str
    count: int
    target_mean: float


@dataclass(frozen=True, slots=True)
class LagCorrelation:
    lag: int
    stat: CorrelationStat


@dataclass(frozen=True, slots=True)
class TargetProjection:
    parallel: np.ndarray
    residual: np.ndarray
    parallel_ratio: float
    residual_ratio: float
    fit_residual_norm: float
    parallel_target_correlation: float | None
    correlation_reason: str | None


@dataclass(frozen=True, slots=True)
class SupervisedReferenceFit:
    weights: np.ndarray
    row_count: int
    feature_size: int
    augmented_rank: int
    residual_norm: float
    singular_values: np.ndarray
    penalty: float


@dataclass(frozen=True, slots=True)
class WeightAlignment:
    global_cosine: float | None
    action_cosines: tuple[float | None, float | None]
    norm_ratio: float | None
    action_norm_ratios: tuple[float | None, float | None]
    supervised_projection_magnitude: float | None
    c4_norm: float
    supervised_norm: float
    reason: str | None


def _vector(value: object, name: str, *, length: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-compatible") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} length mismatch")
    return array


def _finite_float_vector(value: object, name: str, *, length: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be float64-compatible") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} length mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_vector(value: object, name: str, *, length: int) -> np.ndarray:
    array = _vector(value, name, length=length)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return np.asarray(array, dtype=np.int64)


def _correlation(observer: object, target: object) -> CorrelationStat:
    left = np.asarray(observer, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise ValueError("correlation inputs must be equal-length vectors")
    count = int(left.size)
    if count == 0:
        return CorrelationStat(0, None, None, None, "insufficient observations")
    observer_mean = float(np.mean(left))
    target_mean = float(np.mean(right))
    if count < 2:
        return CorrelationStat(
            count,
            observer_mean,
            target_mean,
            None,
            "insufficient observations",
        )
    centered_left = left - observer_mean
    centered_right = right - target_mean
    left_norm = float(np.linalg.norm(centered_left))
    right_norm = float(np.linalg.norm(centered_right))
    if left_norm == 0.0:
        return CorrelationStat(
            count,
            observer_mean,
            target_mean,
            None,
            "zero variance in observer",
        )
    if right_norm == 0.0:
        return CorrelationStat(
            count,
            observer_mean,
            target_mean,
            None,
            "zero variance in target",
        )
    correlation = float(np.dot(centered_left, centered_right) / (left_norm * right_norm))
    correlation = max(-1.0, min(1.0, correlation))
    return CorrelationStat(count, observer_mean, target_mean, correlation, None)


def _categorical_means(categories: np.ndarray, target: np.ndarray) -> tuple[CategoryMean, ...]:
    rows: list[CategoryMean] = []
    for category in np.unique(categories):
        mask = categories == category
        rows.append(
            CategoryMean(
                int(category),
                int(np.count_nonzero(mask)),
                float(np.mean(target[mask])),
            )
        )
    return tuple(rows)


def _arrival_bucket_means(arrivals: np.ndarray, target: np.ndarray) -> tuple[CategoryMean, ...]:
    buckets = (
        ("no_arrival", arrivals == 0),
        ("one_source", arrivals == 1),
        ("collision", arrivals >= 2),
    )
    rows: list[CategoryMean] = []
    for name, mask in buckets:
        if np.any(mask):
            rows.append(
                CategoryMean(
                    name,
                    int(np.count_nonzero(mask)),
                    float(np.mean(target[mask])),
                )
            )
    return tuple(rows)


def _lag_rows(
    values: np.ndarray,
    target: np.ndarray,
    *,
    direction: str,
) -> tuple[LagCorrelation, ...]:
    rows: list[LagCorrelation] = []
    size = int(values.size)
    for lag in range(1, 11):
        if lag >= size:
            stat = _correlation(np.empty(0), np.empty(0))
        elif direction == "previous":
            stat = _correlation(values[:-lag], target[lag:])
        elif direction == "future":
            stat = _correlation(values[lag:], target[:-lag])
        else:
            raise ValueError("direction must be previous or future")
        rows.append(LagCorrelation(lag, stat))
    return tuple(rows)


def associate_target(
    target: object,
    *,
    real_decision_count: int,
    actions: object,
    correct_actions: object,
    cue_delays: object,
    hidden_delays: object,
    normal_rewards: object,
    donor_rewards: object,
    arrival_counts: object,
) -> dict[str, object]:
    """Associate aggregate feedback with fixed observer-only variables."""
    target_array = _finite_float_vector(target, "target")
    if type(real_decision_count) is not int or not (0 < real_decision_count <= target_array.size):
        raise ValueError("real_decision_count must be a positive in-range integer")
    n = real_decision_count
    actions_array = _integer_vector(actions, "actions", length=n)
    correct_array = _integer_vector(correct_actions, "correct_actions", length=n)
    cue_array = _integer_vector(cue_delays, "cue_delays", length=n)
    hidden_array = _integer_vector(hidden_delays, "hidden_delays", length=n)
    normal_array = _finite_float_vector(normal_rewards, "normal_rewards", length=n)
    donor_array = _finite_float_vector(donor_rewards, "donor_rewards", length=n)
    arrivals = _integer_vector(arrival_counts, "arrival_counts", length=target_array.size)
    if np.any(arrivals < 0):
        raise ValueError("arrival_counts must be non-negative")
    if np.any((actions_array < 0) | (actions_array > 1)):
        raise ValueError("actions must be binary action indices")
    if np.any((correct_array < 0) | (correct_array > 1)):
        raise ValueError("correct_actions must be binary action indices")

    real_target = target_array[:n]
    correctness = (actions_array == correct_array).astype(np.float64)
    block_position = np.arange(n, dtype=np.int64) % 10
    real_arrivals = arrivals[:n]
    drain_target = freeze_float64(target_array[n:])
    drain_arrivals = arrivals[n:]

    scalar = {
        "current_action": _correlation(actions_array, real_target),
        "correct_action": _correlation(correct_array, real_target),
        "action_correctness": _correlation(correctness, real_target),
        "normal_latent_reward": _correlation(normal_array, real_target),
        "shuffled_donor_reward": _correlation(donor_array, real_target),
    }
    categorical = {
        "cue_delay": _categorical_means(cue_array, real_target),
        "hidden_delay": _categorical_means(hidden_array, real_target),
        "block_position": _categorical_means(block_position, real_target),
        "arrival_bucket": _arrival_bucket_means(real_arrivals, real_target),
    }
    lagged = {
        "correct_action_previous": _lag_rows(correct_array, real_target, direction="previous"),
        "correct_action_future": _lag_rows(correct_array, real_target, direction="future"),
        "action_correctness_previous": _lag_rows(correctness, real_target, direction="previous"),
        "action_correctness_future": _lag_rows(correctness, real_target, direction="future"),
    }
    return {
        "real_decision_count": n,
        "drain_count": int(target_array.size - n),
        "scalar": scalar,
        "categorical": categorical,
        "lagged": lagged,
        "drain_target": drain_target,
        "drain_arrival_bucket": _arrival_bucket_means(drain_arrivals, target_array[n:])
        if drain_arrivals.size
        else (),
    }


def project_target(design: object, target: object) -> TargetProjection:
    """Project an aggregate target into col(Z) using a stable compact SVD."""
    try:
        matrix = np.asarray(design, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("design must be float64-compatible") from exc
    target_array = _finite_float_vector(target, "target")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design must be a non-empty two-dimensional matrix")
    if matrix.shape[0] != target_array.size:
        raise ValueError("design and target row counts must match")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")

    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    sigma_max = float(singular[0]) if singular.size else 0.0
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * sigma_max
    retained = singular > tolerance
    if np.any(retained):
        basis = u[:, retained]
        parallel = basis @ (basis.T @ target_array)
    else:
        parallel = np.zeros_like(target_array)
    residual = target_array - parallel
    target_norm = float(np.linalg.norm(target_array))
    parallel_norm = float(np.linalg.norm(parallel))
    residual_norm = float(np.linalg.norm(residual))
    if target_norm == 0.0:
        parallel_ratio = 0.0
        residual_ratio = 0.0
    else:
        parallel_ratio = parallel_norm / target_norm
        residual_ratio = residual_norm / target_norm
    correlation = _correlation(parallel, target_array)
    return TargetProjection(
        parallel=freeze_float64(parallel),
        residual=freeze_float64(residual),
        parallel_ratio=float(parallel_ratio),
        residual_ratio=float(residual_ratio),
        fit_residual_norm=residual_norm,
        parallel_target_correlation=correlation.correlation,
        correlation_reason=correlation.reason,
    )


def fit_supervised_reference(
    features: object,
    correct_actions: object,
) -> SupervisedReferenceFit:
    """Fit the observer-only supervised ridge reference with fixed 1e-6 penalty."""
    try:
        phi = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("features must be float64-compatible") from exc
    if phi.ndim != 2 or phi.shape[0] == 0 or phi.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(phi)):
        raise ValueError("features must contain only finite values")
    labels = _integer_vector(correct_actions, "correct_actions", length=phi.shape[0])
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("correct_actions must contain only 0/1")

    target = np.full((phi.shape[0], 2), -1.0, dtype=np.float64)
    target[np.arange(phi.shape[0]), labels] = 1.0
    identity = np.eye(phi.shape[1], dtype=np.float64)
    aug_x = np.vstack((phi, math.sqrt(_REGISTERED_PENALTY) * identity))
    aug_y = np.vstack((target, np.zeros((phi.shape[1], 2), dtype=np.float64)))
    coefficients, _, rank, singular = np.linalg.lstsq(aug_x, aug_y, rcond=None)
    residual_norm = float(np.linalg.norm(aug_x @ coefficients - aug_y))
    return SupervisedReferenceFit(
        weights=freeze_float64(coefficients.T),
        row_count=int(phi.shape[0]),
        feature_size=int(phi.shape[1]),
        augmented_rank=int(rank),
        residual_norm=residual_norm,
        singular_values=freeze_float64(singular),
        penalty=_REGISTERED_PENALTY,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = float(np.dot(left.reshape(-1), right.reshape(-1)) / (left_norm * right_norm))
    return max(-1.0, min(1.0, value))


def compare_weight_alignment(
    c4_weights: object,
    supervised_weights: object,
) -> WeightAlignment:
    """Compare fitted readouts without changing or refitting either model."""
    try:
        c4 = np.asarray(c4_weights, dtype=np.float64)
        supervised = np.asarray(supervised_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be float64-compatible") from exc
    if c4.ndim != 2 or c4.shape[0] != 2 or c4.shape != supervised.shape:
        raise ValueError("weights must be matching two-action matrices")
    if not np.all(np.isfinite(c4)) or not np.all(np.isfinite(supervised)):
        raise ValueError("weights must contain only finite values")

    c4_norm = float(np.linalg.norm(c4))
    supervised_norm = float(np.linalg.norm(supervised))
    global_cosine = _cosine(c4, supervised)
    action_cosines = (
        _cosine(c4[0], supervised[0]),
        _cosine(c4[1], supervised[1]),
    )
    action_norm_ratios: list[float | None] = []
    for action in range(2):
        reference_norm = float(np.linalg.norm(supervised[action]))
        action_norm_ratios.append(
            float(np.linalg.norm(c4[action]) / reference_norm) if reference_norm else None
        )
    norm_ratio = c4_norm / supervised_norm if supervised_norm else None
    projection = (
        abs(float(np.dot(c4.reshape(-1), supervised.reshape(-1)))) / supervised_norm
        if supervised_norm
        else None
    )
    reason = None
    if supervised_norm == 0.0:
        reason = "zero norm in supervised reference"
    elif c4_norm == 0.0:
        reason = "zero norm in C4 readout"
    return WeightAlignment(
        global_cosine=global_cosine,
        action_cosines=action_cosines,
        norm_ratio=norm_ratio,
        action_norm_ratios=(action_norm_ratios[0], action_norm_ratios[1]),
        supervised_projection_magnitude=projection,
        c4_norm=c4_norm,
        supervised_norm=supervised_norm,
        reason=reason,
    )
