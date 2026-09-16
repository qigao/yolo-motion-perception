"""Anonymous batch operator construction and fixed ridge for Phase C4-A."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .phase_c4_delay_model import (
    DecisionCreditRow,
    DelayLaw,
    build_marginalized_features,
)


C4_RIDGE_REGULARIZATION = 1e-6


@dataclass(frozen=True, slots=True)
class BatchProbeFit:
    """Immutable result of the fixed C4-A anonymous operator fit."""

    weights: np.ndarray
    row_count: int
    column_count: int
    augmented_rank: int
    residual_norm: float
    singular_values: np.ndarray
    penalty: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        singular_values = np.asarray(self.singular_values, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] < 2 or weights.shape[1] == 0:
            raise ValueError("weights must be a non-empty action-by-feature matrix")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ValueError("row_count must be a positive integer")
        if type(self.column_count) is not int or self.column_count != weights.size:
            raise ValueError("column_count must equal the flattened weight dimension")
        if type(self.augmented_rank) is not int or not (
            0 < self.augmented_rank <= self.column_count
        ):
            raise ValueError("augmented_rank must be a valid positive rank")
        if isinstance(self.residual_norm, bool):
            raise ValueError("residual_norm must be finite and non-negative")
        residual_norm = float(self.residual_norm)
        if not math.isfinite(residual_norm) or residual_norm < 0.0:
            raise ValueError("residual_norm must be finite and non-negative")
        if singular_values.ndim != 1 or singular_values.size != self.column_count:
            raise ValueError("singular_values must match the fitted column count")
        if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
            raise ValueError("singular_values must be finite and non-negative")
        if self.penalty != C4_RIDGE_REGULARIZATION:
            raise ValueError("penalty must match the registered C4 ridge penalty")

        frozen_weights = np.ascontiguousarray(weights, dtype=np.float64).copy()
        frozen_singular = np.ascontiguousarray(
            singular_values, dtype=np.float64
        ).copy()
        frozen_weights.flags.writeable = False
        frozen_singular.flags.writeable = False
        object.__setattr__(self, "weights", frozen_weights)
        object.__setattr__(self, "residual_norm", residual_norm)
        object.__setattr__(self, "singular_values", frozen_singular)


def build_batch_design(
    rows: object,
    feedback: object,
    action_count: int,
    law: DelayLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical C-order Z rows paired only with anonymous scalar feedback."""
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a non-empty tuple")
    if any(not isinstance(row, DecisionCreditRow) for row in rows):
        raise ValueError("rows must contain only DecisionCreditRow values")
    if type(action_count) is not int or action_count < 2:
        raise ValueError("action_count must be an integer of at least two")
    if not isinstance(law, DelayLaw):
        raise ValueError("law must be a DelayLaw")

    indices = tuple(row.decision_index for row in rows)
    if indices != tuple(range(len(rows))):
        raise ValueError("rows must cover consecutive decision indices from zero")
    feature_size = rows[0].feature.size
    if any(row.feature.size != feature_size for row in rows):
        raise ValueError("all rows must have the same feature size")
    if any(row.action_index >= action_count for row in rows):
        raise ValueError("row action_index must be in range")

    try:
        target = np.asarray(feedback, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("feedback must be float64-compatible") from exc
    if target.ndim != 1 or target.size == 0:
        raise ValueError("feedback must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(target)):
        raise ValueError("feedback must contain only finite values")

    design = np.empty(
        (target.size, action_count * feature_size),
        dtype=np.float64,
    )
    for feedback_step in range(target.size):
        marginalized = build_marginalized_features(
            rows,
            feedback_step=feedback_step,
            action_count=action_count,
            law=law,
        )
        design[feedback_step] = marginalized.expected_feature.reshape(
            -1, order="C"
        )

    if not np.all(np.isfinite(design)):
        raise ValueError("batch design construction overflowed float64")
    frozen_design = np.ascontiguousarray(design, dtype=np.float64)
    frozen_target = np.ascontiguousarray(target, dtype=np.float64).copy()
    frozen_design.flags.writeable = False
    frozen_target.flags.writeable = False
    return frozen_design, frozen_target


def fit_anonymous_batch_probe(
    design: object,
    target: object,
    action_count: int,
    feature_size: int,
    penalty: float = C4_RIDGE_REGULARIZATION,
) -> BatchProbeFit:
    """Fit the registered fixed-penalty anonymous observation operator."""
    if type(action_count) is not int or action_count < 2:
        raise ValueError("action_count must be an integer of at least two")
    if type(feature_size) is not int or feature_size <= 0:
        raise ValueError("feature_size must be a positive integer")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise ValueError("penalty must equal the registered C4 ridge penalty")
    resolved_penalty = float(penalty)
    if (
        not math.isfinite(resolved_penalty)
        or resolved_penalty != C4_RIDGE_REGULARIZATION
    ):
        raise ValueError("penalty must equal the registered C4 ridge penalty")

    try:
        design_array = np.asarray(design, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("design and target must be float64-compatible") from exc

    expected_columns = action_count * feature_size
    if (
        design_array.ndim != 2
        or design_array.shape[0] == 0
        or design_array.shape[1] != expected_columns
    ):
        raise ValueError("design shape must match action_count * feature_size")
    if not np.all(np.isfinite(design_array)):
        raise ValueError("design must contain only finite values")
    if target_array.ndim != 1 or target_array.shape[0] != design_array.shape[0]:
        raise ValueError("target must match the design row count")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target must contain only finite values")

    canonical_design = np.ascontiguousarray(design_array, dtype=np.float64)
    canonical_target = np.ascontiguousarray(target_array, dtype=np.float64)
    column_count = canonical_design.shape[1]
    sqrt_penalty = math.sqrt(resolved_penalty)
    regularizer = sqrt_penalty * np.eye(column_count, dtype=np.float64)
    augmented_design = np.vstack((canonical_design, regularizer))
    augmented_target = np.concatenate(
        (canonical_target, np.zeros(column_count, dtype=np.float64))
    )

    coefficients, _, rank, singular_values = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )
    if coefficients.shape != (column_count,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("ridge solution must be a finite coefficient vector")
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("ridge singular values must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = augmented_design @ coefficients - augmented_target
        residual_norm = float(np.linalg.norm(residual))
    if not math.isfinite(residual_norm):
        raise ValueError("ridge residual norm overflowed float64")

    return BatchProbeFit(
        weights=coefficients.reshape(action_count, feature_size),
        row_count=canonical_design.shape[0],
        column_count=column_count,
        augmented_rank=int(rank),
        residual_norm=residual_norm,
        singular_values=singular_values,
        penalty=resolved_penalty,
    )
