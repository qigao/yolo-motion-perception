"""Anonymous batch operator probe for Phase C4-A."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


C4_RIDGE_REGULARIZATION = 1e-6


@dataclass(frozen=True, slots=True)
class AnonymousBatchRidge:
    """Fixed ridge fit over anonymous marginalized observation rows."""

    weights: np.ndarray
    rank: int
    regularization: float
    observation_count: int

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] < 2 or weights.shape[1] == 0:
            raise ValueError("weights must be a non-empty action-by-feature matrix")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if self.regularization != C4_RIDGE_REGULARIZATION:
            raise ValueError("regularization must match the fixed C4 ridge penalty")
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ValueError("observation_count must be a positive integer")

        copied = np.ascontiguousarray(weights, dtype=np.float64).copy()
        copied.flags.writeable = False
        object.__setattr__(self, "weights", copied)

    def predict_aggregate(self, expected_feature: object) -> float:
        """Apply the fitted matrix to one anonymous marginalized feature row."""
        try:
            feature = np.asarray(expected_feature, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_feature must be float64-compatible") from exc
        if feature.shape != self.weights.shape:
            raise ValueError("expected_feature shape must match fitted weights")
        if not np.all(np.isfinite(feature)):
            raise ValueError("expected_feature must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = float(np.sum(self.weights * feature, dtype=np.float64))
        if not math.isfinite(prediction):
            raise ValueError("aggregate prediction overflowed float64")
        return prediction

    def action_values(self, feature: object) -> np.ndarray:
        """Return greedy-evaluation action values for one learner-owned feature."""
        try:
            values = np.asarray(feature, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature must be float64-compatible") from exc
        if values.ndim != 1 or values.shape[0] != self.weights.shape[1]:
            raise ValueError("feature length must match fitted feature size")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.weights @ values
        if not np.all(np.isfinite(result)):
            raise ValueError("action-value prediction overflowed float64")
        copied = np.ascontiguousarray(result, dtype=np.float64).copy()
        copied.flags.writeable = False
        return copied


def fit_anonymous_batch_ridge(
    expected_features: object,
    aggregate_feedback: object,
) -> AnonymousBatchRidge:
    """Fit the registered C4-A ridge using only Z_t rows and aggregate F_t."""
    try:
        observations = np.asarray(expected_features, dtype=np.float64)
        feedback = np.asarray(aggregate_feedback, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch inputs must be float64-compatible") from exc

    if observations.ndim != 3:
        raise ValueError("expected_features must have shape (rows, actions, features)")
    row_count, action_count, feature_size = observations.shape
    if row_count == 0 or action_count < 2 or feature_size == 0:
        raise ValueError("expected_features dimensions must be non-empty")
    if not np.all(np.isfinite(observations)):
        raise ValueError("expected_features must contain only finite values")
    if feedback.ndim != 1 or feedback.shape[0] != row_count:
        raise ValueError("aggregate_feedback must match the observation row count")
    if not np.all(np.isfinite(feedback)):
        raise ValueError("aggregate_feedback must contain only finite values")

    design = np.ascontiguousarray(
        observations.reshape(row_count, action_count * feature_size),
        dtype=np.float64,
    )
    dimension = design.shape[1]
    penalty = C4_RIDGE_REGULARIZATION
    augmented_design = np.vstack(
        (design, math.sqrt(penalty) * np.eye(dimension, dtype=np.float64))
    )
    augmented_targets = np.concatenate(
        (np.ascontiguousarray(feedback, dtype=np.float64), np.zeros(dimension))
    )
    flat_weights, _, rank, _ = np.linalg.lstsq(
        augmented_design,
        augmented_targets,
        rcond=None,
    )
    if flat_weights.shape != (dimension,) or not np.all(np.isfinite(flat_weights)):
        raise ValueError("ridge solution must be a finite weight vector")

    return AnonymousBatchRidge(
        weights=flat_weights.reshape(action_count, feature_size),
        rank=int(rank),
        regularization=penalty,
        observation_count=row_count,
    )
