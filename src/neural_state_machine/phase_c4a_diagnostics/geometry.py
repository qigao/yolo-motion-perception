"""D1 design-matrix geometry for C4-A failure attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import freeze_float64


_REGISTERED_PENALTY = 1e-6


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    row_count: int
    column_count: int
    feature_size: int
    tolerance: float
    rank: int
    nullity: int
    singular_values: np.ndarray
    sigma_max: float
    sigma_min_retained: float | None
    condition_number: float
    frobenius_norm: float
    stable_rank: float
    ridge_effective_dof: float
    leverage_min: float
    leverage_median: float
    leverage_max: float
    leverage_p10: float
    leverage_p90: float
    action_block_column_norms: tuple[float, float]
    cross_block_gram_frobenius: float
    bias_column_norms: tuple[float, float]


def _nearest_rank(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("nearest-rank summary requires a non-empty vector")
    index = math.ceil(quantile * ordered.size) - 1
    index = max(0, min(index, ordered.size - 1))
    return float(ordered[index])


def analyze_design_geometry(
    design: object,
    *,
    penalty: float = _REGISTERED_PENALTY,
) -> GeometryMetrics:
    """Compute the prospectively fixed D1 geometry without target information."""
    try:
        matrix = np.asarray(design, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("design must be float64-compatible") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design must be a non-empty two-dimensional matrix")
    if matrix.shape[1] % 2:
        raise ValueError("design columns must form two equal action blocks")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("design must contain only finite values")
    if isinstance(penalty, bool) or float(penalty) != _REGISTERED_PENALTY:
        raise ValueError("penalty must be exactly 1e-6")

    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    row_count, column_count = matrix.shape
    feature_size = column_count // 2

    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    singular = np.asarray(singular, dtype=np.float64)
    sigma_max = float(singular[0]) if singular.size else 0.0
    tolerance = float(
        max(row_count, column_count) * np.finfo(np.float64).eps * sigma_max
    )
    retained = singular > tolerance
    rank = int(np.count_nonzero(retained))
    nullity = int(column_count - rank)
    sigma_min_retained = float(singular[retained][-1]) if rank else None
    if rank < min(row_count, column_count) or sigma_min_retained in {None, 0.0}:
        condition_number = math.inf
    else:
        condition_number = float(sigma_max / sigma_min_retained)

    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    stable_rank = (
        float((frobenius_norm * frobenius_norm) / (sigma_max * sigma_max))
        if sigma_max > 0.0
        else 0.0
    )

    squared = singular * singular
    shrink = squared / (squared + _REGISTERED_PENALTY)
    ridge_effective_dof = float(np.sum(shrink))
    leverage = np.sum((u * u) * shrink[np.newaxis, :], axis=1)

    block0 = matrix[:, :feature_size]
    block1 = matrix[:, feature_size:]
    action_block_column_norms = (
        float(np.linalg.norm(block0, ord="fro")),
        float(np.linalg.norm(block1, ord="fro")),
    )
    cross_block_gram_frobenius = float(
        np.linalg.norm(block0.T @ block1, ord="fro")
    )
    bias_column_norms = (
        float(np.linalg.norm(matrix[:, feature_size - 1])),
        float(np.linalg.norm(matrix[:, column_count - 1])),
    )

    return GeometryMetrics(
        row_count=int(row_count),
        column_count=int(column_count),
        feature_size=int(feature_size),
        tolerance=tolerance,
        rank=rank,
        nullity=nullity,
        singular_values=freeze_float64(singular),
        sigma_max=sigma_max,
        sigma_min_retained=sigma_min_retained,
        condition_number=condition_number,
        frobenius_norm=frobenius_norm,
        stable_rank=stable_rank,
        ridge_effective_dof=ridge_effective_dof,
        leverage_min=float(np.min(leverage)),
        leverage_median=float(np.median(leverage)),
        leverage_max=float(np.max(leverage)),
        leverage_p10=_nearest_rank(leverage, 0.10),
        leverage_p90=_nearest_rank(leverage, 0.90),
        action_block_column_norms=action_block_column_norms,
        cross_block_gram_frobenius=cross_block_gram_frobenius,
        bias_column_norms=bias_column_norms,
    )
