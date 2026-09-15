from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _readonly_array(value: object, *, dtype: object | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _require_finite(name: str, *values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite")


def _require_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class FlowObservation:
    start_timestamp: float
    end_timestamp: float
    dx: np.ndarray
    dy: np.ndarray
    valid: np.ndarray | None
    backend: str

    def __post_init__(self) -> None:
        _require_finite("timestamp", self.start_timestamp, self.end_timestamp)
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("end timestamp must be greater than start timestamp")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")

        dx = _readonly_array(self.dx, dtype=float)
        dy = _readonly_array(self.dy, dtype=float)
        if dx.ndim != 2 or dy.ndim != 2 or dx.shape != dy.shape or dx.size == 0:
            raise ValueError("dx and dy must be non-empty 2D arrays with the same shape")
        if not np.isfinite(dx).all() or not np.isfinite(dy).all():
            raise ValueError("flow vectors must be finite")

        valid: np.ndarray | None
        if self.valid is None:
            valid = None
        else:
            valid = _readonly_array(self.valid, dtype=bool)
            if valid.shape != dx.shape:
                raise ValueError("valid mask shape must match flow shape")

        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class CameraMotionEstimate:
    model: str
    matrix: np.ndarray
    matched_count: int
    inlier_count: int
    inlier_ratio: float
    residual_error: float
    quality: float
    valid: bool

    def __post_init__(self) -> None:
        if self.model not in {"identity", "affine"}:
            raise ValueError("camera motion model must be identity or affine")

        matrix = _readonly_array(self.matrix, dtype=float)
        if matrix.shape != (2, 3):
            raise ValueError("camera motion matrix must be 2x3")
        if not np.isfinite(matrix).all():
            raise ValueError("camera motion matrix must be finite")

        if self.matched_count < 0 or self.inlier_count < 0:
            raise ValueError("camera motion counts must be non-negative")
        if self.inlier_count > self.matched_count:
            raise ValueError("inlier count cannot exceed matched count")
        _require_unit_interval("inlier ratio", self.inlier_ratio)
        _require_unit_interval("quality", self.quality)
        if not math.isfinite(self.residual_error) or self.residual_error < 0.0:
            raise ValueError("residual error must be finite and non-negative")

        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class RegionFlowEvidence:
    dx: float
    dy: float
    energy: float
    valid_fraction: float

    def __post_init__(self) -> None:
        _require_finite("region flow", self.dx, self.dy, self.energy, self.valid_fraction)
        if self.energy < 0.0:
            raise ValueError("region flow energy must be non-negative")
        if not 0.0 <= self.valid_fraction <= 1.0:
            raise ValueError("region valid fraction must be in [0, 1]")


@dataclass(frozen=True)
class ArticulatedFlowEvidence:
    track_id: int
    start_timestamp: float
    end_timestamp: float
    torso_dx: float
    torso_dy: float
    normalized_torso_dx: float
    normalized_torso_dy: float
    region_flow: Mapping[str, RegionFlowEvidence]
    pose_flow_agreement: float
    person_height_px: float
    quality: float

    def __post_init__(self) -> None:
        _require_finite("timestamp", self.start_timestamp, self.end_timestamp)
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("end timestamp must be greater than start timestamp")
        _require_finite(
            "articulated flow",
            self.torso_dx,
            self.torso_dy,
            self.normalized_torso_dx,
            self.normalized_torso_dy,
            self.person_height_px,
        )
        if self.person_height_px <= 0.0:
            raise ValueError("person height must be positive")
        _require_unit_interval("pose-flow agreement", self.pose_flow_agreement)
        _require_unit_interval("quality", self.quality)

        region_flow = dict(self.region_flow)
        for name, evidence in region_flow.items():
            if not isinstance(name, str) or not name:
                raise ValueError("region names must be non-empty strings")
            if not isinstance(evidence, RegionFlowEvidence):
                raise TypeError("region flow values must be RegionFlowEvidence")
        object.__setattr__(self, "region_flow", MappingProxyType(region_flow))
