from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class LateralState(str, Enum):
    STATIONARY = "stationary"
    MOVING = "moving"


class RadialState(str, Enum):
    UNKNOWN = "unknown"
    STABLE = "stable"
    APPROACHING = "approaching"
    RECEDING = "receding"


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    timestamp: float
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (
            self.timestamp,
            self.confidence,
            self.cx,
            self.cy,
            self.width,
            self.height,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("observation values must be finite")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be positive")
        if not (0.0 <= self.cx <= 1.0 and 0.0 <= self.cy <= 1.0):
            raise ValueError("cx and cy must be normalized to [0, 1]")
        if not (0.0 < self.width <= 1.0 and 0.0 < self.height <= 1.0):
            raise ValueError("width and height must be normalized to (0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def log_area(self) -> float:
        return math.log(self.area)


@dataclass(frozen=True)
class MotionEvidence:
    vx: float
    vy: float
    speed: float
    expansion_rate: float
    trend_consistency: float
    fit_quality: float
    sample_count: int
    duration: float
    approach_confidence: float
    recede_confidence: float


@dataclass(frozen=True)
class MotionState:
    lateral: LateralState
    radial: RadialState
