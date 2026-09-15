from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PoseLayout(str, Enum):
    COCO_WHOLEBODY_133 = "coco_wholebody_133"


@dataclass(frozen=True)
class PoseObservation:
    track_id: int
    timestamp: float
    frame_width: int
    frame_height: int
    xy: np.ndarray
    confidence: np.ndarray
    layout: PoseLayout

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")

        xy = np.array(self.xy, dtype=float, copy=True)
        confidence = np.array(self.confidence, dtype=float, copy=True)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (K, 2)")
        if confidence.shape != (xy.shape[0],):
            raise ValueError("confidence shape must match keypoint count")
        if self.layout is PoseLayout.COCO_WHOLEBODY_133 and xy.shape[0] != 133:
            raise ValueError("COCO WholeBody layout requires 133 keypoints")
        if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
            raise ValueError("pose values must be finite")
        if np.any(confidence < 0.0) or np.any(confidence > 1.0):
            raise ValueError("confidence values must be in [0, 1]")

        xy.setflags(write=False)
        confidence.setflags(write=False)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "confidence", confidence)
