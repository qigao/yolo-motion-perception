from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .pose_types import PoseLayout, PoseObservation


class BodyPart(str, Enum):
    TORSO = "torso"
    LEFT_THIGH = "left_thigh"
    RIGHT_THIGH = "right_thigh"
    LEFT_CALF = "left_calf"
    RIGHT_CALF = "right_calf"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


@dataclass(frozen=True)
class BodyRegion:
    part: BodyPart
    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    radius_px: float
    confidence: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_px) or self.radius_px <= 0.0:
            raise ValueError("region radius must be positive and finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("region confidence must be in [0, 1]")
        if not self.segments:
            raise ValueError("region must contain at least one segment")

        frozen_segments: list[tuple[np.ndarray, np.ndarray]] = []
        for start_value, end_value in self.segments:
            start = np.array(start_value, dtype=float, copy=True)
            end = np.array(end_value, dtype=float, copy=True)
            if start.shape != (2,) or end.shape != (2,):
                raise ValueError("region segment endpoints must have shape (2,)")
            if not np.isfinite(start).all() or not np.isfinite(end).all():
                raise ValueError("region segment endpoints must be finite")
            start.setflags(write=False)
            end.setflags(write=False)
            frozen_segments.append((start, end))
        object.__setattr__(self, "segments", tuple(frozen_segments))


@dataclass(frozen=True)
class PoseRegionConfig:
    min_keypoint_confidence: float = 0.35
    capsule_width_ratio: float = 0.06

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_keypoint_confidence)
            or not 0.0 <= self.min_keypoint_confidence <= 1.0
        ):
            raise ValueError("minimum keypoint confidence must be in [0, 1]")
        if not math.isfinite(self.capsule_width_ratio) or self.capsule_width_ratio <= 0.0:
            raise ValueError("capsule width ratio must be positive and finite")


def _point_px(pose: PoseObservation, index: int) -> np.ndarray:
    normalized = pose.xy[index]
    return np.array(
        [normalized[0] * pose.frame_width, normalized[1] * pose.frame_height],
        dtype=float,
    )


def _make_region(
    pose: PoseObservation,
    part: BodyPart,
    segment_indices: tuple[tuple[int, int], ...],
    required_indices: tuple[int, ...],
    radius_px: float,
    min_confidence: float,
) -> BodyRegion | None:
    confidence = float(np.min(pose.confidence[list(required_indices)]))
    if confidence < min_confidence:
        return None
    segments = tuple(
        (_point_px(pose, start), _point_px(pose, end))
        for start, end in segment_indices
    )
    return BodyRegion(
        part=part,
        segments=segments,
        radius_px=radius_px,
        confidence=confidence,
    )


def build_body_regions(
    pose: PoseObservation,
    person_height_px: float,
    config: PoseRegionConfig,
) -> dict[BodyPart, BodyRegion]:
    if pose.layout is not PoseLayout.COCO_WHOLEBODY_133:
        raise ValueError("unsupported pose layout")
    if not math.isfinite(person_height_px) or person_height_px <= 0.0:
        raise ValueError("person height must be positive and finite")

    radius_px = max(1.0, config.capsule_width_ratio * person_height_px)
    definitions: tuple[
        tuple[BodyPart, tuple[tuple[int, int], ...], tuple[int, ...]], ...
    ] = (
        (
            BodyPart.TORSO,
            ((5, 6), (6, 12), (12, 11), (11, 5), (5, 12), (6, 11)),
            (5, 6, 11, 12),
        ),
        (BodyPart.LEFT_THIGH, ((11, 13),), (11, 13)),
        (BodyPart.RIGHT_THIGH, ((12, 14),), (12, 14)),
        (BodyPart.LEFT_CALF, ((13, 15),), (13, 15)),
        (BodyPart.RIGHT_CALF, ((14, 16),), (14, 16)),
        (
            BodyPart.LEFT_FOOT,
            ((15, 17), (17, 18), (17, 19)),
            (15, 17, 18, 19),
        ),
        (
            BodyPart.RIGHT_FOOT,
            ((16, 20), (20, 21), (20, 22)),
            (16, 20, 21, 22),
        ),
    )

    regions: dict[BodyPart, BodyRegion] = {}
    for part, segments, required in definitions:
        region = _make_region(
            pose,
            part,
            segments,
            required,
            radius_px,
            config.min_keypoint_confidence,
        )
        if region is not None:
            regions[part] = region
    return regions


def rasterize_region(region: BodyRegion, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError("frame shape must be positive")

    yy, xx = np.ogrid[:height, :width]
    mask = np.zeros((height, width), dtype=bool)
    radius_squared = region.radius_px * region.radius_px

    for start, end in region.segments:
        vx = float(end[0] - start[0])
        vy = float(end[1] - start[1])
        length_squared = vx * vx + vy * vy
        if length_squared <= 1e-12:
            distance_squared = (xx - start[0]) ** 2 + (yy - start[1]) ** 2
        else:
            projection = ((xx - start[0]) * vx + (yy - start[1]) * vy) / length_squared
            projection = np.clip(projection, 0.0, 1.0)
            closest_x = start[0] + projection * vx
            closest_y = start[1] + projection * vy
            distance_squared = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
        mask |= distance_squared <= radius_squared
    return mask
