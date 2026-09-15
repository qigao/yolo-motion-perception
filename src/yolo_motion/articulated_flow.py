from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .flow_types import (
    ArticulatedFlowEvidence,
    FlowObservation,
    RegionFlowEvidence,
)
from .pose_regions import BodyPart, BodyRegion, rasterize_region
from .pose_types import PoseObservation
from .types import TrackObservation

_LOCOMOTION_KEYPOINTS = (5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
_POSE_FLOW_NEIGHBORHOOD_RATIO = 0.02
_POSE_FLOW_ERROR_QUANTILE = 0.25


@dataclass(frozen=True)
class ArticulatedFlowConfig:
    min_person_height_px: float = 40.0
    min_region_valid_fraction: float = 0.50
    min_torso_valid_fraction: float = 0.60
    max_pose_flow_error_norm: float = 0.08

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_person_height_px) or self.min_person_height_px <= 0.0:
            raise ValueError("minimum person height must be positive and finite")
        for name, value in (
            ("minimum region valid fraction", self.min_region_valid_fraction),
            ("minimum torso valid fraction", self.min_torso_valid_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            not math.isfinite(self.max_pose_flow_error_norm)
            or self.max_pose_flow_error_norm <= 0.0
        ):
            raise ValueError("maximum pose/flow error must be positive and finite")


def _sample_region(
    flow: FlowObservation,
    region: BodyRegion,
) -> tuple[float, float, float] | None:
    mask = rasterize_region(region, flow.dx.shape)
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        return None

    valid = mask if flow.valid is None else mask & flow.valid
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = valid_count / pixel_count
    if valid_count == 0:
        return 0.0, 0.0, valid_fraction
    return (
        float(np.median(flow.dx[valid])),
        float(np.median(flow.dy[valid])),
        float(valid_fraction),
    )


def _keypoint_pixel(pose: PoseObservation, index: int) -> tuple[float, float]:
    return (
        float(pose.xy[index, 0] * pose.frame_width),
        float(pose.xy[index, 1] * pose.frame_height),
    )


def _local_flow_error(
    flow: FlowObservation,
    x: float,
    y: float,
    pose_dx: float,
    pose_dy: float,
    radius_px: int,
) -> float | None:
    height, width = flow.dx.shape
    cx = int(np.clip(round(x), 0, width - 1))
    cy = int(np.clip(round(y), 0, height - 1))
    x0 = max(0, cx - radius_px)
    x1 = min(width, cx + radius_px + 1)
    y0 = max(0, cy - radius_px)
    y1 = min(height, cy + radius_px + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    local_mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px * radius_px
    if flow.valid is not None:
        local_mask &= flow.valid[y0:y1, x0:x1]
    if not np.any(local_mask):
        return None

    local_dx = flow.dx[y0:y1, x0:x1][local_mask]
    local_dy = flow.dy[y0:y1, x0:x1][local_mask]
    errors = np.hypot(local_dx - pose_dx, local_dy - pose_dy)
    if errors.size == 0 or not np.isfinite(errors).all():
        return None
    return float(np.quantile(errors, _POSE_FLOW_ERROR_QUANTILE))


def _pose_flow_agreement(
    previous_pose: PoseObservation,
    current_pose: PoseObservation,
    flow: FlowObservation,
    person_height_px: float,
    max_error_norm: float,
) -> float:
    agreements: list[float] = []
    radius_px = max(1, round(_POSE_FLOW_NEIGHBORHOOD_RATIO * person_height_px))
    for index in _LOCOMOTION_KEYPOINTS:
        if previous_pose.confidence[index] <= 0.0 or current_pose.confidence[index] <= 0.0:
            continue
        previous_x, previous_y = _keypoint_pixel(previous_pose, index)
        current_x, current_y = _keypoint_pixel(current_pose, index)
        pose_dx = current_x - previous_x
        pose_dy = current_y - previous_y
        error = _local_flow_error(
            flow,
            previous_x,
            previous_y,
            pose_dx,
            pose_dy,
            radius_px,
        )
        if error is None:
            continue
        normalized_error = error / person_height_px
        agreements.append(
            float(np.clip(1.0 - normalized_error / max_error_norm, 0.0, 1.0))
        )

    if not agreements:
        return 0.0
    return float(np.mean(agreements))


def _validate_alignment(
    track: TrackObservation,
    previous_pose: PoseObservation,
    current_pose: PoseObservation,
    flow: FlowObservation,
) -> None:
    if previous_pose.track_id != track.track_id or current_pose.track_id != track.track_id:
        raise ValueError("track and pose track IDs must agree")
    if (
        previous_pose.frame_width != current_pose.frame_width
        or previous_pose.frame_height != current_pose.frame_height
    ):
        raise ValueError("pose frame dimensions must agree")
    if flow.dx.shape != (current_pose.frame_height, current_pose.frame_width):
        raise ValueError("flow shape must match pose frame dimensions")
    if not math.isclose(previous_pose.timestamp, flow.start_timestamp, abs_tol=1e-9):
        raise ValueError("previous pose timestamp must match flow start timestamp")
    if not math.isclose(current_pose.timestamp, flow.end_timestamp, abs_tol=1e-9):
        raise ValueError("current pose timestamp must match flow end timestamp")
    if not math.isclose(track.timestamp, current_pose.timestamp, abs_tol=1e-9):
        raise ValueError("track timestamp must match current pose timestamp")


def estimate_articulated_flow(
    track: TrackObservation,
    previous_pose: PoseObservation,
    current_pose: PoseObservation,
    regions: Mapping[BodyPart, BodyRegion],
    compensated_flow: FlowObservation,
    config: ArticulatedFlowConfig,
) -> ArticulatedFlowEvidence | None:
    _validate_alignment(track, previous_pose, current_pose, compensated_flow)

    person_height_px = float(track.height * current_pose.frame_height)
    if person_height_px < config.min_person_height_px:
        return None

    torso = regions.get(BodyPart.TORSO)
    if torso is None:
        return None
    torso_sample = _sample_region(compensated_flow, torso)
    if torso_sample is None:
        return None
    torso_dx, torso_dy, torso_valid_fraction = torso_sample
    if torso_valid_fraction < config.min_torso_valid_fraction:
        return None

    region_flow: dict[str, RegionFlowEvidence] = {}
    valid_fractions: list[float] = []
    for part, region in regions.items():
        if part is BodyPart.TORSO:
            continue
        sample = _sample_region(compensated_flow, region)
        if sample is None:
            continue
        region_dx, region_dy, valid_fraction = sample
        if valid_fraction < config.min_region_valid_fraction:
            continue
        normalized_dx = (region_dx - torso_dx) / person_height_px
        normalized_dy = (region_dy - torso_dy) / person_height_px
        energy = math.hypot(normalized_dx, normalized_dy)
        region_flow[part.value] = RegionFlowEvidence(
            dx=normalized_dx,
            dy=normalized_dy,
            energy=energy,
            valid_fraction=valid_fraction,
        )
        valid_fractions.append(valid_fraction)

    if not region_flow:
        return None

    agreement = _pose_flow_agreement(
        previous_pose,
        current_pose,
        compensated_flow,
        person_height_px,
        config.max_pose_flow_error_norm,
    )
    support_quality = min(
        torso_valid_fraction,
        float(np.mean(valid_fractions)) if valid_fractions else 0.0,
    )
    quality = float(np.clip(support_quality * agreement, 0.0, 1.0))

    return ArticulatedFlowEvidence(
        track_id=track.track_id,
        start_timestamp=compensated_flow.start_timestamp,
        end_timestamp=compensated_flow.end_timestamp,
        torso_dx=torso_dx,
        torso_dy=torso_dy,
        normalized_torso_dx=torso_dx / person_height_px,
        normalized_torso_dy=torso_dy / person_height_px,
        region_flow=region_flow,
        pose_flow_agreement=agreement,
        person_height_px=person_height_px,
        quality=quality,
    )
