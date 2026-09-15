import numpy as np
import pytest

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.flow_types import FlowObservation
from yolo_motion.pose_regions import BodyPart, PoseRegionConfig, build_body_regions, rasterize_region
from yolo_motion.pose_types import PoseLayout, PoseObservation
from yolo_motion.types import TrackObservation


FRAME_SIZE = 400


def body_points(person_height_px: float) -> dict[int, tuple[float, float]]:
    cx = FRAME_SIZE / 2.0
    top = (FRAME_SIZE - person_height_px) / 2.0
    p = person_height_px
    return {
        5: (cx - 0.12 * p, top + 0.18 * p),
        6: (cx + 0.12 * p, top + 0.18 * p),
        11: (cx - 0.08 * p, top + 0.48 * p),
        12: (cx + 0.08 * p, top + 0.48 * p),
        13: (cx - 0.10 * p, top + 0.68 * p),
        14: (cx + 0.10 * p, top + 0.68 * p),
        15: (cx - 0.11 * p, top + 0.88 * p),
        16: (cx + 0.11 * p, top + 0.88 * p),
        17: (cx - 0.13 * p, top + 0.94 * p),
        18: (cx - 0.09 * p, top + 0.95 * p),
        19: (cx - 0.16 * p, top + 0.95 * p),
        20: (cx + 0.13 * p, top + 0.94 * p),
        21: (cx + 0.09 * p, top + 0.95 * p),
        22: (cx + 0.16 * p, top + 0.95 * p),
    }


def make_pose(
    person_height_px: float,
    timestamp: float,
    *,
    translation=(0.0, 0.0),
    point_offsets=None,
    low_confidence=(),
) -> PoseObservation:
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.95, dtype=float)
    offsets = point_offsets or {}
    for index, (x, y) in body_points(person_height_px).items():
        ox, oy = offsets.get(index, (0.0, 0.0))
        xy[index] = (
            (x + translation[0] + ox) / FRAME_SIZE,
            (y + translation[1] + oy) / FRAME_SIZE,
        )
    for index in low_confidence:
        confidence[index] = 0.1
    return PoseObservation(
        track_id=17,
        timestamp=timestamp,
        frame_width=FRAME_SIZE,
        frame_height=FRAME_SIZE,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )


def make_track(person_height_px: float, timestamp: float = 1.1) -> TrackObservation:
    return TrackObservation(
        track_id=17,
        timestamp=timestamp,
        class_id=0,
        confidence=0.95,
        cx=0.5,
        cy=0.5,
        width=min(0.9, person_height_px * 0.45 / FRAME_SIZE),
        height=person_height_px / FRAME_SIZE,
    )


def make_flow(
    previous_pose: PoseObservation,
    person_height_px: float,
    *,
    body_translation=(0.0, 0.0),
    residuals=None,
    omitted_parts=(),
):
    regions = build_body_regions(
        previous_pose,
        person_height_px,
        PoseRegionConfig(capsule_width_ratio=0.035),
    )
    for part in omitted_parts:
        regions.pop(part, None)

    dx = np.full((FRAME_SIZE, FRAME_SIZE), body_translation[0], dtype=float)
    dy = np.full((FRAME_SIZE, FRAME_SIZE), body_translation[1], dtype=float)
    for part, (rx, ry) in (residuals or {}).items():
        mask = rasterize_region(regions[part], (FRAME_SIZE, FRAME_SIZE))
        dx[mask] = body_translation[0] + rx
        dy[mask] = body_translation[1] + ry

    flow = FlowObservation(
        start_timestamp=1.0,
        end_timestamp=1.1,
        dx=dx,
        dy=dy,
        valid=np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool),
        backend="synthetic-compensated",
    )
    return regions, flow


def estimate_case(
    person_height_px: float,
    *,
    body_translation=(0.0, 0.0),
    residuals=None,
    current_point_offsets=None,
    previous_low_confidence=(),
):
    previous = make_pose(
        person_height_px,
        1.0,
        low_confidence=previous_low_confidence,
    )
    current = make_pose(
        person_height_px,
        1.1,
        translation=body_translation,
        point_offsets=current_point_offsets,
        low_confidence=previous_low_confidence,
    )
    regions, flow = make_flow(
        previous,
        person_height_px,
        body_translation=body_translation,
        residuals=residuals,
    )
    evidence = estimate_articulated_flow(
        make_track(person_height_px),
        previous,
        current,
        regions,
        flow,
        ArticulatedFlowConfig(
            min_person_height_px=40.0,
            min_region_valid_fraction=0.50,
            min_torso_valid_fraction=0.60,
            max_pose_flow_error_norm=0.08,
        ),
    )
    return evidence


def test_rigid_person_translation_is_removed_from_limb_motion():
    evidence = estimate_case(180.0, body_translation=(6.0, 2.0))

    assert evidence is not None
    assert evidence.torso_dx == pytest.approx(6.0)
    assert evidence.torso_dy == pytest.approx(2.0)
    assert evidence.normalized_torso_dx == pytest.approx(6.0 / 180.0)
    assert evidence.normalized_torso_dy == pytest.approx(2.0 / 180.0)
    assert evidence.region_flow
    assert max(region.energy for region in evidence.region_flow.values()) < 1e-6


def test_walking_in_place_keeps_opposite_leg_residual_motion():
    residual = 9.0
    evidence = estimate_case(
        180.0,
        residuals={
            BodyPart.LEFT_CALF: (residual, 0.0),
            BodyPart.LEFT_FOOT: (residual, 0.0),
            BodyPart.RIGHT_CALF: (-residual, 0.0),
            BodyPart.RIGHT_FOOT: (-residual, 0.0),
        },
        current_point_offsets={
            15: (residual, 0.0),
            17: (residual, 0.0),
            18: (residual, 0.0),
            19: (residual, 0.0),
            16: (-residual, 0.0),
            20: (-residual, 0.0),
            21: (-residual, 0.0),
            22: (-residual, 0.0),
        },
    )

    assert evidence is not None
    assert abs(evidence.torso_dx) < 1e-6
    assert evidence.region_flow[BodyPart.LEFT_CALF.value].dx > 0.03
    assert evidence.region_flow[BodyPart.RIGHT_CALF.value].dx < -0.03
    assert evidence.region_flow[BodyPart.LEFT_CALF.value].energy > 0.03
    assert evidence.region_flow[BodyPart.RIGHT_CALF.value].energy > 0.03


def test_translation_plus_articulation_preserves_limb_residual_after_torso_subtraction():
    residual = 7.0
    evidence = estimate_case(
        140.0,
        body_translation=(5.0, -2.0),
        residuals={
            BodyPart.LEFT_CALF: (residual, 0.0),
            BodyPart.RIGHT_CALF: (-residual, 0.0),
        },
        current_point_offsets={15: (residual, 0.0), 16: (-residual, 0.0)},
    )

    assert evidence is not None
    assert evidence.torso_dx == pytest.approx(5.0)
    assert evidence.torso_dy == pytest.approx(-2.0)
    assert evidence.region_flow[BodyPart.LEFT_CALF.value].dx == pytest.approx(
        residual / 140.0, abs=0.01
    )
    assert evidence.region_flow[BodyPart.RIGHT_CALF.value].dx == pytest.approx(
        -residual / 140.0, abs=0.01
    )


def test_scale_normalization_makes_equivalent_far_and_near_gait_comparable():
    small_height = 50.0
    large_height = 250.0
    small_residual = 0.05 * small_height
    large_residual = 0.05 * large_height

    small = estimate_case(
        small_height,
        residuals={BodyPart.LEFT_CALF: (small_residual, 0.0)},
        current_point_offsets={15: (small_residual, 0.0)},
    )
    large = estimate_case(
        large_height,
        residuals={BodyPart.LEFT_CALF: (large_residual, 0.0)},
        current_point_offsets={15: (large_residual, 0.0)},
    )

    assert small is not None and large is not None
    small_energy = small.region_flow[BodyPart.LEFT_CALF.value].energy
    large_energy = large.region_flow[BodyPart.LEFT_CALF.value].energy
    assert abs(small_energy - large_energy) <= 0.01


def test_pose_jump_disagreement_reduces_agreement_and_quality():
    stable = estimate_case(180.0, body_translation=(2.0, 0.0))
    jumped = estimate_case(
        180.0,
        body_translation=(2.0, 0.0),
        current_point_offsets={15: (70.0, 0.0)},
    )

    assert stable is not None and jumped is not None
    assert stable.pose_flow_agreement > jumped.pose_flow_agreement
    assert stable.quality > jumped.quality


def test_one_leg_occlusion_retains_available_opposite_leg_evidence():
    low_right = (14, 16, 20, 21, 22)
    previous = make_pose(180.0, 1.0, low_confidence=low_right)
    current = make_pose(180.0, 1.1, low_confidence=low_right)
    regions, flow = make_flow(previous, 180.0)

    evidence = estimate_articulated_flow(
        make_track(180.0),
        previous,
        current,
        regions,
        flow,
        ArticulatedFlowConfig(),
    )

    assert evidence is not None
    assert BodyPart.LEFT_CALF.value in evidence.region_flow
    assert BodyPart.LEFT_FOOT.value in evidence.region_flow
    assert BodyPart.RIGHT_CALF.value not in evidence.region_flow
    assert BodyPart.RIGHT_FOOT.value not in evidence.region_flow


def test_too_small_person_fails_closed_without_fabricated_gait_evidence():
    previous = make_pose(30.0, 1.0)
    current = make_pose(30.0, 1.1)
    regions, flow = make_flow(previous, 30.0)

    assert (
        estimate_articulated_flow(
            make_track(30.0),
            previous,
            current,
            regions,
            flow,
            ArticulatedFlowConfig(min_person_height_px=40.0),
        )
        is None
    )


def test_articulated_flow_config_rejects_invalid_quality_thresholds():
    with pytest.raises(ValueError):
        ArticulatedFlowConfig(min_person_height_px=0.0)
    with pytest.raises(ValueError):
        ArticulatedFlowConfig(min_region_valid_fraction=1.1)
    with pytest.raises(ValueError):
        ArticulatedFlowConfig(max_pose_flow_error_norm=0.0)
