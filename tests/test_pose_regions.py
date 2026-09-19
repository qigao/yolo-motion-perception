import numpy as np
import pytest

from yolo_motion.pose_regions import (
    BodyPart,
    PoseRegionConfig,
    build_body_regions,
    rasterize_region,
)
from yolo_motion.pose_types import PoseLayout, PoseObservation


def make_wholebody_pose(*, confidence_overrides=None):
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.95, dtype=float)

    points = {
        5: (0.40, 0.30),
        6: (0.60, 0.30),
        11: (0.45, 0.55),
        12: (0.55, 0.55),
        13: (0.44, 0.72),
        14: (0.56, 0.72),
        15: (0.43, 0.90),
        16: (0.57, 0.90),
        17: (0.42, 0.94),
        18: (0.44, 0.95),
        19: (0.40, 0.95),
        20: (0.58, 0.94),
        21: (0.56, 0.95),
        22: (0.60, 0.95),
    }
    for index, value in points.items():
        xy[index] = value
    for index, value in (confidence_overrides or {}).items():
        confidence[index] = value

    return PoseObservation(
        track_id=17,
        timestamp=2.0,
        frame_width=640,
        frame_height=480,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )


def test_build_body_regions_returns_all_locomotion_regions():
    regions = build_body_regions(
        make_wholebody_pose(),
        person_height_px=200.0,
        config=PoseRegionConfig(),
    )

    assert set(regions) == {
        BodyPart.TORSO,
        BodyPart.LEFT_THIGH,
        BodyPart.RIGHT_THIGH,
        BodyPart.LEFT_CALF,
        BodyPart.RIGHT_CALF,
        BodyPart.LEFT_FOOT,
        BodyPart.RIGHT_FOOT,
    }
    assert all(region.confidence == pytest.approx(0.95) for region in regions.values())
    assert all(region.radius_px == pytest.approx(12.0) for region in regions.values())


def test_region_segments_are_converted_from_normalized_coordinates_to_pixels():
    regions = build_body_regions(
        make_wholebody_pose(),
        person_height_px=200.0,
        config=PoseRegionConfig(),
    )

    start, end = regions[BodyPart.LEFT_THIGH].segments[0]
    np.testing.assert_allclose(start, np.array([0.45 * 640, 0.55 * 480]))
    np.testing.assert_allclose(end, np.array([0.44 * 640, 0.72 * 480]))
    assert not start.flags.writeable
    assert not end.flags.writeable


def test_capsule_width_scales_with_person_height():
    pose = make_wholebody_pose()
    config = PoseRegionConfig(capsule_width_ratio=0.05)
    small = build_body_regions(pose, person_height_px=80.0, config=config)
    large = build_body_regions(pose, person_height_px=240.0, config=config)

    assert small[BodyPart.LEFT_CALF].radius_px == pytest.approx(4.0)
    assert large[BodyPart.LEFT_CALF].radius_px == pytest.approx(12.0)


def test_rasterize_region_returns_deterministic_boolean_capsule_mask():
    regions = build_body_regions(
        make_wholebody_pose(),
        person_height_px=120.0,
        config=PoseRegionConfig(capsule_width_ratio=0.04),
    )
    region = regions[BodyPart.LEFT_CALF]
    mask_a = rasterize_region(region, (480, 640))
    mask_b = rasterize_region(region, (480, 640))

    assert mask_a.shape == (480, 640)
    assert mask_a.dtype == np.bool_
    np.testing.assert_array_equal(mask_a, mask_b)

    start, end = region.segments[0]
    midpoint = np.rint((start + end) / 2.0).astype(int)
    assert bool(mask_a[midpoint[1], midpoint[0]]) is True
    assert bool(mask_a[0, 0]) is False


def test_low_confidence_right_leg_is_omitted_without_dropping_left_leg():
    pose = make_wholebody_pose(
        confidence_overrides={14: 0.1, 16: 0.1, 20: 0.1, 21: 0.1, 22: 0.1}
    )
    regions = build_body_regions(
        pose,
        person_height_px=180.0,
        config=PoseRegionConfig(min_keypoint_confidence=0.35),
    )

    assert BodyPart.LEFT_THIGH in regions
    assert BodyPart.LEFT_CALF in regions
    assert BodyPart.LEFT_FOOT in regions
    assert BodyPart.RIGHT_THIGH not in regions
    assert BodyPart.RIGHT_CALF not in regions
    assert BodyPart.RIGHT_FOOT not in regions
    assert BodyPart.TORSO in regions


def test_build_body_regions_rejects_non_positive_person_height():
    with pytest.raises(ValueError, match="height"):
        build_body_regions(
            make_wholebody_pose(),
            person_height_px=0.0,
            config=PoseRegionConfig(),
        )


def test_pose_region_config_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        PoseRegionConfig(min_keypoint_confidence=1.1)
    with pytest.raises(ValueError):
        PoseRegionConfig(capsule_width_ratio=0.0)
