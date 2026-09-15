import numpy as np

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.flow_types import FlowObservation
from yolo_motion.pose_regions import PoseRegionConfig, build_body_regions
from yolo_motion.pose_types import PoseLayout, PoseObservation
from yolo_motion.types import TrackObservation

FRAME_SIZE = 400
PERSON_HEIGHT = 180.0
LOCOMOTION_KEYPOINTS = (5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def _points() -> dict[int, tuple[float, float]]:
    cx = FRAME_SIZE / 2.0
    top = (FRAME_SIZE - PERSON_HEIGHT) / 2.0
    p = PERSON_HEIGHT
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


def _pose(timestamp: float, outliers: set[int] | None = None) -> PoseObservation:
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.95, dtype=float)
    outliers = outliers or set()
    for index, (x, y) in _points().items():
        dx = 70.0 if index in outliers else 0.0
        xy[index] = ((x + dx) / FRAME_SIZE, y / FRAME_SIZE)
    return PoseObservation(
        track_id=17,
        timestamp=timestamp,
        frame_width=FRAME_SIZE,
        frame_height=FRAME_SIZE,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )


def _estimate(outliers: set[int]):
    previous = _pose(1.0)
    current = _pose(1.1, outliers)
    track = TrackObservation(
        track_id=17,
        timestamp=1.1,
        class_id=0,
        confidence=0.95,
        cx=0.5,
        cy=0.5,
        width=0.25,
        height=PERSON_HEIGHT / FRAME_SIZE,
    )
    regions = build_body_regions(previous, PERSON_HEIGHT, PoseRegionConfig())
    flow = FlowObservation(
        start_timestamp=1.0,
        end_timestamp=1.1,
        dx=np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=float),
        dy=np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=float),
        valid=np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool),
        backend="synthetic-outlier-test",
    )
    return estimate_articulated_flow(
        track,
        previous,
        current,
        regions,
        flow,
        ArticulatedFlowConfig(max_pose_flow_error_norm=0.08),
    )


def test_two_catastrophic_joint_outliers_do_not_overrule_twelve_consistent_joints():
    evidence = _estimate({20, 21})

    assert evidence is not None
    assert 0.75 < evidence.pose_flow_agreement < 0.95
    assert 0.75 < evidence.quality < 0.95


def test_widespread_pose_flow_disagreement_still_fails_closed():
    evidence = _estimate(set(LOCOMOTION_KEYPOINTS))

    assert evidence is not None
    assert evidence.pose_flow_agreement < 0.10
    assert evidence.quality < 0.10
