import numpy as np

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.flow_types import FlowObservation
from yolo_motion.pose_regions import PoseRegionConfig, build_body_regions
from yolo_motion.pose_types import PoseLayout, PoseObservation
from yolo_motion.types import TrackObservation

FRAME_SIZE = 400
PERSON_HEIGHT = 180.0
LOCOMOTION_KEYPOINTS = (5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def _pose(timestamp: float, translation_x: float = 0.0) -> PoseObservation:
    cx = FRAME_SIZE / 2.0
    top = (FRAME_SIZE - PERSON_HEIGHT) / 2.0
    p = PERSON_HEIGHT
    points = {
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
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.95, dtype=float)
    for index, (x, y) in points.items():
        xy[index] = ((x + translation_x) / FRAME_SIZE, y / FRAME_SIZE)
    return PoseObservation(
        track_id=17,
        timestamp=timestamp,
        frame_width=FRAME_SIZE,
        frame_height=FRAME_SIZE,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )


def test_pose_flow_agreement_uses_local_support_not_one_contaminated_pixel():
    previous = _pose(1.0)
    current = _pose(1.1, translation_x=2.0)
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

    dx = np.full((FRAME_SIZE, FRAME_SIZE), 2.0, dtype=float)
    dy = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=float)

    # Simulate a limb-boundary failure: the optical field around every joint is
    # correct, but the exact sampled center pixel is an outlier. A robust local
    # neighborhood estimate should reject these isolated center-pixel outliers.
    for index in LOCOMOTION_KEYPOINTS:
        x = int(round(previous.xy[index, 0] * FRAME_SIZE))
        y = int(round(previous.xy[index, 1] * FRAME_SIZE))
        dx[y, x] = 20.0

    flow = FlowObservation(
        start_timestamp=1.0,
        end_timestamp=1.1,
        dx=dx,
        dy=dy,
        valid=np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool),
        backend="synthetic-contaminated-centers",
    )

    evidence = estimate_articulated_flow(
        track,
        previous,
        current,
        regions,
        flow,
        ArticulatedFlowConfig(max_pose_flow_error_norm=0.08),
    )

    assert evidence is not None
    assert evidence.pose_flow_agreement > 0.8
    assert evidence.quality > 0.8
