from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from yolo_motion.pose_types import PoseLayout, PoseObservation


def make_pose(*, xy=None, confidence=None, frame_width=640, frame_height=480):
    if xy is None:
        xy = np.full((133, 2), 0.5, dtype=float)
    if confidence is None:
        confidence = np.full(133, 0.9, dtype=float)
    return PoseObservation(
        track_id=7,
        timestamp=1.25,
        frame_width=frame_width,
        frame_height=frame_height,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )


def test_pose_observation_copies_and_freezes_arrays():
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.9, dtype=float)
    pose = make_pose(xy=xy, confidence=confidence)

    xy[0, 0] = 0.1
    confidence[0] = 0.1

    assert pose.xy[0, 0] == pytest.approx(0.5)
    assert pose.confidence[0] == pytest.approx(0.9)
    assert not pose.xy.flags.writeable
    assert not pose.confidence.flags.writeable

    with pytest.raises(ValueError):
        pose.xy[0, 0] = 0.25
    with pytest.raises(FrozenInstanceError):
        pose.track_id = 8


def test_coco_wholebody_layout_requires_exactly_133_points():
    with pytest.raises(ValueError, match="133"):
        make_pose(xy=np.zeros((132, 2), dtype=float), confidence=np.ones(132, dtype=float))


def test_pose_observation_rejects_mismatched_confidence_shape():
    with pytest.raises(ValueError, match="confidence"):
        make_pose(confidence=np.ones(132, dtype=float))


@pytest.mark.parametrize(
    "xy, confidence",
    [
        (np.full((133, 2), np.nan), np.ones(133, dtype=float)),
        (np.zeros((133, 2), dtype=float), np.full(133, 1.1, dtype=float)),
        (np.zeros((133, 2), dtype=float), np.full(133, -0.1, dtype=float)),
    ],
)
def test_pose_observation_rejects_non_finite_or_out_of_range_values(xy, confidence):
    with pytest.raises(ValueError):
        make_pose(xy=xy, confidence=confidence)


@pytest.mark.parametrize("frame_width, frame_height", [(0, 480), (640, 0), (-1, 480)])
def test_pose_observation_rejects_non_positive_frame_dimensions(frame_width, frame_height):
    with pytest.raises(ValueError, match="frame"):
        make_pose(frame_width=frame_width, frame_height=frame_height)
