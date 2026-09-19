import numpy as np
import pytest

from yolo_motion.rtmw_adapter import RtmwPoseAdapter
from yolo_motion.types import TrackObservation


class FakeInferencer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((np.array(image, copy=True), dict(kwargs)))
        return iter([self.result])


def make_track() -> TrackObservation:
    return TrackObservation(
        track_id=17,
        timestamp=1.25,
        class_id=0,
        confidence=0.95,
        cx=0.5,
        cy=0.5,
        width=0.5,
        height=0.5,
    )


def sample(point_count=133, *, score=0.9, x=10.0, y=20.0):
    return {
        "keypoints": [[x, y] for _ in range(point_count)],
        "keypoint_scores": [score for _ in range(point_count)],
    }


def test_adapter_converts_one_133_point_crop_prediction_to_full_frame_pose():
    inferencer = FakeInferencer({"predictions": [[sample()]]})
    adapter = RtmwPoseAdapter(inferencer)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    pose = adapter.infer_track(frame, make_track(), timestamp=1.25)

    assert pose is not None
    assert pose.track_id == 17
    assert pose.timestamp == pytest.approx(1.25)
    assert pose.frame_width == 200
    assert pose.frame_height == 100
    assert pose.xy.shape == (133, 2)
    assert pose.confidence.shape == (133,)
    assert pose.xy[0, 0] == pytest.approx(60.0 / 200.0)
    assert pose.xy[0, 1] == pytest.approx(45.0 / 100.0)
    assert pose.confidence[0] == pytest.approx(0.9)

    assert len(inferencer.calls) == 1
    crop, kwargs = inferencer.calls[0]
    assert crop.shape == (50, 100, 3)
    assert kwargs.get("return_vis") is False


def test_adapter_returns_none_when_inferencer_has_no_pose_sample():
    adapter = RtmwPoseAdapter(FakeInferencer({"predictions": [[]]}))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    assert adapter.infer_track(frame, make_track(), timestamp=1.25) is None


def test_adapter_rejects_malformed_non_wholebody_output_with_context():
    adapter = RtmwPoseAdapter(FakeInferencer({"predictions": [[sample(17)]]}))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"track 17.*133"):
        adapter.infer_track(frame, make_track(), timestamp=1.25)


def test_adapter_returns_none_for_all_low_quality_keypoints():
    adapter = RtmwPoseAdapter(FakeInferencer({"predictions": [[sample(score=0.0)]]}))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    assert adapter.infer_track(frame, make_track(), timestamp=1.25) is None


def test_adapter_clamps_tracker_crop_to_frame_bounds_and_offsets_coordinates():
    track = TrackObservation(
        track_id=23,
        timestamp=2.0,
        class_id=0,
        confidence=0.9,
        cx=0.1,
        cy=0.1,
        width=0.4,
        height=0.4,
    )
    inferencer = FakeInferencer({"predictions": [[sample(x=5.0, y=6.0)]]})
    adapter = RtmwPoseAdapter(inferencer)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    pose = adapter.infer_track(frame, track, timestamp=2.0)

    assert pose is not None
    crop, _ = inferencer.calls[0]
    assert crop.shape == (24, 36, 3)
    assert pose.xy[0, 0] == pytest.approx(5.0 / 120.0)
    assert pose.xy[0, 1] == pytest.approx(6.0 / 80.0)
