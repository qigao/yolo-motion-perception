import numpy as np
import pytest

from yolo_motion.ultralytics_adapter import observations_from_result


class FakeBoxes:
    def __init__(self):
        self.id = np.asarray([11, 12])
        self.xywh = np.asarray([[320.0, 240.0, 64.0, 96.0], [160.0, 120.0, 32.0, 48.0]])
        self.conf = np.asarray([0.91, 0.82])
        self.cls = np.asarray([0, 2])


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_adapter_converts_xywh_to_normalized_observations():
    observations = observations_from_result(
        FakeResult(FakeBoxes()), timestamp=1.25, frame_shape=(480, 640)
    )

    assert [obs.track_id for obs in observations] == [11, 12]
    assert observations[0].cx == pytest.approx(0.5)
    assert observations[0].cy == pytest.approx(0.5)
    assert observations[0].width == pytest.approx(0.1)
    assert observations[0].height == pytest.approx(0.2)
    assert observations[0].confidence == pytest.approx(0.91)
    assert observations[0].class_id == 0


def test_adapter_returns_empty_when_tracker_has_no_ids():
    boxes = FakeBoxes()
    boxes.id = None

    assert observations_from_result(FakeResult(boxes), 0.0, (480, 640)) == []
