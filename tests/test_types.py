import math

import pytest

from yolo_motion.types import TrackObservation


def test_observation_exposes_area_and_log_area():
    obs = TrackObservation(
        track_id=7,
        timestamp=1.0,
        class_id=0,
        confidence=0.9,
        cx=0.5,
        cy=0.4,
        width=0.2,
        height=0.3,
    )

    assert obs.area == pytest.approx(0.06)
    assert obs.log_area == pytest.approx(math.log(0.06))


def test_observation_rejects_non_positive_bbox_dimensions():
    with pytest.raises(ValueError, match="width and height"):
        TrackObservation(
            track_id=1,
            timestamp=0.0,
            class_id=0,
            confidence=0.8,
            cx=0.5,
            cy=0.5,
            width=0.0,
            height=0.1,
        )
