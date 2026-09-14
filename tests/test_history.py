import pytest

from yolo_motion.history import TrackHistory
from yolo_motion.types import TrackObservation


def obs(t: float, area_scale: float = 1.0) -> TrackObservation:
    return TrackObservation(
        track_id=3,
        timestamp=t,
        class_id=0,
        confidence=0.95,
        cx=0.5,
        cy=0.5,
        width=0.1 * area_scale,
        height=0.2 * area_scale,
    )


def test_history_rejects_non_monotonic_timestamps():
    history = TrackHistory(track_id=3, history_seconds=1.0)
    history.add(obs(1.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        history.add(obs(1.0))


def test_history_prunes_observations_older_than_window():
    history = TrackHistory(track_id=3, history_seconds=1.0)
    history.add(obs(0.0))
    history.add(obs(0.5))
    history.add(obs(1.2))

    assert [item.timestamp for item in history.observations()] == [0.5, 1.2]


def test_history_rejects_wrong_track_id():
    history = TrackHistory(track_id=3, history_seconds=1.0)
    wrong = TrackObservation(4, 0.0, 0, 0.9, 0.5, 0.5, 0.1, 0.2)

    with pytest.raises(ValueError, match="track_id"):
        history.add(wrong)
