import math

import pytest
from test_gait_pipeline import gait_config, gait_observation

from yolo_motion.gait_pipeline import OpticalGaitPipeline
from yolo_motion.gait_state import LocomotionState
from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline
from yolo_motion.types import LateralState, RadialState, TrackObservation


def motion_config() -> MotionConfig:
    return MotionConfig(
        history_seconds=1.0,
        min_samples=5,
        min_duration=0.3,
        stationary_speed_threshold=0.02,
        radial_rate_threshold=0.15,
        min_radial_confidence=0.6,
        min_trend_consistency=0.75,
    )


def feed_v1_motion(track_id: int, expansion_rate: float):
    pipeline = MotionPipeline(motion_config())
    result = None
    for index in range(7):
        timestamp = index * 0.1
        size = 0.20 * math.exp(0.5 * expansion_rate * timestamp)
        result = pipeline.update(
            TrackObservation(
                track_id=track_id,
                timestamp=timestamp,
                class_id=0,
                confidence=0.95,
                cx=0.5,
                cy=0.5,
                width=size,
                height=size,
            )
        )
    assert result is not None
    return result


def feed_v2_walking(track_id: int):
    pipeline = OpticalGaitPipeline(gait_config())
    result = None
    for index in range(40):
        result = pipeline.update(gait_observation(track_id, index, amplitude=0.04))
    assert result is not None
    return result


def test_walking_in_place_is_orthogonal_to_v1_lateral_stationary_state():
    v1 = feed_v1_motion(41, expansion_rate=0.0)
    v2 = feed_v2_walking(41)

    assert v1.state.lateral is LateralState.STATIONARY
    assert v2.state is LocomotionState.WALKING


@pytest.mark.parametrize(
    ("expansion_rate", "expected_radial"),
    [
        (0.40, RadialState.APPROACHING),
        (-0.40, RadialState.RECEDING),
        (0.0, RadialState.STABLE),
    ],
)
def test_walking_combines_independently_with_v1_radial_state(
    expansion_rate: float,
    expected_radial: RadialState,
):
    v1 = feed_v1_motion(73, expansion_rate=expansion_rate)
    v2 = feed_v2_walking(73)

    assert v1.state.lateral is LateralState.STATIONARY
    assert v1.state.radial is expected_radial
    assert v2.state is LocomotionState.WALKING


def test_v2_pipeline_does_not_mutate_v1_track_history():
    v1_pipeline = MotionPipeline(motion_config())
    v2_pipeline = OpticalGaitPipeline(gait_config())

    for index in range(7):
        timestamp = index * 0.1
        result = v1_pipeline.update(
            TrackObservation(
                track_id=8,
                timestamp=timestamp,
                class_id=0,
                confidence=0.95,
                cx=0.5,
                cy=0.5,
                width=0.2,
                height=0.2,
            )
        )
    assert result is not None
    state_before = result.state
    ids_before = v1_pipeline.track_ids()

    for index in range(40):
        v2_pipeline.update(gait_observation(8, index, amplitude=0.04))

    assert v1_pipeline.track_ids() == ids_before
    assert result.state == state_before
    assert v2_pipeline.track_ids() == {8}
