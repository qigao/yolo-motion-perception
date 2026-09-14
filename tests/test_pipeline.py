import math

from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline
from yolo_motion.types import RadialState, TrackObservation


def observation(track_id: int, t: float, *, expansion_rate: float = 0.0, cx: float = 0.5):
    scale = math.exp(expansion_rate * t / 2.0)
    return TrackObservation(
        track_id=track_id,
        timestamp=t,
        class_id=0,
        confidence=0.95,
        cx=cx,
        cy=0.5,
        width=0.1 * scale,
        height=0.2 * scale,
    )


def test_pipeline_keeps_histories_independent_per_track():
    pipeline = MotionPipeline(MotionConfig(min_samples=5, min_duration=0.3))

    result_1 = result_2 = None
    for i in range(6):
        t = i * 0.1
        result_1 = pipeline.update(observation(1, t, expansion_rate=0.5))
        result_2 = pipeline.update(observation(2, t, expansion_rate=-0.5))

    assert result_1 is not None
    assert result_2 is not None
    assert result_1.track_id == 1
    assert result_2.track_id == 2
    assert result_1.state.radial is RadialState.APPROACHING
    assert result_2.state.radial is RadialState.RECEDING


def test_pipeline_returns_none_until_history_is_sufficient():
    pipeline = MotionPipeline(MotionConfig(min_samples=5, min_duration=0.3))

    outputs = [pipeline.update(observation(3, i * 0.1, expansion_rate=0.5)) for i in range(4)]

    assert outputs == [None, None, None, None]


def test_pipeline_can_drop_stale_histories():
    pipeline = MotionPipeline(MotionConfig())
    pipeline.update(observation(1, 0.0))
    pipeline.update(observation(2, 0.0))

    removed = pipeline.drop_stale(active_track_ids={2})

    assert removed == {1}
    assert pipeline.track_ids() == {2}
