import math

from yolo_motion.motion import MotionConfig, estimate_motion
from yolo_motion.state import classify_motion
from yolo_motion.types import LateralState, RadialState, TrackObservation


def make_sequence(*, expansion_rate=0.0, vx=0.0, count=7, dt=0.1):
    items = []
    for i in range(count):
        t = i * dt
        scale = math.exp(expansion_rate * t / 2.0)
        items.append(
            TrackObservation(1, t, 0, 0.95, 0.4 + vx * t, 0.4, 0.1 * scale, 0.2 * scale)
        )
    return items


def test_sustained_growth_classifies_as_approaching():
    config = MotionConfig()
    evidence = estimate_motion(make_sequence(expansion_rate=0.5), config)

    state = classify_motion(evidence, config)

    assert state.radial is RadialState.APPROACHING


def test_sustained_shrink_classifies_as_receding():
    config = MotionConfig()
    evidence = estimate_motion(make_sequence(expansion_rate=-0.5), config)

    state = classify_motion(evidence, config)

    assert state.radial is RadialState.RECEDING


def test_lateral_motion_is_orthogonal_to_radial_state():
    config = MotionConfig()
    evidence = estimate_motion(make_sequence(vx=0.08, expansion_rate=0.5), config)

    state = classify_motion(evidence, config)

    assert state.lateral is LateralState.MOVING
    assert state.radial is RadialState.APPROACHING


def test_single_scale_spike_is_not_classified_as_approaching():
    config = MotionConfig()
    areas = [1.0, 1.0, 1.0, 1.8, 1.0, 1.0, 1.0]
    observations = []
    for i, area_scale in enumerate(areas):
        side_scale = math.sqrt(area_scale)
        observations.append(
            TrackObservation(
                1,
                i * 0.1,
                0,
                0.95,
                0.5,
                0.5,
                0.1 * side_scale,
                0.2 * side_scale,
            )
        )

    evidence = estimate_motion(observations, config)
    state = classify_motion(evidence, config)

    assert state.radial is RadialState.STABLE
    assert evidence.approach_confidence < config.min_radial_confidence


def test_insufficient_history_is_unknown():
    config = MotionConfig(min_samples=5)
    evidence = estimate_motion(make_sequence(expansion_rate=0.8, count=3), config)

    state = classify_motion(evidence, config)

    assert state.radial is RadialState.UNKNOWN
