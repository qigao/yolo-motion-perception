import math

import pytest

from yolo_motion.motion import MotionConfig, estimate_motion
from yolo_motion.types import TrackObservation


def make_sequence(
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    expansion_rate: float = 0.0,
    count: int = 7,
    dt: float = 0.1,
):
    items = []
    for i in range(count):
        t = i * dt
        scale = math.exp(expansion_rate * t / 2.0)
        items.append(
            TrackObservation(
                track_id=1,
                timestamp=t,
                class_id=0,
                confidence=0.95,
                cx=0.4 + vx * t,
                cy=0.4 + vy * t,
                width=0.1 * scale,
                height=0.2 * scale,
            )
        )
    return items


def test_stationary_track_has_near_zero_motion():
    evidence = estimate_motion(make_sequence(), MotionConfig())

    assert evidence.vx == pytest.approx(0.0, abs=1e-9)
    assert evidence.vy == pytest.approx(0.0, abs=1e-9)
    assert evidence.expansion_rate == pytest.approx(0.0, abs=1e-9)
    assert evidence.approach_confidence == pytest.approx(0.0)
    assert evidence.recede_confidence == pytest.approx(0.0)


def test_lateral_track_estimates_normalized_velocity():
    evidence = estimate_motion(make_sequence(vx=0.08, vy=-0.03), MotionConfig())

    assert evidence.vx == pytest.approx(0.08, rel=1e-6)
    assert evidence.vy == pytest.approx(-0.03, rel=1e-6)
    assert evidence.speed == pytest.approx(math.hypot(0.08, -0.03), rel=1e-6)


def test_monotonic_growth_estimates_log_area_expansion():
    evidence = estimate_motion(make_sequence(expansion_rate=0.5), MotionConfig())

    assert evidence.expansion_rate == pytest.approx(0.5, rel=1e-6)
    assert evidence.trend_consistency == pytest.approx(1.0)
    assert evidence.fit_quality == pytest.approx(1.0)
    assert evidence.approach_confidence > 0.9
    assert evidence.recede_confidence == pytest.approx(0.0)


def test_monotonic_shrink_estimates_recede_evidence():
    evidence = estimate_motion(make_sequence(expansion_rate=-0.5), MotionConfig())

    assert evidence.expansion_rate == pytest.approx(-0.5, rel=1e-6)
    assert evidence.recede_confidence > 0.9
    assert evidence.approach_confidence == pytest.approx(0.0)


def test_insufficient_history_has_zero_radial_confidence():
    evidence = estimate_motion(
        make_sequence(expansion_rate=0.8, count=3), MotionConfig(min_samples=5)
    )

    assert evidence.sample_count == 3
    assert evidence.approach_confidence == 0.0
    assert evidence.recede_confidence == 0.0
