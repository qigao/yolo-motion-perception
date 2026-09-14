from __future__ import annotations

from .motion import MotionConfig
from .types import LateralState, MotionEvidence, MotionState, RadialState


def classify_motion(evidence: MotionEvidence, config: MotionConfig) -> MotionState:
    lateral = (
        LateralState.MOVING
        if evidence.speed >= config.stationary_speed_threshold
        else LateralState.STATIONARY
    )

    enough_history = (
        evidence.sample_count >= config.min_samples and evidence.duration >= config.min_duration
    )
    if not enough_history:
        radial = RadialState.UNKNOWN
    elif (
        evidence.expansion_rate >= config.radial_rate_threshold
        and evidence.approach_confidence >= config.min_radial_confidence
        and evidence.trend_consistency >= config.min_trend_consistency
    ):
        radial = RadialState.APPROACHING
    elif (
        evidence.expansion_rate <= -config.radial_rate_threshold
        and evidence.recede_confidence >= config.min_radial_confidence
        and evidence.trend_consistency >= config.min_trend_consistency
    ):
        radial = RadialState.RECEDING
    else:
        radial = RadialState.STABLE

    return MotionState(lateral=lateral, radial=radial)
