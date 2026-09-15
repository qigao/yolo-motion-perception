from __future__ import annotations

from enum import Enum

from .gait import GaitConfig, GaitEvidence


class LocomotionState(str, Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"


def classify_gait(evidence: GaitEvidence, config: GaitConfig) -> LocomotionState:
    if evidence.sample_count < config.min_samples:
        return LocomotionState.UNKNOWN
    if evidence.duration < config.min_duration:
        return LocomotionState.UNKNOWN
    if evidence.quality < config.min_quality:
        return LocomotionState.UNKNOWN
    if evidence.left_support_fraction < config.min_leg_support_fraction:
        return LocomotionState.UNKNOWN
    if evidence.right_support_fraction < config.min_leg_support_fraction:
        return LocomotionState.UNKNOWN

    if evidence.articulated_amplitude <= config.standing_energy_threshold:
        return LocomotionState.STANDING

    if evidence.periodicity < config.min_periodicity:
        return LocomotionState.UNKNOWN
    if evidence.bilateral_correlation < config.min_bilateral_correlation:
        return LocomotionState.UNKNOWN

    if evidence.cadence_hz >= config.running_cadence_min_hz:
        return LocomotionState.RUNNING
    if config.walking_cadence_min_hz <= evidence.cadence_hz <= config.walking_cadence_max_hz:
        return LocomotionState.WALKING
    return LocomotionState.UNKNOWN
