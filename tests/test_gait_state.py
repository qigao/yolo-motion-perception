from test_gait import make_history

from yolo_motion.flow_types import ArticulatedFlowEvidence, RegionFlowEvidence
from yolo_motion.gait import GaitConfig, GaitEvidence, estimate_gait
from yolo_motion.gait_state import LocomotionState, classify_gait


def config() -> GaitConfig:
    return GaitConfig(
        history_seconds=2.0,
        min_samples=12,
        min_duration=0.8,
        min_quality=0.55,
        standing_energy_threshold=0.008,
        min_periodicity=0.55,
        min_bilateral_correlation=0.45,
        min_leg_support_fraction=0.70,
        walking_cadence_min_hz=0.7,
        walking_cadence_max_hz=2.4,
        running_cadence_min_hz=2.2,
    )


def test_standing_noise_classifies_standing():
    evidence = estimate_gait(make_history(1.2, amplitude=0.002), config())
    assert classify_gait(evidence, config()) is LocomotionState.STANDING


def test_antiphase_walking_classifies_walking():
    evidence = estimate_gait(make_history(1.5, amplitude=0.04), config())
    assert classify_gait(evidence, config()) is LocomotionState.WALKING


def test_fast_periodic_gait_classifies_running():
    evidence = estimate_gait(make_history(3.0, amplitude=0.06), config())
    assert classify_gait(evidence, config()) is LocomotionState.RUNNING


def test_low_quality_history_fails_closed_to_unknown():
    evidence = estimate_gait(make_history(1.5, quality=0.25), config())
    assert classify_gait(evidence, config()) is LocomotionState.UNKNOWN


def test_one_leg_missing_fails_closed_to_unknown():
    evidence = estimate_gait(make_history(1.5, right_support=False), config())
    assert classify_gait(evidence, config()) is LocomotionState.UNKNOWN


def test_irregular_nonperiodic_motion_fails_closed_to_unknown():
    evidence = estimate_gait(make_history(1.5, irregular=True), config())
    assert classify_gait(evidence, config()) is LocomotionState.UNKNOWN


def test_insufficient_history_fails_closed_to_unknown():
    short = make_history(1.5, seconds=0.4)
    evidence = estimate_gait(short, config())
    assert classify_gait(evidence, config()) is LocomotionState.UNKNOWN


def test_classification_is_pure_for_explicit_evidence():
    evidence = GaitEvidence(
        sample_count=40,
        duration=2.0,
        left_support_fraction=1.0,
        right_support_fraction=1.0,
        left_energy=0.04,
        right_energy=0.04,
        periodicity=0.8,
        bilateral_correlation=0.9,
        phase_lag_seconds=0.33,
        cadence_hz=1.5,
        articulated_amplitude=0.04,
        temporal_consistency=0.9,
        quality=0.9,
    )
    assert classify_gait(evidence, config()) is LocomotionState.WALKING


def test_zero_residual_with_bilateral_support_is_standing_not_unknown():
    observations: list[ArticulatedFlowEvidence] = []
    for index in range(40):
        start = index * 0.05
        end = (index + 1) * 0.05
        zero = RegionFlowEvidence(0.0, 0.0, 0.0, 1.0)
        observations.append(
            ArticulatedFlowEvidence(
                track_id=7,
                start_timestamp=start,
                end_timestamp=end,
                torso_dx=2.0,
                torso_dy=0.0,
                normalized_torso_dx=0.01,
                normalized_torso_dy=0.0,
                region_flow={
                    "left_calf": zero,
                    "left_foot": zero,
                    "right_calf": zero,
                    "right_foot": zero,
                },
                pose_flow_agreement=0.95,
                person_height_px=160.0,
                quality=0.95,
            )
        )

    evidence = estimate_gait(observations, config())
    assert classify_gait(evidence, config()) is LocomotionState.STANDING
