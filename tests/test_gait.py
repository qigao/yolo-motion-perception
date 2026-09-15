import math

import pytest

from yolo_motion.flow_types import ArticulatedFlowEvidence, RegionFlowEvidence
from yolo_motion.gait import GaitConfig, estimate_gait

SAMPLE_RATE_HZ = 20.0
DT = 1.0 / SAMPLE_RATE_HZ


def make_history(
    stride_cycle_hz: float,
    *,
    seconds: float = 2.0,
    amplitude: float = 0.04,
    quality: float = 0.95,
    right_support: bool = True,
    irregular: bool = False,
) -> list[ArticulatedFlowEvidence]:
    count = int(seconds * SAMPLE_RATE_HZ)
    observations: list[ArticulatedFlowEvidence] = []
    for index in range(count):
        start = index * DT
        end = (index + 1) * DT
        phase = 2.0 * math.pi * stride_cycle_hz * end
        if irregular:
            left = amplitude * math.sin(phase + 0.37 * index * index)
        else:
            left = amplitude * math.sin(phase)
        right = -left

        region_flow = {
            "left_thigh": RegionFlowEvidence(left * 0.55, 0.0, abs(left * 0.55), 1.0),
            "left_calf": RegionFlowEvidence(left, 0.0, abs(left), 1.0),
            "left_foot": RegionFlowEvidence(left * 1.15, 0.0, abs(left * 1.15), 1.0),
        }
        if right_support:
            region_flow.update(
                {
                    "right_thigh": RegionFlowEvidence(
                        right * 0.55, 0.0, abs(right * 0.55), 1.0
                    ),
                    "right_calf": RegionFlowEvidence(right, 0.0, abs(right), 1.0),
                    "right_foot": RegionFlowEvidence(
                        right * 1.15, 0.0, abs(right * 1.15), 1.0
                    ),
                }
            )

        observations.append(
            ArticulatedFlowEvidence(
                track_id=7,
                start_timestamp=start,
                end_timestamp=end,
                torso_dx=0.0,
                torso_dy=0.0,
                normalized_torso_dx=0.0,
                normalized_torso_dy=0.0,
                region_flow=region_flow,
                pose_flow_agreement=quality,
                person_height_px=160.0,
                quality=quality,
            )
        )
    return observations


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


def test_estimate_gait_reports_bilateral_step_cadence_for_walking():
    evidence = estimate_gait(make_history(0.75), config())

    assert evidence.sample_count == 40
    assert evidence.duration == pytest.approx(2.0, abs=1e-6)
    assert evidence.left_support_fraction == pytest.approx(1.0)
    assert evidence.right_support_fraction == pytest.approx(1.0)
    assert evidence.cadence_hz == pytest.approx(1.5, abs=0.15)
    assert evidence.periodicity >= 0.55
    assert evidence.bilateral_correlation >= 0.45
    assert abs(abs(evidence.phase_lag_seconds) - (1.0 / 1.5)) <= 0.08


def test_estimate_gait_reports_bilateral_step_cadence_for_running():
    evidence = estimate_gait(make_history(1.5, amplitude=0.06), config())

    assert evidence.cadence_hz == pytest.approx(3.0, abs=0.20)
    assert evidence.periodicity >= 0.55
    assert evidence.articulated_amplitude > 0.03


def test_translation_only_zero_residual_has_low_articulated_energy():
    evidence = estimate_gait(make_history(0.75, amplitude=0.0), config())

    assert evidence.left_support_fraction == pytest.approx(1.0)
    assert evidence.right_support_fraction == pytest.approx(1.0)
    assert evidence.left_energy < 1e-9
    assert evidence.right_energy < 1e-9
    assert evidence.articulated_amplitude < 1e-9


def test_one_leg_missing_tracks_support_fraction_without_fabrication():
    evidence = estimate_gait(make_history(0.75, right_support=False), config())

    assert evidence.left_support_fraction == pytest.approx(1.0)
    assert evidence.right_support_fraction == pytest.approx(0.0)
    assert evidence.left_energy > 0.0
    assert evidence.right_energy == pytest.approx(0.0)


def test_irregular_leg_motion_does_not_look_strongly_periodic():
    evidence = estimate_gait(make_history(0.75, irregular=True), config())

    assert evidence.periodicity < config().min_periodicity


def test_gait_config_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        GaitConfig(history_seconds=0.0)
    with pytest.raises(ValueError):
        GaitConfig(min_leg_support_fraction=1.1)
    with pytest.raises(ValueError):
        GaitConfig(walking_cadence_min_hz=2.5, walking_cadence_max_hz=2.0)
