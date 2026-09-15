import math

import pytest

from yolo_motion.flow_types import ArticulatedFlowEvidence, RegionFlowEvidence
from yolo_motion.gait import GaitConfig, estimate_gait

SAMPLE_RATE_HZ = 25.0
DT = 1.0 / SAMPLE_RATE_HZ


def _history() -> list[ArticulatedFlowEvidence]:
    observations: list[ArticulatedFlowEvidence] = []
    for index in range(50):
        start = index * DT
        end = (index + 1) * DT
        fundamental = math.sin(2.0 * math.pi * 0.8 * end)
        ripple = 0.5 * math.sin(2.0 * math.pi * 3.2 * end + 0.4)
        left = 0.04 * (fundamental + ripple)
        right = -left
        region_flow = {
            "left_thigh": RegionFlowEvidence(left * 0.55, 0.0, abs(left * 0.55), 1.0),
            "left_calf": RegionFlowEvidence(left, 0.0, abs(left), 1.0),
            "left_foot": RegionFlowEvidence(left * 1.15, 0.0, abs(left * 1.15), 1.0),
            "right_thigh": RegionFlowEvidence(right * 0.55, 0.0, abs(right * 0.55), 1.0),
            "right_calf": RegionFlowEvidence(right, 0.0, abs(right), 1.0),
            "right_foot": RegionFlowEvidence(right * 1.15, 0.0, abs(right * 1.15), 1.0),
        }
        observations.append(
            ArticulatedFlowEvidence(
                track_id=9,
                start_timestamp=start,
                end_timestamp=end,
                torso_dx=0.0,
                torso_dy=0.0,
                normalized_torso_dx=0.0,
                normalized_torso_dy=0.0,
                region_flow=region_flow,
                pose_flow_agreement=0.95,
                person_height_px=180.0,
                quality=0.95,
            )
        )
    return observations


def test_gait_uses_strongest_plausible_autocorrelation_peak():
    evidence = estimate_gait(_history(), GaitConfig())

    assert evidence.cadence_hz == pytest.approx(0.8, abs=0.10)
    assert evidence.periodicity > 0.80
    assert evidence.bilateral_correlation > 0.45
