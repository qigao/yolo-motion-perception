import math

from yolo_motion.flow_types import ArticulatedFlowEvidence, RegionFlowEvidence
from yolo_motion.gait import GaitConfig
from yolo_motion.gait_pipeline import OpticalGaitPipeline
from yolo_motion.gait_state import LocomotionState

DT = 0.05


def gait_config() -> GaitConfig:
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


def gait_observation(
    track_id: int,
    index: int,
    *,
    stride_cycle_hz: float = 0.75,
    amplitude: float = 0.04,
    quality: float = 0.95,
) -> ArticulatedFlowEvidence:
    start = index * DT
    end = (index + 1) * DT
    left = amplitude * math.sin(2.0 * math.pi * stride_cycle_hz * end)
    right = -left
    region_flow = {
        "left_thigh": RegionFlowEvidence(left * 0.55, 0.0, abs(left * 0.55), 1.0),
        "left_calf": RegionFlowEvidence(left, 0.0, abs(left), 1.0),
        "left_foot": RegionFlowEvidence(left * 1.15, 0.0, abs(left * 1.15), 1.0),
        "right_thigh": RegionFlowEvidence(right * 0.55, 0.0, abs(right * 0.55), 1.0),
        "right_calf": RegionFlowEvidence(right, 0.0, abs(right), 1.0),
        "right_foot": RegionFlowEvidence(right * 1.15, 0.0, abs(right * 1.15), 1.0),
    }
    return ArticulatedFlowEvidence(
        track_id=track_id,
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


def feed(
    pipeline: OpticalGaitPipeline,
    track_id: int,
    *,
    amplitude: float,
    sample_count: int = 40,
):
    result = None
    for index in range(sample_count):
        result = pipeline.update(gait_observation(track_id, index, amplitude=amplitude))
    return result


def test_pipeline_keeps_independent_histories_per_tracker_id():
    pipeline = OpticalGaitPipeline(gait_config())
    walking_result = None
    standing_result = None

    for index in range(40):
        walking_result = pipeline.update(gait_observation(11, index, amplitude=0.04))
        standing_result = pipeline.update(gait_observation(22, index, amplitude=0.002))

    assert walking_result is not None
    assert standing_result is not None
    assert walking_result.track_id == 11
    assert walking_result.state is LocomotionState.WALKING
    assert standing_result.track_id == 22
    assert standing_result.state is LocomotionState.STANDING
    assert pipeline.track_ids() == {11, 22}


def test_pipeline_returns_none_until_minimum_temporal_history_is_met():
    pipeline = OpticalGaitPipeline(gait_config())

    result = None
    for index in range(15):
        result = pipeline.update(gait_observation(5, index))
    assert result is None

    result = pipeline.update(gait_observation(5, 15))
    assert result is not None
    assert result.state is LocomotionState.WALKING


def test_drop_stale_removes_only_inactive_histories():
    pipeline = OpticalGaitPipeline(gait_config())
    feed(pipeline, 1, amplitude=0.04, sample_count=20)
    feed(pipeline, 2, amplitude=0.04, sample_count=20)

    assert pipeline.drop_stale({2}) == {1}
    assert pipeline.track_ids() == {2}


def test_track_id_reuse_after_drop_starts_with_clean_history():
    pipeline = OpticalGaitPipeline(gait_config())

    walking = feed(pipeline, 9, amplitude=0.04)
    assert walking is not None
    assert walking.state is LocomotionState.WALKING

    assert pipeline.drop_stale(set()) == {9}
    assert pipeline.track_ids() == set()

    standing = feed(pipeline, 9, amplitude=0.002)
    assert standing is not None
    assert standing.state is LocomotionState.STANDING


def test_pipeline_rejects_overlapping_or_reversed_intervals_per_track():
    pipeline = OpticalGaitPipeline(gait_config())
    pipeline.update(gait_observation(3, 0))

    duplicate = ArticulatedFlowEvidence(
        track_id=3,
        start_timestamp=0.0,
        end_timestamp=0.05,
        torso_dx=0.0,
        torso_dy=0.0,
        normalized_torso_dx=0.0,
        normalized_torso_dy=0.0,
        region_flow=gait_observation(3, 0).region_flow,
        pose_flow_agreement=0.95,
        person_height_px=160.0,
        quality=0.95,
    )

    try:
        pipeline.update(duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping gait intervals must be rejected")
