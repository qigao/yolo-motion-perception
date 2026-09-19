from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from yolo_motion.flow_types import (
    ArticulatedFlowEvidence,
    CameraMotionEstimate,
    FlowObservation,
    RegionFlowEvidence,
)


def make_flow(*, dx=None, dy=None, valid=None, start=1.0, end=1.1):
    if dx is None:
        dx = np.ones((4, 5), dtype=float)
    if dy is None:
        dy = np.zeros((4, 5), dtype=float)
    if valid is None:
        valid = np.ones((4, 5), dtype=bool)
    return FlowObservation(
        start_timestamp=start,
        end_timestamp=end,
        dx=dx,
        dy=dy,
        valid=valid,
        backend="synthetic",
    )


def test_flow_observation_copies_and_freezes_fields():
    dx = np.ones((4, 5), dtype=float)
    dy = np.zeros((4, 5), dtype=float)
    valid = np.ones((4, 5), dtype=bool)
    flow = make_flow(dx=dx, dy=dy, valid=valid)

    dx[0, 0] = 99.0
    valid[0, 0] = False

    assert flow.dx[0, 0] == pytest.approx(1.0)
    assert bool(flow.valid[0, 0]) is True
    assert not flow.dx.flags.writeable
    assert not flow.dy.flags.writeable
    assert not flow.valid.flags.writeable

    with pytest.raises(FrozenInstanceError):
        flow.backend = "other"


def test_flow_observation_rejects_invalid_time_or_shapes():
    with pytest.raises(ValueError, match="timestamp"):
        make_flow(start=1.0, end=1.0)

    with pytest.raises(ValueError, match="shape"):
        make_flow(dy=np.zeros((3, 5), dtype=float))

    with pytest.raises(ValueError, match="valid"):
        make_flow(valid=np.ones((4, 4), dtype=bool))


def test_flow_observation_rejects_non_finite_vectors():
    dx = np.ones((4, 5), dtype=float)
    dx[2, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        make_flow(dx=dx)


def test_camera_motion_estimate_validates_matrix_and_quality():
    camera = CameraMotionEstimate(
        model="identity",
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        matched_count=30,
        inlier_count=30,
        inlier_ratio=1.0,
        residual_error=0.0,
        quality=1.0,
        valid=True,
    )
    assert camera.matrix.shape == (2, 3)
    assert not camera.matrix.flags.writeable

    with pytest.raises(ValueError, match="2x3"):
        CameraMotionEstimate(
            model="affine",
            matrix=np.eye(3),
            matched_count=20,
            inlier_count=10,
            inlier_ratio=0.5,
            residual_error=1.0,
            quality=0.5,
            valid=True,
        )

    with pytest.raises(ValueError, match="quality"):
        CameraMotionEstimate(
            model="affine",
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            matched_count=20,
            inlier_count=10,
            inlier_ratio=0.5,
            residual_error=1.0,
            quality=1.1,
            valid=True,
        )


def test_articulated_evidence_preserves_partial_regions_but_freezes_mapping():
    left = RegionFlowEvidence(dx=0.02, dy=-0.01, energy=0.03, valid_fraction=0.9)
    evidence = ArticulatedFlowEvidence(
        track_id=7,
        start_timestamp=1.0,
        end_timestamp=1.1,
        torso_dx=2.0,
        torso_dy=0.0,
        normalized_torso_dx=0.02,
        normalized_torso_dy=0.0,
        region_flow={"left_calf": left},
        pose_flow_agreement=0.8,
        person_height_px=100.0,
        quality=0.75,
    )

    assert set(evidence.region_flow) == {"left_calf"}
    assert evidence.region_flow["left_calf"] == left
    with pytest.raises(TypeError):
        evidence.region_flow["right_calf"] = left


def test_articulated_evidence_rejects_invalid_height_or_quality():
    region = RegionFlowEvidence(dx=0.0, dy=0.0, energy=0.0, valid_fraction=1.0)

    with pytest.raises(ValueError, match="height"):
        ArticulatedFlowEvidence(
            track_id=1,
            start_timestamp=1.0,
            end_timestamp=1.1,
            torso_dx=0.0,
            torso_dy=0.0,
            normalized_torso_dx=0.0,
            normalized_torso_dy=0.0,
            region_flow={"left_calf": region},
            pose_flow_agreement=1.0,
            person_height_px=0.0,
            quality=1.0,
        )

    with pytest.raises(ValueError, match="quality"):
        ArticulatedFlowEvidence(
            track_id=1,
            start_timestamp=1.0,
            end_timestamp=1.1,
            torso_dx=0.0,
            torso_dy=0.0,
            normalized_torso_dx=0.0,
            normalized_torso_dy=0.0,
            region_flow={"left_calf": region},
            pose_flow_agreement=1.0,
            person_height_px=100.0,
            quality=1.2,
        )
