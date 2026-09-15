import numpy as np
import pytest

from yolo_motion.camera_motion import (
    CameraMotionConfig,
    compensate_flow,
    estimate_camera_motion,
)
from yolo_motion.flow_types import CameraMotionEstimate, FlowObservation


def textured_frame(height=128, width=160):
    rng = np.random.default_rng(424242)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def translation_matrix(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=float)


def test_fixed_camera_mode_returns_valid_identity():
    frame = textured_frame()
    estimate = estimate_camera_motion(
        frame,
        frame.copy(),
        person_masks=[],
        config=CameraMotionConfig(mode="fixed"),
    )

    assert estimate.valid is True
    assert estimate.model == "identity"
    np.testing.assert_allclose(estimate.matrix, translation_matrix(0.0, 0.0))
    assert estimate.quality == pytest.approx(1.0)


def test_affine_camera_mode_recovers_background_translation():
    previous = textured_frame()
    current = np.roll(previous, shift=(2, 3), axis=(0, 1))
    estimate = estimate_camera_motion(
        previous,
        current,
        person_masks=[],
        config=CameraMotionConfig(
            mode="affine",
            min_background_features=20,
            min_inlier_ratio=0.60,
            max_residual_error=2.0,
        ),
    )

    assert estimate.valid is True
    assert estimate.model == "affine"
    assert estimate.matched_count >= 20
    assert estimate.inlier_count <= estimate.matched_count
    assert estimate.matrix[0, 2] == pytest.approx(3.0, abs=0.6)
    assert estimate.matrix[1, 2] == pytest.approx(2.0, abs=0.6)


def test_person_mask_excludes_independent_foreground_motion_from_affine_fit():
    previous = textured_frame()
    current = np.roll(previous, shift=(1, 2), axis=(0, 1))

    # Overwrite a large person-like region with unrelated high-contrast motion.
    current = current.copy()
    current[30:100, 50:120] = np.roll(previous[30:100, 50:120], shift=15, axis=1)
    person_mask = np.zeros_like(previous, dtype=bool)
    person_mask[24:106, 44:126] = True

    estimate = estimate_camera_motion(
        previous,
        current,
        person_masks=[person_mask],
        config=CameraMotionConfig(
            mode="affine",
            min_background_features=15,
            min_inlier_ratio=0.50,
            max_residual_error=2.0,
        ),
    )

    assert estimate.valid is True
    assert estimate.matrix[0, 2] == pytest.approx(2.0, abs=0.8)
    assert estimate.matrix[1, 2] == pytest.approx(1.0, abs=0.8)


def test_affine_camera_mode_fails_closed_with_insufficient_background_features():
    blank = np.zeros((64, 64), dtype=np.uint8)
    estimate = estimate_camera_motion(
        blank,
        blank.copy(),
        person_masks=[],
        config=CameraMotionConfig(mode="affine", min_background_features=10),
    )

    assert estimate.valid is False
    assert estimate.model == "affine"
    assert estimate.matched_count < 10


def test_compensate_flow_subtracts_dense_camera_translation():
    flow = FlowObservation(
        start_timestamp=1.0,
        end_timestamp=1.1,
        dx=np.full((3, 4), 5.0),
        dy=np.full((3, 4), 3.0),
        valid=np.ones((3, 4), dtype=bool),
        backend="synthetic",
    )
    camera = CameraMotionEstimate(
        model="affine",
        matrix=translation_matrix(2.0, 1.0),
        matched_count=30,
        inlier_count=28,
        inlier_ratio=28 / 30,
        residual_error=0.1,
        quality=0.9,
        valid=True,
    )

    compensated = compensate_flow(flow, camera)

    np.testing.assert_allclose(compensated.dx, 3.0)
    np.testing.assert_allclose(compensated.dy, 2.0)
    np.testing.assert_array_equal(compensated.valid, flow.valid)
    assert "camera:affine" in compensated.backend


def test_compensate_flow_rejects_invalid_camera_estimate_instead_of_falling_back():
    flow = FlowObservation(
        start_timestamp=1.0,
        end_timestamp=1.1,
        dx=np.ones((2, 2)),
        dy=np.ones((2, 2)),
        valid=None,
        backend="synthetic",
    )
    camera = CameraMotionEstimate(
        model="affine",
        matrix=translation_matrix(0.0, 0.0),
        matched_count=0,
        inlier_count=0,
        inlier_ratio=0.0,
        residual_error=0.0,
        quality=0.0,
        valid=False,
    )

    with pytest.raises(ValueError, match="invalid camera"):
        compensate_flow(flow, camera)


def test_camera_motion_config_rejects_invalid_mode_and_quality_thresholds():
    with pytest.raises(ValueError):
        CameraMotionConfig(mode="homography")
    with pytest.raises(ValueError):
        CameraMotionConfig(min_inlier_ratio=1.1)
    with pytest.raises(ValueError):
        CameraMotionConfig(max_residual_error=0.0)
