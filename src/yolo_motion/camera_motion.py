from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .flow_types import CameraMotionEstimate, FlowObservation


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised by optional runtime only
        raise RuntimeError(
            "OpenCV is required for camera-motion estimation; "
            "install yolo-motion-perception[optical]"
        ) from exc
    return cv2


@dataclass(frozen=True)
class CameraMotionConfig:
    mode: str = "fixed"
    min_background_features: int = 20
    min_inlier_ratio: float = 0.60
    max_residual_error: float = 2.0
    max_features: int = 500
    feature_quality: float = 0.01
    feature_min_distance: float = 5.0
    ransac_reproj_threshold: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "affine"}:
            raise ValueError("camera motion mode must be fixed or affine")
        if self.min_background_features < 3:
            raise ValueError("minimum background features must be at least 3")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("minimum inlier ratio must be in [0, 1]")
        if not math.isfinite(self.max_residual_error) or self.max_residual_error <= 0.0:
            raise ValueError("maximum residual error must be positive and finite")
        if self.max_features < self.min_background_features:
            raise ValueError("maximum features must cover minimum background features")
        if not math.isfinite(self.feature_quality) or not 0.0 < self.feature_quality <= 1.0:
            raise ValueError("feature quality must be in (0, 1]")
        if not math.isfinite(self.feature_min_distance) or self.feature_min_distance <= 0.0:
            raise ValueError("feature minimum distance must be positive and finite")
        if (
            not math.isfinite(self.ransac_reproj_threshold)
            or self.ransac_reproj_threshold <= 0.0
        ):
            raise ValueError("RANSAC reprojection threshold must be positive and finite")


def _identity_matrix() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)


def _invalid_affine(matched_count: int = 0) -> CameraMotionEstimate:
    return CameraMotionEstimate(
        model="affine",
        matrix=_identity_matrix(),
        matched_count=matched_count,
        inlier_count=0,
        inlier_ratio=0.0,
        residual_error=0.0,
        quality=0.0,
        valid=False,
    )


def _as_gray_u8(frame: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        code = cv2.COLOR_BGR2GRAY if array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(array, code)
    else:
        raise ValueError("camera frames must be grayscale or 3/4-channel images")
    if not np.issubdtype(gray.dtype, np.number) or not np.isfinite(gray).all():
        raise ValueError("camera frame values must be finite numeric values")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _background_mask(
    shape: tuple[int, int], person_masks: Sequence[np.ndarray]
) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    for person_mask in person_masks:
        current = np.asarray(person_mask)
        if current.shape != shape:
            raise ValueError("person mask shape must match camera frame shape")
        mask[np.asarray(current, dtype=bool)] = 0
    return mask


def estimate_camera_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    person_masks: Sequence[np.ndarray],
    config: CameraMotionConfig,
) -> CameraMotionEstimate:
    previous = _as_gray_u8(previous_gray)
    current = _as_gray_u8(current_gray)
    if previous.shape != current.shape:
        raise ValueError("previous and current camera frame shape must match")

    if config.mode == "fixed":
        return CameraMotionEstimate(
            model="identity",
            matrix=_identity_matrix(),
            matched_count=0,
            inlier_count=0,
            inlier_ratio=1.0,
            residual_error=0.0,
            quality=1.0,
            valid=True,
        )

    cv2 = _import_cv2()
    background = _background_mask(previous.shape, person_masks)
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=config.max_features,
        qualityLevel=config.feature_quality,
        minDistance=config.feature_min_distance,
        mask=background,
    )
    if points is None or len(points) < config.min_background_features:
        return _invalid_affine(0 if points is None else len(points))

    tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return _invalid_affine(0)

    source = points.reshape(-1, 2)
    destination = tracked.reshape(-1, 2)
    good = status.reshape(-1).astype(bool)
    good &= np.isfinite(source).all(axis=1) & np.isfinite(destination).all(axis=1)
    source = source[good]
    destination = destination[good]
    matched_count = len(source)
    if matched_count < config.min_background_features:
        return _invalid_affine(matched_count)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None or not np.isfinite(matrix).all():
        return _invalid_affine(matched_count)

    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / matched_count)
    if inlier_count:
        homogeneous = np.column_stack((source, np.ones(matched_count, dtype=float)))
        predicted = homogeneous @ np.asarray(matrix, dtype=float).T
        errors = np.linalg.norm(predicted[inliers] - destination[inliers], axis=1)
        residual_error = float(np.mean(errors))
    else:
        residual_error = config.max_residual_error

    residual_factor = max(0.0, 1.0 - residual_error / config.max_residual_error)
    quality = float(np.clip(inlier_ratio * residual_factor, 0.0, 1.0))
    valid = (
        matched_count >= config.min_background_features
        and inlier_ratio >= config.min_inlier_ratio
        and residual_error <= config.max_residual_error
    )
    return CameraMotionEstimate(
        model="affine",
        matrix=np.asarray(matrix, dtype=float),
        matched_count=matched_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        residual_error=residual_error,
        quality=quality,
        valid=valid,
    )


def compensate_flow(
    flow: FlowObservation,
    camera: CameraMotionEstimate,
) -> FlowObservation:
    if not camera.valid:
        raise ValueError("invalid camera motion estimate cannot compensate flow")

    height, width = flow.dx.shape
    yy, xx = np.mgrid[:height, :width]
    matrix = camera.matrix
    projected_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    projected_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    camera_dx = projected_x - xx
    camera_dy = projected_y - yy

    return FlowObservation(
        start_timestamp=flow.start_timestamp,
        end_timestamp=flow.end_timestamp,
        dx=flow.dx - camera_dx,
        dy=flow.dy - camera_dy,
        valid=flow.valid,
        backend=f"{flow.backend}|camera:{camera.model}",
    )
