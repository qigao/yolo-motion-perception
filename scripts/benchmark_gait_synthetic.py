from __future__ import annotations

import json
import math
import sys

import numpy as np

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.camera_motion import compensate_flow
from yolo_motion.flow_types import (
    ArticulatedFlowEvidence,
    CameraMotionEstimate,
    FlowObservation,
    RegionFlowEvidence,
)
from yolo_motion.gait import GaitConfig
from yolo_motion.gait_pipeline import OpticalGaitPipeline, OpticalGaitResult
from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline, TrackMotionResult
from yolo_motion.pose_types import PoseLayout, PoseObservation
from yolo_motion.types import TrackObservation

_SAMPLE_RATE_HZ = 20.0
_DT = 1.0 / _SAMPLE_RATE_HZ


def _gait_config() -> GaitConfig:
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


def _gait_observation(
    track_id: int,
    index: int,
    *,
    frequency_hz: float,
    amplitude: float,
    torso_dx: float = 0.0,
    torso_dy: float = 0.0,
) -> ArticulatedFlowEvidence:
    start = index * _DT
    end = (index + 1) * _DT
    left = amplitude * math.sin(2.0 * math.pi * frequency_hz * end)
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
        torso_dx=torso_dx,
        torso_dy=torso_dy,
        normalized_torso_dx=torso_dx / 160.0,
        normalized_torso_dy=torso_dy / 160.0,
        region_flow=region_flow,
        pose_flow_agreement=0.95,
        person_height_px=160.0,
        quality=0.95,
    )


def _run_gait(
    *,
    frequency_hz: float,
    amplitude: float,
    torso_dx: float = 0.0,
    torso_dy: float = 0.0,
) -> OpticalGaitResult:
    pipeline = OpticalGaitPipeline(_gait_config())
    result = None
    for index in range(40):
        result = pipeline.update(
            _gait_observation(
                1,
                index,
                frequency_hz=frequency_hz,
                amplitude=amplitude,
                torso_dx=torso_dx,
                torso_dy=torso_dy,
            )
        )
    if result is None:
        raise RuntimeError("synthetic gait sequence did not produce enough history")
    return result


def _run_v1(*, vx: float = 0.0, expansion_rate: float = 0.0) -> TrackMotionResult:
    pipeline = MotionPipeline(MotionConfig())
    result = None
    for index in range(8):
        timestamp = index * 0.1
        side_scale = math.exp(expansion_rate * timestamp / 2.0)
        result = pipeline.update(
            TrackObservation(
                track_id=1,
                timestamp=timestamp,
                class_id=0,
                confidence=0.95,
                cx=0.4 + vx * timestamp,
                cy=0.5,
                width=0.1 * side_scale,
                height=0.2 * side_scale,
            )
        )
    if result is None:
        raise RuntimeError("synthetic V1 sequence did not produce enough history")
    return result


def _summary(
    gait: OpticalGaitResult,
    motion: TrackMotionResult,
    *,
    camera_compensated: bool = False,
) -> dict[str, object]:
    return {
        "locomotion": gait.state.value,
        "cadence_hz": round(gait.evidence.cadence_hz, 6),
        "articulated_amplitude": round(gait.evidence.articulated_amplitude, 6),
        "gait_quality": round(gait.evidence.quality, 6),
        "lateral": motion.state.lateral.value,
        "radial": motion.state.radial.value,
        "expansion_rate": round(motion.evidence.expansion_rate, 6),
        "camera_compensated": camera_compensated,
    }


def _too_small_unknown() -> dict[str, object]:
    frame_size = 100
    xy = np.full((133, 2), 0.5, dtype=float)
    confidence = np.full(133, 0.95, dtype=float)
    previous = PoseObservation(
        track_id=1,
        timestamp=0.0,
        frame_width=frame_size,
        frame_height=frame_size,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )
    current = PoseObservation(
        track_id=1,
        timestamp=0.05,
        frame_width=frame_size,
        frame_height=frame_size,
        xy=xy,
        confidence=confidence,
        layout=PoseLayout.COCO_WHOLEBODY_133,
    )
    track = TrackObservation(
        track_id=1,
        timestamp=0.05,
        class_id=0,
        confidence=0.95,
        cx=0.5,
        cy=0.5,
        width=0.15,
        height=0.30,
    )
    flow = FlowObservation(
        start_timestamp=0.0,
        end_timestamp=0.05,
        dx=np.zeros((frame_size, frame_size), dtype=float),
        dy=np.zeros((frame_size, frame_size), dtype=float),
        valid=np.ones((frame_size, frame_size), dtype=bool),
        backend="synthetic-too-small",
    )
    evidence = estimate_articulated_flow(
        track,
        previous,
        current,
        {},
        flow,
        ArticulatedFlowConfig(min_person_height_px=40.0),
    )
    if evidence is not None:
        raise RuntimeError("too-small synthetic target unexpectedly produced articulated evidence")
    motion = _run_v1()
    return {
        "locomotion": "unknown",
        "cadence_hz": 0.0,
        "articulated_amplitude": 0.0,
        "gait_quality": 0.0,
        "lateral": motion.state.lateral.value,
        "radial": motion.state.radial.value,
        "expansion_rate": round(motion.evidence.expansion_rate, 6),
        "camera_compensated": False,
    }


def _camera_translation_is_compensated() -> bool:
    flow = FlowObservation(
        start_timestamp=0.0,
        end_timestamp=0.05,
        dx=np.full((16, 16), 3.0, dtype=float),
        dy=np.full((16, 16), -2.0, dtype=float),
        valid=np.ones((16, 16), dtype=bool),
        backend="synthetic-camera-translation",
    )
    camera = CameraMotionEstimate(
        model="affine",
        matrix=np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=float),
        matched_count=100,
        inlier_count=100,
        inlier_ratio=1.0,
        residual_error=0.0,
        quality=1.0,
        valid=True,
    )
    compensated = compensate_flow(flow, camera)
    return bool(
        np.max(np.abs(compensated.dx)) <= 1e-12
        and np.max(np.abs(compensated.dy)) <= 1e-12
    )


def run_benchmark() -> dict[str, dict[str, object]]:
    camera_compensated = _camera_translation_is_compensated()
    if not camera_compensated:
        raise RuntimeError("synthetic camera translation was not removed")

    return {
        "standing": _summary(_run_gait(frequency_hz=0.6, amplitude=0.002), _run_v1()),
        "walking_in_place": _summary(_run_gait(frequency_hz=0.75, amplitude=0.04), _run_v1()),
        "walking_transverse": _summary(
            _run_gait(frequency_hz=0.75, amplitude=0.04),
            _run_v1(vx=0.08),
        ),
        "walking_approaching": _summary(
            _run_gait(frequency_hz=0.75, amplitude=0.04),
            _run_v1(expansion_rate=0.5),
        ),
        "walking_receding": _summary(
            _run_gait(frequency_hz=0.75, amplitude=0.04),
            _run_v1(expansion_rate=-0.5),
        ),
        "running": _summary(_run_gait(frequency_hz=1.5, amplitude=0.06), _run_v1()),
        "rigid_translation_control": _summary(
            _run_gait(frequency_hz=0.75, amplitude=0.0, torso_dx=6.0, torso_dy=2.0),
            _run_v1(vx=0.08),
        ),
        "too_small_unknown": _too_small_unknown(),
        "camera_translation_compensated": _summary(
            _run_gait(frequency_hz=0.75, amplitude=0.04),
            _run_v1(),
            camera_compensated=camera_compensated,
        ),
    }


def _validate(summary: dict[str, dict[str, object]]) -> list[str]:
    expected_locomotion = {
        "standing": "standing",
        "walking_in_place": "walking",
        "walking_transverse": "walking",
        "walking_approaching": "walking",
        "walking_receding": "walking",
        "running": "running",
        "rigid_translation_control": "standing",
        "too_small_unknown": "unknown",
        "camera_translation_compensated": "walking",
    }
    failures = [
        f"{name}: expected locomotion {expected}, got {summary[name]['locomotion']}"
        for name, expected in expected_locomotion.items()
        if summary[name]["locomotion"] != expected
    ]
    expected_motion = {
        "walking_in_place": ("stationary", "stable"),
        "walking_transverse": ("moving", "stable"),
        "walking_approaching": ("stationary", "approaching"),
        "walking_receding": ("stationary", "receding"),
    }
    for name, expected in expected_motion.items():
        actual = (summary[name]["lateral"], summary[name]["radial"])
        if actual != expected:
            failures.append(f"{name}: expected V1 motion {expected}, got {actual}")
    if not summary["camera_translation_compensated"]["camera_compensated"]:
        failures.append("camera_translation_compensated: camera flow was not removed")
    return failures


def main() -> int:
    summary = run_benchmark()
    print(json.dumps(summary, indent=2, sort_keys=True))
    failures = _validate(summary)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
