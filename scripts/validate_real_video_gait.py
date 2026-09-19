from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
import ultralytics
import yaml
from rtmlib import RTMPose
from ultralytics import YOLO

from yolo_motion.articulated_flow import ArticulatedFlowConfig, estimate_articulated_flow
from yolo_motion.camera_motion import CameraMotionConfig, compensate_flow, estimate_camera_motion
from yolo_motion.flow_backend import OpenCvFarnebackBackend
from yolo_motion.gait import GaitConfig
from yolo_motion.gait_pipeline import OpticalGaitPipeline
from yolo_motion.motion import MotionConfig
from yolo_motion.pipeline import MotionPipeline
from yolo_motion.pose_regions import PoseRegionConfig, build_body_regions
from yolo_motion.rtmw_adapter import RtmwPoseAdapter
from yolo_motion.ultralytics_adapter import observations_from_result

KTH_BASE = "https://www.csc.kth.se/cvap/actions"
RTMW_MODEL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip"
)

SCENARIOS = {
    "standing": {
        "filename": "person15_handclapping_d1_uncomp.avi",
        "end_frame": 312,
        "expected": "standing",
    },
    "walking": {
        "filename": "person15_walking_d1_uncomp.avi",
        "end_frame": 741,
        "expected": "walking",
    },
    "jogging": {
        "filename": "person15_jogging_d1_uncomp.avi",
        "end_frame": 420,
        "expected": "running",
    },
}


class RtmLibCropInferencer:
    def __init__(self, pose_model: RTMPose):
        self.pose_model = pose_model

    def __call__(self, crop: np.ndarray, **_kwargs):
        keypoints, scores = self.pose_model(crop)
        if len(keypoints) == 0:
            yield {"predictions": [[]]}
            return
        sample = {
            "keypoints": np.asarray(keypoints[0], dtype=float).tolist(),
            "keypoint_scores": np.asarray(scores[0], dtype=float).reshape(-1).tolist(),
        }
        yield {"predictions": [[sample]]}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"identity file is missing or empty: {path}")
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _resolve_yolo_weights(model_name: str) -> Path:
    model = YOLO(model_name)
    candidates = [
        Path(model_name),
        Path(str(getattr(model, "ckpt_path", ""))),
        Path(str(getattr(model.model, "pt_path", ""))),
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"cannot resolve downloaded YOLO weights for {model_name}")


def _resolve_botsort_config() -> Path:
    package_root = Path(ultralytics.__file__).resolve().parent
    path = package_root / "cfg" / "trackers" / "botsort.yaml"
    if not path.is_file():
        raise RuntimeError(f"cannot resolve Ultralytics BoT-SORT config: {path}")
    return path


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "yolo-motion-perception-ci"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    if destination.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty file from {url}")


def _extract_rtmw_onnx(archive_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".onnx")
        )
        if len(members) != 1:
            raise RuntimeError(
                f"expected exactly one ONNX model in {archive_path}, got {members}"
            )
        member = members[0]
        output_path = destination_dir / Path(member).name
        with archive.open(member) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"extracted empty RTMW ONNX model: {output_path}")
    return output_path


def _load_gait_config(path: Path) -> GaitConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("gait config must be a YAML mapping")
    return GaitConfig(**raw)


def _largest_person(observations):
    people = [observation for observation in observations if observation.class_id == 0]
    if not people:
        return None
    return max(people, key=lambda observation: observation.area)


def _intervals(predictions: list[dict[str, object]], expected: str) -> list[dict[str, object]]:
    by_track: dict[int, list[float]] = defaultdict(list)
    for prediction in predictions:
        by_track[int(prediction["track_id"])].append(float(prediction["timestamp"]))
    return [
        {
            "track_id": track_id,
            "start": min(times),
            "end": max(times),
            "label": expected,
        }
        for track_id, times in sorted(by_track.items())
    ]


def _summarize(predictions: list[dict[str, object]], expected: str) -> dict[str, object]:
    counts = Counter(str(item["locomotion"]) for item in predictions)
    known_counts = Counter({state: count for state, count in counts.items() if state != "unknown"})
    known_total = sum(known_counts.values())
    dominant_known = known_counts.most_common(1)[0][0] if known_counts else None
    expected_known_fraction = (
        known_counts.get(expected, 0) / known_total if known_total else 0.0
    )
    passed = (
        len(predictions) >= 5
        and dominant_known == expected
        and expected_known_fraction >= 0.50
    )
    return {
        "expected": expected,
        "prediction_count": len(predictions),
        "state_counts": dict(sorted(counts.items())),
        "known_prediction_count": known_total,
        "unknown_prediction_count": counts.get("unknown", 0),
        "dominant_known": dominant_known,
        "expected_known_fraction": round(expected_known_fraction, 6),
        "passed": passed,
        "labeled_track_intervals": _intervals(predictions, expected),
    }


def _run_scenario(
    *,
    name: str,
    video_path: Path,
    end_frame: int,
    expected: str,
    rtmw_adapter: RtmwPoseAdapter,
    gait_config: GaitConfig,
    detector_model_path: Path,
    tracker_config_path: Path,
    output_dir: Path,
    stride: int,
) -> dict[str, object]:
    yolo = YOLO(str(detector_model_path))
    flow_backend = OpenCvFarnebackBackend(backward_check=False)
    camera_config = CameraMotionConfig(mode="fixed")
    articulated_config = ArticulatedFlowConfig()
    region_config = PoseRegionConfig()
    gait_pipeline = OpticalGaitPipeline(gait_config)
    motion_pipeline = MotionPipeline(MotionConfig())

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open KTH video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 25.0

    previous_samples: dict[int, tuple[np.ndarray, object]] = {}
    predictions: list[dict[str, object]] = []
    processed_frames = 0
    detected_frames = 0
    pose_frames = 0
    evidence_frames = 0
    frame_number = 0

    jsonl_path = output_dir / f"{name}.jsonl"
    articulated_path = output_dir / f"{name}.articulated.jsonl"
    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl,
        articulated_path.open("w", encoding="utf-8") as articulated_jsonl,
    ):
        try:
            while True:
                ok, raw_frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number > end_frame:
                    break
                if (frame_number - 1) % stride != 0:
                    continue

                processed_frames += 1
                frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                timestamp = (frame_number - 1) / fps

                tracked = yolo.track(
                    frame,
                    persist=True,
                    tracker=str(tracker_config_path),
                    classes=[0],
                    conf=0.10,
                    imgsz=640,
                    verbose=False,
                )[0]
                observation = _largest_person(
                    observations_from_result(tracked, timestamp, frame.shape[:2])
                )
                if observation is None:
                    # A missing detector result is not proof that BoT-SORT has ended
                    # the track. Keep temporal histories until a later tracker result
                    # identifies the active ID, but never compute optical flow across
                    # the missing frame.
                    previous_samples.clear()
                    continue

                detected_frames += 1
                active_ids = {observation.track_id}
                gait_pipeline.drop_stale(active_ids)
                motion_pipeline.drop_stale(active_ids)
                for stale_id in set(previous_samples) - active_ids:
                    del previous_samples[stale_id]

                motion_result = motion_pipeline.update(observation)
                current_pose = rtmw_adapter.infer_track(frame, observation, timestamp)
                if current_pose is None:
                    continue
                pose_frames += 1

                previous = previous_samples.get(observation.track_id)
                if previous is not None:
                    previous_frame, previous_pose = previous
                    flow = flow_backend.compute(
                        previous_frame,
                        frame,
                        previous_pose.timestamp,
                        current_pose.timestamp,
                    )
                    camera = estimate_camera_motion(
                        previous_frame,
                        frame,
                        [],
                        camera_config,
                    )
                    compensated = compensate_flow(flow, camera)
                    regions = build_body_regions(
                        previous_pose,
                        observation.height * current_pose.frame_height,
                        region_config,
                    )
                    articulated = estimate_articulated_flow(
                        observation,
                        previous_pose,
                        current_pose,
                        regions,
                        compensated,
                        articulated_config,
                    )
                    if articulated is not None:
                        evidence_frames += 1
                        articulated_jsonl.write(
                            json.dumps(
                                {
                                    "scenario": name,
                                    "frame": frame_number,
                                    "timestamp": timestamp,
                                    "track_id": articulated.track_id,
                                    "start_timestamp": articulated.start_timestamp,
                                    "end_timestamp": articulated.end_timestamp,
                                    "quality": articulated.quality,
                                    "pose_flow_agreement":
                                        articulated.pose_flow_agreement,
                                    "person_height_px":
                                        articulated.person_height_px,
                                    "normalized_torso_dx":
                                        articulated.normalized_torso_dx,
                                    "normalized_torso_dy":
                                        articulated.normalized_torso_dy,
                                    "region_flow": {
                                        region_name: {
                                            "dx": region.dx,
                                            "dy": region.dy,
                                            "energy": region.energy,
                                            "valid_fraction":
                                                region.valid_fraction,
                                        }
                                        for region_name, region
                                        in articulated.region_flow.items()
                                    },
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        gait_result = gait_pipeline.update(articulated)
                        if gait_result is not None:
                            payload = {
                                "scenario": name,
                                "expected": expected,
                                "frame": frame_number,
                                "timestamp": timestamp,
                                "track_id": observation.track_id,
                                "locomotion": gait_result.state.value,
                                "gait": asdict(gait_result.evidence),
                                "lateral": (
                                    motion_result.state.lateral.value
                                    if motion_result is not None
                                    else None
                                ),
                                "radial": (
                                    motion_result.state.radial.value
                                    if motion_result is not None
                                    else None
                                ),
                            }
                            predictions.append(payload)
                            jsonl.write(json.dumps(payload, sort_keys=True) + "\n")

                previous_samples[observation.track_id] = (frame.copy(), current_pose)
        finally:
            capture.release()

    summary = _summarize(predictions, expected)
    summary.update(
        {
            "video": video_path.name,
            "source_url": f"{KTH_BASE}/{video_path.name}",
            "annotated_frame_range": [1, end_frame],
            "fps": fps,
            "stride": stride,
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "pose_frames": pose_frames,
            "articulated_evidence_frames": evidence_frames,
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/real-video-gait"))
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    gait_config_path = root / "configs/gait.yaml"
    gait_config = _load_gait_config(gait_config_path)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    resolved_yolo_weights = _resolve_yolo_weights("yolo11n.pt")
    detector_model_path = models_dir / "yolo11n.pt"
    if resolved_yolo_weights != detector_model_path.resolve():
        shutil.copy2(resolved_yolo_weights, detector_model_path)

    tracker_config_path = _resolve_botsort_config()

    rtmw_archive_path = models_dir / Path(RTMW_MODEL).name
    _download(RTMW_MODEL, rtmw_archive_path)
    rtmw_onnx_path = _extract_rtmw_onnx(
        rtmw_archive_path,
        models_dir / "rtmw",
    )
    pose_model = RTMPose(
        onnx_model=str(rtmw_onnx_path),
        model_input_size=(192, 256),
        backend="onnxruntime",
        device="cpu",
    )
    rtmw_adapter = RtmwPoseAdapter(RtmLibCropInferencer(pose_model))

    provenance = {
        "git_sha": os.environ.get("GITHUB_SHA"),
        "validation_script": _file_identity(Path(__file__).resolve()),
        "detector": {
            "implementation": "Ultralytics YOLO11n",
            "model_name": "yolo11n.pt",
            "weights": _file_identity(detector_model_path),
        },
        "tracker": {
            "implementation": "Ultralytics BoT-SORT",
            "config": _file_identity(tracker_config_path),
        },
        "rtmw": {
            "runtime": "rtmlib RTMPose / ONNX Runtime CPU",
            "archive_url": RTMW_MODEL,
            "archive": _file_identity(rtmw_archive_path),
            "onnx_model": _file_identity(rtmw_onnx_path),
            "model_input_size": [192, 256],
            "backend": "onnxruntime",
            "device": "cpu",
        },
        "flow_backend": "opencv-farneback forward",
        "camera_mode": "fixed",
        "gait_config": asdict(gait_config),
        "gait_config_file": _file_identity(gait_config_path),
        "articulated_config": asdict(ArticulatedFlowConfig()),
        "pose_region_config": asdict(PoseRegionConfig()),
        "processing_resolution": [640, 480],
        "frame_stride": args.stride,
        "dataset": "KTH human actions sample sequences",
        "dataset_home": KTH_BASE,
        "dataset_note": "KTH page states public availability for non-commercial use",
        "source_videos": {},
        "versions": {
            "ultralytics": metadata.version("ultralytics"),
            "rtmlib": metadata.version("rtmlib"),
            "onnxruntime": metadata.version("onnxruntime"),
            "opencv-python": metadata.version("opencv-python"),
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results: dict[str, object] = {}
    failures: list[str] = []
    for name, scenario in SCENARIOS.items():
        filename = str(scenario["filename"])
        video_path = videos_dir / filename
        source_url = f"{KTH_BASE}/{filename}"
        _download(source_url, video_path)
        provenance["source_videos"][name] = {
            "url": source_url,
            **_file_identity(video_path),
        }
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = _run_scenario(
            name=name,
            video_path=video_path,
            end_frame=int(scenario["end_frame"]),
            expected=str(scenario["expected"]),
            rtmw_adapter=rtmw_adapter,
            gait_config=gait_config,
            detector_model_path=detector_model_path,
            tracker_config_path=tracker_config_path,
            output_dir=output_dir,
            stride=args.stride,
        )
        results[name] = result
        if not bool(result["passed"]):
            failures.append(
                f"{name}: expected={result['expected']} dominant={result['dominant_known']} "
                f"known_fraction={result['expected_known_fraction']} "
                f"counts={result['state_counts']}"
            )

    report = {
        "provenance": provenance,
        "scenarios": results,
        "passed": not failures,
        "failures": failures,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
