from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractionRow:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_json_bytes(payload))


def _current_head() -> str:
    exact_head = os.environ.get("R1_E3M_EXACT_HEAD")
    if exact_head:
        return exact_head
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_lock(path: Path) -> str:
    packages = sorted(
        {
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in metadata.distributions()
            if dist.metadata.get("Name")
        },
        key=str.casefold,
    )
    path.write_text("\n".join(packages) + "\n", encoding="utf-8")
    return _sha256_file(path)


def _resolve_weights(model_name: str, output_path: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_name)
    candidates = (
        Path(model_name),
        Path(str(getattr(model, "ckpt_path", ""))),
        Path(str(getattr(model.model, "pt_path", ""))),
    )
    source = next(
        (
            candidate.resolve()
            for candidate in candidates
            if str(candidate) and candidate.is_file()
        ),
        None,
    )
    if source is None:
        raise RuntimeError(
            f"cannot resolve downloaded YOLO weights for {model_name}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source != output_path.resolve():
        shutil.copy2(source, output_path)
    return output_path.resolve()


def _verify_source_files(
    video_root: Path,
    manifest_payload: dict[str, Any],
) -> None:
    for item in manifest_payload["videos"]:
        video_id = str(item["source_video_id"])
        path = video_root / f"{video_id}.avi"
        if not path.is_file():
            raise RuntimeError(f"missing frozen source video: {video_id}")
        if path.stat().st_size != int(item["byte_size"]):
            raise RuntimeError(
                f"source byte-size mismatch: {video_id}"
            )
        if _sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(
                f"source sha256 mismatch: {video_id}"
            )


def _round_row(
    *,
    video_id: str,
    frame_index: int,
    fps: float,
    track_id: int,
    confidence: float,
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> ExtractionRow | None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) * 0.5 / float(width)
    cy = (y1 + y2) * 0.5 / float(height)
    w = (x2 - x1) / float(width)
    h = (y2 - y1) / float(height)

    return ExtractionRow(
        video_id=video_id,
        frame_index=frame_index,
        timestamp_seconds=round(frame_index / fps, 9),
        track_id=track_id,
        class_id=0,
        confidence=round(float(confidence), 8),
        x1=round(x1, 6),
        y1=round(y1, 6),
        x2=round(x2, 6),
        y2=round(y2, 6),
        cx_norm=round(cx, 9),
        cy_norm=round(cy, 9),
        w_norm=round(w, 9),
        h_norm=round(h, 9),
    )


def _extract_video(
    *,
    video_path: Path,
    video_id: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    weights_path: Path,
    tracker_config: Path,
) -> list[ExtractionRow]:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    from neural_state_machine.r1_e3m_extraction import (
        CLASS_FILTER,
        CONFIDENCE_THRESHOLD,
        DEVICE,
        HALF,
        IMGSZ,
        IOU_THRESHOLD,
        NATIVE_FRAME_STRIDE,
    )

    cv2.setNumThreads(1)
    np.random.seed(0)
    torch.manual_seed(0)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_id}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    native_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    native_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if abs(native_fps - fps) > 1e-6:
        raise RuntimeError(f"source fps mismatch: {video_id}")
    if native_width != width or native_height != height:
        raise RuntimeError(f"source geometry mismatch: {video_id}")

    model = YOLO(str(weights_path))
    rows: list[ExtractionRow] = []
    decoded_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded_frames
            decoded_frames += 1
            if frame_index % NATIVE_FRAME_STRIDE != 0:
                continue

            result = model.track(
                frame,
                persist=True,
                tracker=str(tracker_config),
                classes=list(CLASS_FILTER),
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMGSZ,
                device=DEVICE,
                half=HALF,
                verbose=False,
            )[0]
            boxes = result.boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue

            ids = boxes.id.detach().cpu().numpy().astype(int)
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            coordinates = boxes.xyxy.detach().cpu().numpy()

            frame_rows: list[ExtractionRow] = []
            for index, track_id in enumerate(ids):
                if int(classes[index]) != 0:
                    continue
                row = _round_row(
                    video_id=video_id,
                    frame_index=frame_index,
                    fps=fps,
                    track_id=int(track_id),
                    confidence=float(confidences[index]),
                    xyxy=tuple(
                        float(value)
                        for value in coordinates[index].tolist()
                    ),
                    width=width,
                    height=height,
                )
                if row is not None:
                    frame_rows.append(row)
            rows.extend(
                sorted(frame_rows, key=lambda row: row.track_id)
            )
    finally:
        capture.release()

    if decoded_frames != frame_count:
        raise RuntimeError(
            f"decoded frame-count mismatch for {video_id}: "
            f"{decoded_frames} != {frame_count}"
        )
    return rows


def _write_tracks(path: Path, rows: list[ExtractionRow]) -> str:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json_bytes(asdict(row)))
    return _sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sealed label-free R1-E3M YOLO11 + BoT-SORT extraction."
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weights-source",
        default="yolo11n.pt",
    )
    args = parser.parse_args()

    from neural_state_machine.r1_e3m_artifact import MechanismArtifact
    from neural_state_machine.r1_e3m_artifact_evidence import (
        write_mechanism_candidate,
    )
    from neural_state_machine.r1_e3m_extraction import (
        DETECTOR_MODEL,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        TRACKER_CONFIG,
        ULTRALYTICS_VERSION,
        extraction_settings_payload,
        mechanism_videos_from_source_manifest,
        verify_window_video_coverage,
    )
    from neural_state_machine.r1_e3m_window import build_track_windows

    actual_ultralytics = metadata.version("ultralytics")
    if actual_ultralytics != ULTRALYTICS_VERSION:
        raise RuntimeError(
            "ultralytics version mismatch: "
            f"{actual_ultralytics} != {ULTRALYTICS_VERSION}"
        )

    source_manifest_sha = _sha256_file(args.source_manifest)
    if source_manifest_sha != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "source manifest SHA-256 mismatch: "
            f"{source_manifest_sha}"
        )
    source_payload = json.loads(
        args.source_manifest.read_text(encoding="utf-8")
    )
    videos = mechanism_videos_from_source_manifest(source_payload)
    _verify_source_files(args.video_root, source_payload)

    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    raw_dir = output / "raw"
    models_dir = output / "models"
    candidate_dir = output / "candidate"
    raw_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    weights_path = _resolve_weights(
        args.weights_source,
        models_dir / DETECTOR_MODEL,
    )
    tracker_config = Path(TRACKER_CONFIG).resolve()
    if not tracker_config.is_file():
        raise RuntimeError(
            f"missing frozen tracker config: {tracker_config}"
        )

    all_rows: list[ExtractionRow] = []
    row_counts: Counter[str] = Counter()
    for video in videos:
        rows = _extract_video(
            video_path=args.video_root / f"{video.video_id}.avi",
            video_id=video.video_id,
            fps=video.fps,
            frame_count=video.frame_count,
            width=video.width,
            height=video.height,
            weights_path=weights_path,
            tracker_config=tracker_config,
        )
        all_rows.extend(rows)
        row_counts[video.video_id] += len(rows)

    all_rows.sort(
        key=lambda row: (
            row.video_id,
            row.frame_index,
            row.track_id,
        )
    )
    if not all_rows:
        raise RuntimeError("sealed extraction produced zero track rows")

    tracks_path = raw_dir / "tracks.jsonl"
    raw_tracks_sha = _write_tracks(tracks_path, all_rows)

    windows = build_track_windows(all_rows, videos)
    window_counts = Counter(window.video_id for window in windows)
    verify_window_video_coverage(videos, window_counts)

    lock_path = raw_dir / "package-lock.txt"
    package_lock_sha = _package_lock(lock_path)

    import cv2
    import torch
    import ultralytics
    import yaml

    tracker_payload = yaml.safe_load(
        tracker_config.read_text(encoding="utf-8")
    )
    provenance = {
        "schema": "r1-e3m-extraction-provenance-v1",
        "extractor_commit": _current_head(),
        "source_manifest_sha256": source_manifest_sha,
        "raw_tracks_sha256": raw_tracks_sha,
        "settings": extraction_settings_payload(),
        "yolo": {
            "implementation": "Ultralytics YOLO11",
            "package_version": ultralytics.__version__,
            "model": DETECTOR_MODEL,
            "weights_sha256": _sha256_file(weights_path),
        },
        "botsort": {
            "implementation": "Ultralytics BoT-SORT",
            "package_version": ultralytics.__version__,
            "config_sha256": _sha256_file(tracker_config),
            "config": tracker_payload,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": metadata.version("numpy"),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "package_lock_sha256": package_lock_sha,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "canonicalization": {
            "timestamp_decimals": 9,
            "absolute_bbox_decimals": 6,
            "normalized_bbox_decimals": 9,
            "confidence_decimals": 8,
        },
        "source_video_count": len(videos),
        "track_row_count": len(all_rows),
        "eligible_window_count": len(windows),
        "track_rows_by_video": dict(sorted(row_counts.items())),
        "eligible_windows_by_video": dict(sorted(window_counts.items())),
        "source_videos": [
            {
                "video_id": video.video_id,
                "sha256": video.sha256,
                "split": video.split,
                "fps": video.fps,
                "frame_count": video.frame_count,
                "width": video.width,
                "height": video.height,
                "source_window_component_id":
                    video.source_window_component_id,
            }
            for video in videos
        ],
    }
    provenance_path = raw_dir / "extraction-provenance.json"
    _write_json(provenance_path, provenance)
    provenance_sha = _sha256_file(provenance_path)

    artifact = MechanismArtifact(
        source_manifest_sha256=source_manifest_sha,
        raw_track_sha256=raw_tracks_sha,
        extraction_provenance_sha256=provenance_sha,
        videos=videos,
        windows=windows,
    )
    write_mechanism_candidate(artifact, candidate_dir)

    summary = {
        "schema": "r1-e3m-extraction-summary-v1",
        "source_manifest_sha256": source_manifest_sha,
        "raw_tracks_sha256": raw_tracks_sha,
        "extraction_provenance_sha256": provenance_sha,
        "weights_sha256": _sha256_file(weights_path),
        "tracker_config_sha256": _sha256_file(tracker_config),
        "package_lock_sha256": package_lock_sha,
        "source_video_count": len(videos),
        "track_row_count": len(all_rows),
        "eligible_window_count": len(windows),
        "train_window_count": sum(
            1 for window in windows if window.split == "train"
        ),
        "eval_window_count": sum(
            1 for window in windows if window.split == "eval"
        ),
        "eligible_windows_by_video": dict(
            sorted(window_counts.items())
        ),
    }
    _write_json(output / "extraction-summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
