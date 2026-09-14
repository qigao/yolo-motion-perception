from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any


class CoverageError(ValueError):
    pass


@dataclass(frozen=True)
class GroundTruthBox:
    frame: int
    object_id: int
    xc: float
    yc: float
    width: float
    height: float


@dataclass(frozen=True)
class ObservationRecord:
    frame_index: int
    track_id: int
    timestamp: float
    class_id: int
    confidence: float
    cx: float
    cy: float
    width: float
    height: float


def load_caviar_ground_truth(
    path: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    object_ids: set[int] | None = None,
) -> list[GroundTruthBox]:
    if start_frame < 0 or end_frame < start_frame:
        raise CoverageError("requires 0 <= start_frame <= end_frame")

    root = ET.parse(path).getroot()
    boxes: list[GroundTruthBox] = []
    for frame_node in root.findall("frame"):
        frame = int(frame_node.attrib["number"])
        if frame < start_frame or frame > end_frame:
            continue
        for object_node in frame_node.findall("./objectlist/object"):
            object_id = int(object_node.attrib["id"])
            if object_ids is not None and object_id not in object_ids:
                continue
            box = object_node.find("box")
            if box is None:
                continue
            boxes.append(
                GroundTruthBox(
                    frame=frame,
                    object_id=object_id,
                    xc=float(box.attrib["xc"]),
                    yc=float(box.attrib["yc"]),
                    width=float(box.attrib["w"]),
                    height=float(box.attrib["h"]),
                )
            )
    return boxes


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoverageError(f"{context} must be a mapping")
    return value


def load_observations(path: str | Path) -> list[ObservationRecord]:
    observations: list[ObservationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _require_mapping(
                    json.loads(line), context=f"observation line {line_number}"
                )
            except json.JSONDecodeError as exc:
                raise CoverageError(f"invalid JSON at line {line_number}") from exc
            try:
                observations.append(
                    ObservationRecord(
                        frame_index=int(row["frame_index"]),
                        track_id=int(row["track_id"]),
                        timestamp=float(row["timestamp"]),
                        class_id=int(row["class_id"]),
                        confidence=float(row["detection_confidence"]),
                        cx=float(row["cx"]),
                        cy=float(row["cy"]),
                        width=float(row["width"]),
                        height=float(row["height"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CoverageError(
                    f"observation line {line_number} requires frame_index/track_id/"
                    "timestamp/class_id/detection_confidence/cx/cy/width/height"
                ) from exc
    return observations


def _corners(
    xc: float, yc: float, width: float, height: float
) -> tuple[float, float, float, float]:
    return (
        xc - width / 2.0,
        yc - height / 2.0,
        xc + width / 2.0,
        yc + height / 2.0,
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_caviar_coverage(
    ground_truth: list[GroundTruthBox],
    observations: list[ObservationRecord],
    *,
    source_width: int,
    source_height: int,
    start_frame: int,
    iou_threshold: float = 0.3,
    person_class_id: int = 0,
) -> dict[str, Any]:
    if source_width <= 0 or source_height <= 0:
        raise CoverageError("source dimensions must be positive")
    if not (0.0 <= iou_threshold <= 1.0):
        raise CoverageError("iou_threshold must be in [0, 1]")

    gt_by_frame: dict[int, list[GroundTruthBox]] = {}
    for box in ground_truth:
        gt_by_frame.setdefault(box.frame, []).append(box)

    obs_by_frame: dict[int, list[ObservationRecord]] = {}
    for observation in observations:
        source_frame = start_frame + observation.frame_index
        obs_by_frame.setdefault(source_frame, []).append(observation)

    frames_with_person_detection = sum(
        1
        for frame_observations in obs_by_frame.values()
        if any(obs.class_id == person_class_id for obs in frame_observations)
    )

    matches_by_gt: dict[int, list[tuple[int, int]]] = {}
    matched_object_frames = 0
    for frame in sorted(gt_by_frame):
        gt_boxes = gt_by_frame[frame]
        candidates = [
            observation
            for observation in obs_by_frame.get(frame, [])
            if observation.class_id == person_class_id
        ]
        used_tracks: set[int] = set()
        for gt in gt_boxes:
            gt_rect = _corners(gt.xc, gt.yc, gt.width, gt.height)
            best: ObservationRecord | None = None
            best_iou = -1.0
            for observation in candidates:
                if observation.track_id in used_tracks:
                    continue
                obs_rect = _corners(
                    observation.cx * source_width,
                    observation.cy * source_height,
                    observation.width * source_width,
                    observation.height * source_height,
                )
                score = _iou(gt_rect, obs_rect)
                if score > best_iou:
                    best_iou = score
                    best = observation
            if best is not None and best_iou >= iou_threshold:
                used_tracks.add(best.track_id)
                matched_object_frames += 1
                matches_by_gt.setdefault(gt.object_id, []).append((frame, best.track_id))

    gt_object_frames = len(ground_truth)
    coverage = matched_object_frames / gt_object_frames if gt_object_frames else None

    fragments_by_gt: dict[str, int] = {}
    id_switches_by_gt: dict[str, int] = {}
    matched_track_ids: set[int] = set()
    for object_id in sorted({box.object_id for box in ground_truth}):
        rows = sorted(matches_by_gt.get(object_id, []))
        track_ids = [track_id for _, track_id in rows]
        matched_track_ids.update(track_ids)
        fragments_by_gt[str(object_id)] = len(set(track_ids))
        id_switches_by_gt[str(object_id)] = sum(
            1 for previous, current in pairwise(track_ids) if current != previous
        )

    return {
        "gt_object_frames": gt_object_frames,
        "matched_object_frames": matched_object_frames,
        "coverage": coverage,
        "frames_with_person_detection": frames_with_person_detection,
        "matched_track_ids": sorted(matched_track_ids),
        "fragments_by_gt": fragments_by_gt,
        "id_switches_by_gt": id_switches_by_gt,
    }
