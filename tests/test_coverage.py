import json
from pathlib import Path

from yolo_motion.coverage import (
    evaluate_caviar_coverage,
    load_caviar_ground_truth,
    load_observations,
)


def _write_gt(path: Path, frames: list[tuple[int, int, int, int, int, int]]) -> None:
    body = []
    for frame, object_id, xc, yc, width, height in frames:
        body.append(
            f'<frame number="{frame}"><objectlist><object id="{object_id}">'
            f'<box xc="{xc}" yc="{yc}" w="{width}" h="{height}"/>'
            "</object></objectlist></frame>"
        )
    path.write_text("<dataset>" + "".join(body) + "</dataset>", encoding="utf-8")


def _write_observations(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_coverage_counts_only_person_detections_matching_selected_gt(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.xml"
    obs_path = tmp_path / "observations.jsonl"
    _write_gt(
        gt_path,
        [
            (100, 5, 192, 144, 96, 72),
            (101, 5, 192, 144, 96, 72),
        ],
    )
    _write_observations(
        obs_path,
        [
            {
                "frame_index": 0,
                "track_id": 7,
                "timestamp": 0.0,
                "class_id": 0,
                "detection_confidence": 0.8,
                "cx": 0.5,
                "cy": 0.5,
                "width": 0.25,
                "height": 0.25,
            },
            {
                "frame_index": 1,
                "track_id": 7,
                "timestamp": 0.04,
                "class_id": 14,
                "detection_confidence": 0.9,
                "cx": 0.5,
                "cy": 0.5,
                "width": 0.25,
                "height": 0.25,
            },
        ],
    )

    gt = load_caviar_ground_truth(gt_path, start_frame=100, end_frame=101, object_ids={5})
    observations = load_observations(obs_path)
    report = evaluate_caviar_coverage(
        gt,
        observations,
        source_width=384,
        source_height=288,
        start_frame=100,
        iou_threshold=0.3,
    )

    assert report["gt_object_frames"] == 2
    assert report["matched_object_frames"] == 1
    assert report["coverage"] == 0.5
    assert report["frames_with_person_detection"] == 1
    assert report["matched_track_ids"] == [7]
    assert report["fragments_by_gt"] == {"5": 1}
    assert report["id_switches_by_gt"] == {"5": 0}


def test_coverage_reports_track_fragmentation_and_switches(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.xml"
    obs_path = tmp_path / "observations.jsonl"
    _write_gt(
        gt_path,
        [
            (20, 3, 100, 100, 40, 60),
            (21, 3, 100, 100, 40, 60),
            (22, 3, 100, 100, 40, 60),
        ],
    )
    rows = []
    for frame_index, track_id in enumerate([7, 8, 8]):
        rows.append(
            {
                "frame_index": frame_index,
                "track_id": track_id,
                "timestamp": frame_index / 25,
                "class_id": 0,
                "detection_confidence": 0.9,
                "cx": 100 / 384,
                "cy": 100 / 288,
                "width": 40 / 384,
                "height": 60 / 288,
            }
        )
    _write_observations(obs_path, rows)

    report = evaluate_caviar_coverage(
        load_caviar_ground_truth(gt_path, start_frame=20, end_frame=22, object_ids={3}),
        load_observations(obs_path),
        source_width=384,
        source_height=288,
        start_frame=20,
        iou_threshold=0.3,
    )

    assert report["coverage"] == 1.0
    assert report["matched_track_ids"] == [7, 8]
    assert report["fragments_by_gt"] == {"3": 2}
    assert report["id_switches_by_gt"] == {"3": 1}


def test_load_observations_accepts_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_observations(path) == []
