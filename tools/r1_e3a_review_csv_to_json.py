"""Convert completed human R1-E3A review CSVs into canonical review JSON.

This tool is deliberately mechanical. It never infers labels, boundaries, actor
or target identity, multiview identity, or review outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ANNOTATION_COLUMNS = [
    "episode_id",
    "physical_event_id",
    "split",
    "source_window_component_id",
    "registered_video_id",
    "context_video_ids",
    "start_frame_inclusive",
    "end_frame_exclusive",
    "label",
    "actor_description",
    "target_description",
    "target_category_hint",
    "annotation_revision",
    "annotator",
    "review_status",
    "reviewer",
    "notes",
]

COMPLETION_COLUMNS = [
    "source_video_id",
    "full_range_reviewed",
    "reviewed_candidate_count",
    "reviewer",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path}: expected columns {expected_columns}, got {reader.fieldnames}"
            )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]


def _parse_bool(value: str, *, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"{field}: expected true/false, got {value!r}")


def _parse_nonnegative_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field}: expected integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field}: expected non-negative integer")
    return parsed


def _parse_positive_int(value: str, *, field: str) -> int:
    parsed = _parse_nonnegative_int(value, field=field)
    if parsed < 1:
        raise ValueError(f"{field}: expected positive integer")
    return parsed


def _manifest_videos(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for record in manifest.get("videos", []):
        video_id = str(record.get("source_video_id", ""))
        if not video_id:
            raise ValueError("manifest video missing source_video_id")
        if video_id in videos:
            raise ValueError(f"duplicate manifest video: {video_id}")
        videos[video_id] = record
    if not videos:
        raise ValueError("manifest has no videos")
    return videos


def convert_review(
    annotations_path: Path,
    completion_path: Path,
    manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    handbook_sha256: str,
    sampling_protocol_sha256: str,
) -> dict[str, Any]:
    videos = _manifest_videos(manifest)
    completion_rows = _read_csv(completion_path, COMPLETION_COLUMNS)
    annotation_rows = _read_csv(annotations_path, ANNOTATION_COLUMNS)

    video_reviews: dict[str, dict[str, Any]] = {}
    for row in completion_rows:
        video_id = row["source_video_id"]
        if video_id not in videos:
            raise ValueError(f"completion references unknown video: {video_id}")
        if video_id in video_reviews:
            raise ValueError(f"duplicate completion row: {video_id}")
        video_reviews[video_id] = {
            "video_id": video_id,
            "full_range_reviewed": _parse_bool(
                row["full_range_reviewed"],
                field=f"{video_id}.full_range_reviewed",
            ),
            "candidate_count": _parse_nonnegative_int(
                row["reviewed_candidate_count"],
                field=f"{video_id}.reviewed_candidate_count",
            ),
            "reviewer": row["reviewer"],
        }

    missing = sorted(set(videos) - set(video_reviews))
    extra = sorted(set(video_reviews) - set(videos))
    if missing or extra:
        raise ValueError(
            f"completion coverage mismatch: missing={missing}, extra={extra}"
        )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(annotation_rows, start=1):
        context = f"annotation row {index}"
        registered_video_id = row["registered_video_id"]
        if registered_video_id not in videos:
            raise ValueError(
                f"{context}: unknown registered_video_id {registered_video_id}"
            )
        registered = videos[registered_video_id]

        if row["split"] != str(registered.get("split")):
            raise ValueError(
                f"{context}: split mismatch for {registered_video_id}: "
                f"csv={row['split']} manifest={registered.get('split')}"
            )
        if row["source_window_component_id"] != str(
            registered.get("source_window_component_id")
        ):
            raise ValueError(
                f"{context}: source-window component mismatch for "
                f"{registered_video_id}"
            )

        context_video_ids = [
            value.strip()
            for value in row["context_video_ids"].split(";")
            if value.strip()
        ]
        for context_video_id in context_video_ids:
            if context_video_id not in videos:
                raise ValueError(
                    f"{context}: unknown context video {context_video_id}"
                )
            context_video = videos[context_video_id]
            if context_video.get("split") != registered.get("split"):
                raise ValueError(f"{context}: context video crosses split")
            if (
                context_video.get("source_window_component_id")
                != registered.get("source_window_component_id")
            ):
                raise ValueError(
                    f"{context}: context video crosses source-window component"
                )

        record = {
            "episode_id": row["episode_id"],
            "physical_event_id": row["physical_event_id"],
            "split": row["split"],
            "source_window_component_id": row["source_window_component_id"],
            "registered_video_id": registered_video_id,
            "context_video_ids": context_video_ids,
            "start_frame_inclusive": _parse_nonnegative_int(
                row["start_frame_inclusive"],
                field=f"{context}.start_frame_inclusive",
            ),
            "end_frame_exclusive": _parse_positive_int(
                row["end_frame_exclusive"],
                field=f"{context}.end_frame_exclusive",
            ),
            "label": row["label"],
            "actor_description": row["actor_description"],
            "target_description": row["target_description"],
            "target_category_hint": row["target_category_hint"],
            "annotation_revision": _parse_positive_int(
                row["annotation_revision"],
                field=f"{context}.annotation_revision",
            ),
            "annotator": row["annotator"],
            "review_status": row["review_status"],
            "reviewer": row["reviewer"],
            "notes": row["notes"],
        }
        records.append(record)

    ordered_reviews = [
        video_reviews[video_id]
        for video_id in sorted(
            videos,
            key=lambda video_id: (
                0 if str(videos[video_id].get("split")) == "train" else 1,
                video_id,
            ),
        )
    ]

    return {
        "schema": "r1-e3a-semantic-annotation-review-v1",
        "status": "reviewed",
        "science_head_sha": manifest["science_head_sha"],
        "source_manifest_sha256": source_manifest_sha256,
        "handbook_sha256": handbook_sha256,
        "sampling_protocol_sha256": sampling_protocol_sha256,
        "video_reviews": ordered_reviews,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--completion-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, required=True)
    parser.add_argument("--sampling-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review = convert_review(
        args.annotations_csv,
        args.completion_csv,
        manifest,
        source_manifest_sha256=sha256_file(args.manifest),
        handbook_sha256=sha256_file(args.handbook),
        sampling_protocol_sha256=sha256_file(args.sampling_protocol),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"canonical review JSON -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
