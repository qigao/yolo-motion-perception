"""Validate and freeze pre-extraction R1-E3A human semantic annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "r1-e3a-source-video-manifest-v1"
FREEZE_SCHEMA = "r1-e3a-semantic-annotation-freeze-v1"
REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")
REJECTION_CODES = (
    "ambiguous_actor_identity",
    "ambiguous_target_identity",
    "ambiguous_contact",
    "ambiguous_pickup_transition",
    "compound_or_overlapping_event",
    "insufficient_pre_event_context",
    "insufficient_post_event_context",
    "camera_cut_or_view_change",
    "target_not_visually_resolvable",
    "preexisting_target_motion_confounds_outcome",
    "other_predeclared_semantic_ambiguity",
)
REQUIRED_COLUMNS = (
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
)
REVIEW_COMPLETION_COLUMNS = (
    "source_video_id",
    "full_range_reviewed",
    "reviewed_candidate_count",
    "reviewer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"expected source manifest schema {SOURCE_SCHEMA}")
    videos = data.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("source manifest videos must be a non-empty list")
    return data


def parse_context_ids(value: str) -> list[str]:
    if not value.strip():
        return []
    ids = [part.strip() for part in value.split(";") if part.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("context_video_ids contains duplicates")
    return ids


def positive_int(value: str, field: str, row_number: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer") from exc
    if result < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return result


def _source_videos(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for record in source_manifest["videos"]:
        if not isinstance(record, dict):
            raise TypeError("source manifest video row must be an object")
        video_id = str(record.get("source_video_id", "")).strip()
        if not video_id:
            raise ValueError("source manifest video id must be non-empty")
        if video_id in videos:
            raise ValueError(f"duplicate source_video_id: {video_id}")
        videos[video_id] = record
    return videos


def validate_annotations(
    source_manifest: dict[str, Any],
    annotation_csv: Path,
) -> dict[str, Any]:
    videos = _source_videos(source_manifest)
    episode_ids: set[str] = set()
    physical_ids: set[str] = set()
    accepted_counts: Counter[tuple[str, str]] = Counter()
    rejection_counts: Counter[str] = Counter()
    reviewed_candidate_counts: Counter[str] = Counter()
    accepted_video_ids: dict[str, set[str]] = {"train": set(), "eval": set()}
    accepted_components: dict[str, set[str]] = {"train": set(), "eval": set()}
    accepted = 0
    rejected = 0

    with annotation_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("annotation CSV columns do not match frozen template")
        for row_number, row in enumerate(reader, start=2):
            episode_id = row["episode_id"].strip()
            physical_id = row["physical_event_id"].strip()
            if not episode_id or not physical_id:
                raise ValueError(
                    f"row {row_number}: episode/physical event id is required"
                )
            if episode_id in episode_ids:
                raise ValueError(f"row {row_number}: duplicate episode_id")
            if physical_id in physical_ids:
                raise ValueError(f"row {row_number}: duplicate physical_event_id")
            episode_ids.add(episode_id)
            physical_ids.add(physical_id)

            video_id = row["registered_video_id"].strip()
            video = videos.get(video_id)
            if video is None:
                raise ValueError(f"row {row_number}: unknown registered_video_id")
            split = row["split"].strip()
            component = row["source_window_component_id"].strip()
            if split not in ("train", "eval"):
                raise ValueError(f"row {row_number}: split must be train or eval")
            if split != str(video["split"]):
                raise ValueError(
                    f"row {row_number}: split disagrees with source manifest"
                )
            if component != str(video["source_window_component_id"]):
                raise ValueError(
                    f"row {row_number}: source-window component disagrees with "
                    "source manifest"
                )

            contexts = parse_context_ids(row["context_video_ids"])
            if video_id in contexts:
                raise ValueError(
                    f"row {row_number}: registered video repeated as context"
                )
            for context_id in contexts:
                context = videos.get(context_id)
                if context is None:
                    raise ValueError(
                        f"row {row_number}: unknown context video {context_id}"
                    )
                if str(context["split"]) != split:
                    raise ValueError(f"row {row_number}: context video crosses split")
                if str(context["source_window_component_id"]) != component:
                    raise ValueError(
                        f"row {row_number}: context video crosses component"
                    )

            # One reviewed physical event contributes once to the audit count of
            # every selected source view in which it was actually reviewed.
            reviewed_candidate_counts[video_id] += 1
            for context_id in contexts:
                reviewed_candidate_counts[context_id] += 1

            start = positive_int(
                row["start_frame_inclusive"],
                "start frame",
                row_number,
            )
            end = positive_int(
                row["end_frame_exclusive"],
                "end frame",
                row_number,
            )
            if end <= start:
                raise ValueError(f"row {row_number}: frame interval must increase")
            if end > int(video["frame_count"]):
                raise ValueError(
                    f"row {row_number}: frame interval exceeds source video"
                )

            review_status = row["review_status"].strip()
            reviewer = row["reviewer"].strip()
            annotator = row["annotator"].strip()
            revision = row["annotation_revision"].strip()
            if not reviewer or not annotator or not revision:
                raise ValueError(
                    f"row {row_number}: review provenance is incomplete"
                )
            if review_status == "needs_resolution":
                raise ValueError(
                    f"row {row_number}: unresolved annotation cannot be frozen"
                )

            label = row["label"].strip()
            if review_status == "accepted":
                if label not in REGISTERED_LABELS:
                    raise ValueError(
                        f"row {row_number}: accepted row has unregistered label"
                    )
                for field in (
                    "actor_description",
                    "target_description",
                    "target_category_hint",
                ):
                    if not row[field].strip():
                        raise ValueError(
                            f"row {row_number}: accepted row missing {field}"
                        )
                accepted += 1
                accepted_counts[(split, label)] += 1
                accepted_video_ids[split].add(video_id)
                accepted_components[split].add(component)
                continue

            prefix = "rejected:"
            if not review_status.startswith(prefix):
                raise ValueError(f"row {row_number}: invalid review_status")
            reason = review_status[len(prefix) :]
            if reason not in REJECTION_CODES:
                raise ValueError(f"row {row_number}: unknown rejection reason")
            if label and label not in REGISTERED_LABELS:
                raise ValueError(f"row {row_number}: rejected row has invalid label")
            rejected += 1
            rejection_counts[reason] += 1

    if not episode_ids:
        raise ValueError("annotation CSV must contain at least one reviewed event")

    counts = {
        split: {
            label: accepted_counts[(split, label)]
            for label in REGISTERED_LABELS
        }
        for split in ("train", "eval")
    }
    accepted_video_counts = {
        split: len(accepted_video_ids[split]) for split in ("train", "eval")
    }
    accepted_component_counts = {
        split: len(accepted_components[split]) for split in ("train", "eval")
    }
    size_gate_pass = all(
        counts["train"][label] >= 20 for label in REGISTERED_LABELS
    )
    size_gate_pass = size_gate_pass and all(
        counts["eval"][label] >= 10 for label in REGISTERED_LABELS
    )
    size_gate_pass = size_gate_pass and all(
        accepted_video_counts[split] >= 2 for split in ("train", "eval")
    )
    size_gate_pass = size_gate_pass and all(
        accepted_component_counts[split] >= 2 for split in ("train", "eval")
    )
    return {
        "accepted_events": accepted,
        "rejected_events": rejected,
        "reviewed_physical_events": len(physical_ids),
        "reviewed_candidate_counts_by_video": {
            video_id: reviewed_candidate_counts[video_id]
            for video_id in sorted(videos)
        },
        "accepted_counts": counts,
        "accepted_video_counts": accepted_video_counts,
        "accepted_component_counts": accepted_component_counts,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "size_gate_pass": size_gate_pass,
        "multiview_physical_event_deduplication_asserted": True,
    }


def _full_range_reviewed(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise ValueError(
            f"row {row_number}: full_range_reviewed must be true or false"
        )
    return normalized == "true"


def validate_review_completion(
    source_manifest: dict[str, Any],
    annotation_summary: dict[str, Any],
    review_completion_csv: Path,
) -> dict[str, Any]:
    videos = _source_videos(source_manifest)
    expected_counts = annotation_summary.get("reviewed_candidate_counts_by_video")
    if not isinstance(expected_counts, dict):
        raise TypeError("annotation summary lacks reviewed candidate counts")

    seen: set[str] = set()
    observed_counts: dict[str, int] = {}
    with review_completion_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COMPLETION_COLUMNS:
            raise ValueError(
                "review completion CSV columns do not match frozen template"
            )
        for row_number, row in enumerate(reader, start=2):
            video_id = row["source_video_id"].strip()
            if video_id not in videos:
                raise ValueError(
                    f"row {row_number}: unknown review-completion source video"
                )
            if video_id in seen:
                raise ValueError(
                    f"row {row_number}: duplicate review-completion source video"
                )
            seen.add(video_id)

            if not _full_range_reviewed(row["full_range_reviewed"], row_number):
                raise ValueError(
                    f"row {row_number}: full-range review is incomplete for "
                    f"{video_id}"
                )
            count = positive_int(
                row["reviewed_candidate_count"],
                "reviewed candidate count",
                row_number,
            )
            expected = expected_counts.get(video_id)
            if type(expected) is not int or count != expected:
                raise ValueError(
                    f"row {row_number}: candidate count mismatch for {video_id}: "
                    f"{count} != {expected}"
                )
            if not row["reviewer"].strip():
                raise ValueError(
                    f"row {row_number}: review-completion reviewer is required"
                )
            observed_counts[video_id] = count

    missing = sorted(set(videos) - seen)
    if missing:
        raise ValueError(
            "full-range review completion marker is missing for: "
            + ", ".join(missing)
        )

    return {
        "complete_source_videos": len(seen),
        "reviewed_candidate_counts_by_video": {
            video_id: observed_counts[video_id] for video_id in sorted(observed_counts)
        },
    }


def build_freeze(
    source_manifest_path: Path,
    annotation_csv: Path,
    review_completion_csv: Path,
    sampling_protocol_path: Path,
) -> dict[str, Any]:
    source_manifest = load_source_manifest(source_manifest_path)
    summary = validate_annotations(source_manifest, annotation_csv)
    review_completion = validate_review_completion(
        source_manifest,
        summary,
        review_completion_csv,
    )
    return {
        "schema": FREEZE_SCHEMA,
        "status": "semantic_annotations_reviewed_pre_extraction",
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "handbook_sha256": source_manifest["handbook_sha256"],
        "sampling_protocol_sha256": sha256_file(sampling_protocol_path),
        "annotation_csv_sha256": sha256_file(annotation_csv),
        "review_completion_csv_sha256": sha256_file(review_completion_csv),
        "science_head_sha": source_manifest["science_head_sha"],
        "constraints": {
            "human_review_required": True,
            "all_activated_source_videos_fully_reviewed": True,
            "detector_tracker_outputs_forbidden_during_semantic_review": True,
            "r1_e3_scores_forbidden_before_freeze": True,
            "frame_bounds_authoritative": True,
        },
        "summary": summary,
        "review_completion": review_completion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--review-completion", type=Path, required=True)
    parser.add_argument("--sampling-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-size-gate", action="store_true")
    args = parser.parse_args()

    freeze = build_freeze(
        args.source_manifest,
        args.annotations,
        args.review_completion,
        args.sampling_protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(freeze["summary"], indent=2, sort_keys=True))
    if args.require_size_gate and not freeze["summary"]["size_gate_pass"]:
        raise SystemExit("registered semantic dataset-size gate is not satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
