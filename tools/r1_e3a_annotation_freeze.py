"""Validate and summarize a completed R1-E3A semantic annotation review.

This is a fail-closed acquisition/annotation gate. It does not infer labels and
does not run YOLO, BoT-SORT, reservoir/readout code, prospective sealing, or any
registered R1-E3 measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "r1-e3a-semantic-annotation-review-v1"
LABELS = ("approach", "touch", "pick_up", "pass_by")
REJECTION_REASONS = {
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
}
TRAIN_MIN_PER_CLASS = 20
EVAL_MIN_PER_CLASS = 10
MIN_SOURCE_VIDEOS_PER_SPLIT = 2
MIN_SOURCE_WINDOW_COMPONENTS_PER_SPLIT = 2



def registration_minima_are_met(
    accepted_counts: dict[str, dict[str, int]],
    *,
    accepted_source_video_counts: dict[str, int],
    accepted_component_counts: dict[str, int],
) -> bool:
    class_minima_met = all(
        accepted_counts["train"][label] >= TRAIN_MIN_PER_CLASS
        and accepted_counts["eval"][label] >= EVAL_MIN_PER_CLASS
        for label in LABELS
    )
    source_video_minima_met = all(
        accepted_source_video_counts[split] >= MIN_SOURCE_VIDEOS_PER_SPLIT
        for split in ("train", "eval")
    )
    component_minima_met = all(
        accepted_component_counts[split]
        >= MIN_SOURCE_WINDOW_COMPONENTS_PER_SPLIT
        for split in ("train", "eval")
    )
    return class_minima_met and source_video_minima_met and component_minima_met


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(row: dict[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing {key}")
    if value.strip().upper() == "UNASSIGNED":
        raise ValueError(f"{context}: {key} is UNASSIGNED")
    return value.strip()


def _manifest_videos(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for row in manifest.get("videos", []):
        video_id = str(row.get("source_video_id", ""))
        if not video_id:
            raise ValueError("manifest video missing source_video_id")
        if video_id in videos:
            raise ValueError(f"duplicate manifest source_video_id: {video_id}")
        split = row.get("split")
        if split not in {"train", "eval"}:
            raise ValueError(f"manifest video {video_id}: invalid split {split}")
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"manifest video {video_id}: invalid frame_count")
        component = row.get("source_window_component_id")
        if not isinstance(component, str) or not component:
            raise ValueError(
                f"manifest video {video_id}: missing source_window_component_id"
            )
        videos[video_id] = row
    if not videos:
        raise ValueError("manifest has no videos")
    return videos


def validate_review(
    review: dict[str, Any],
    manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    handbook_sha256: str,
    sampling_protocol_sha256: str,
) -> dict[str, Any]:
    if review.get("schema") != SCHEMA:
        raise ValueError(f"expected review schema {SCHEMA}")
    if review.get("status") != "reviewed":
        raise ValueError("review status must be reviewed before freeze")

    bindings = {
        "source_manifest_sha256": source_manifest_sha256,
        "handbook_sha256": handbook_sha256,
        "sampling_protocol_sha256": sampling_protocol_sha256,
    }
    for field, expected in bindings.items():
        if review.get(field) != expected:
            raise ValueError(
                f"{field} mismatch: expected {expected}, got {review.get(field)}"
            )

    science_head = manifest.get("science_head_sha")
    if review.get("science_head_sha") != science_head:
        raise ValueError("science_head_sha mismatch")

    videos = _manifest_videos(manifest)

    review_rows: dict[str, dict[str, Any]] = {}
    for row in review.get("video_reviews", []):
        video_id = str(row.get("video_id", ""))
        if video_id not in videos:
            raise ValueError(f"video review references unknown video: {video_id}")
        if video_id in review_rows:
            raise ValueError(f"duplicate video review: {video_id}")
        if row.get("full_range_reviewed") is not True:
            raise ValueError(f"full-range review incomplete: {video_id}")
        candidate_count = row.get("candidate_count")
        if not isinstance(candidate_count, int) or candidate_count < 0:
            raise ValueError(f"invalid candidate_count: {video_id}")
        _required_text(row, "reviewer", context=f"video review {video_id}")
        review_rows[video_id] = row

    missing_reviews = sorted(set(videos) - set(review_rows))
    extra_reviews = sorted(set(review_rows) - set(videos))
    if missing_reviews or extra_reviews:
        raise ValueError(
            f"video review coverage mismatch: missing={missing_reviews}, "
            f"extra={extra_reviews}"
        )

    episode_ids: set[str] = set()
    event_ids: set[str] = set()
    appearances: Counter[str] = Counter()
    accepted: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    accepted_components: dict[str, set[str]] = {
        "train": set(),
        "eval": set(),
    }
    accepted_videos: dict[str, set[str]] = {
        "train": set(),
        "eval": set(),
    }
    rejected = Counter()

    for index, row in enumerate(review.get("records", []), start=1):
        context = f"record {index}"
        episode_id = _required_text(row, "episode_id", context=context)
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)

        event_id = _required_text(
            row,
            "physical_event_id",
            context=context,
        )
        if event_id in event_ids:
            raise ValueError(f"duplicate physical_event_id: {event_id}")
        event_ids.add(event_id)

        registered_video_id = _required_text(
            row,
            "registered_video_id",
            context=context,
        )
        if registered_video_id not in videos:
            raise ValueError(
                f"{context}: unknown registered_video_id {registered_video_id}"
            )
        registered_video = videos[registered_video_id]

        if row.get("split") != registered_video["split"]:
            raise ValueError(
                f"{context}: split mismatch for {registered_video_id}"
            )
        if (
            row.get("source_window_component_id")
            != registered_video["source_window_component_id"]
        ):
            raise ValueError(
                f"{context}: source-window component mismatch for "
                f"{registered_video_id}"
            )

        context_video_ids = row.get("context_video_ids", [])
        if not isinstance(context_video_ids, list):
            raise TypeError(f"{context}: context_video_ids must be a list")
        if len(context_video_ids) != len(set(context_video_ids)):
            raise ValueError(f"{context}: duplicate context_video_ids")
        if registered_video_id in context_video_ids:
            raise ValueError(
                f"{context}: registered video also appears as context"
            )

        all_video_ids = [registered_video_id]
        for value in context_video_ids:
            context_video_id = str(value)
            if context_video_id not in videos:
                raise ValueError(
                    f"{context}: unknown context_video_id {context_video_id}"
                )
            context_video = videos[context_video_id]
            if context_video["split"] != registered_video["split"]:
                raise ValueError(
                    f"{context}: multiview context crosses split"
                )
            if (
                context_video["source_window_component_id"]
                != registered_video["source_window_component_id"]
            ):
                raise ValueError(
                    f"{context}: multiview context crosses source-window component"
                )
            all_video_ids.append(context_video_id)

        for video_id in set(all_video_ids):
            appearances[video_id] += 1

        start = row.get("start_frame_inclusive")
        end = row.get("end_frame_exclusive")
        frame_count = registered_video["frame_count"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > frame_count
        ):
            raise ValueError(
                f"{context}: frame bounds [{start}, {end}) outside "
                f"[0, {frame_count})"
            )

        status = _required_text(row, "review_status", context=context)
        if status == "needs_resolution":
            raise ValueError(f"{context}: needs_resolution is not freezeable")

        label = row.get("label")
        if status == "accepted":
            if label not in LABELS:
                raise ValueError(f"{context}: invalid accepted label {label}")
            _required_text(row, "actor_description", context=context)
            _required_text(row, "target_description", context=context)
            _required_text(row, "annotator", context=context)
            _required_text(row, "reviewer", context=context)
            revision = row.get("annotation_revision")
            if not isinstance(revision, int) or revision < 1:
                raise ValueError(f"{context}: invalid annotation_revision")

            split = registered_video["split"]
            accepted[split][label] += 1
            accepted_videos[split].add(registered_video_id)
            accepted_components[split].add(
                registered_video["source_window_component_id"]
            )
        elif status.startswith("rejected:"):
            reason = status.split(":", 1)[1]
            if reason not in REJECTION_REASONS:
                raise ValueError(
                    f"{context}: invalid rejection reason {reason}"
                )
            _required_text(row, "reviewer", context=context)
            rejected[reason] += 1
        else:
            raise ValueError(f"{context}: invalid review_status {status}")

    for video_id, row in review_rows.items():
        expected = row["candidate_count"]
        actual = appearances[video_id]
        if expected != actual:
            raise ValueError(
                f"candidate_count mismatch for {video_id}: "
                f"expected {expected}, reconciled {actual}"
            )

    accepted_summary = {
        split: {label: int(accepted[split][label]) for label in LABELS}
        for split in ("train", "eval")
    }
    accepted_source_video_counts = {
        split: len(accepted_videos[split])
        for split in ("train", "eval")
    }
    accepted_component_counts = {
        split: len(accepted_components[split])
        for split in ("train", "eval")
    }
    minima_met = registration_minima_are_met(
        accepted_summary,
        accepted_source_video_counts=accepted_source_video_counts,
        accepted_component_counts=accepted_component_counts,
    )

    return {
        "schema": "r1-e3a-semantic-annotation-freeze-summary-v1",
        "review_schema": SCHEMA,
        "science_head_sha": science_head,
        "video_count": len(videos),
        "episode_count": len(episode_ids),
        "physical_event_count": len(event_ids),
        "reviewed_candidate_appearances": {
            video_id: int(appearances[video_id])
            for video_id in sorted(videos)
        },
        "accepted": accepted_summary,
        "rejected_by_reason": {
            reason: int(rejected[reason])
            for reason in sorted(rejected)
        },
        "distinct_accepted_source_videos": accepted_source_video_counts,
        "distinct_accepted_source_window_components": accepted_component_counts,
        "registration_minima": {
            "train_per_class": TRAIN_MIN_PER_CLASS,
            "eval_per_class": EVAL_MIN_PER_CLASS,
            "source_videos_per_split": MIN_SOURCE_VIDEOS_PER_SPLIT,
            "source_window_components_per_split":
                MIN_SOURCE_WINDOW_COMPONENTS_PER_SPLIT,
        },
        "registration_minima_met": minima_met,
        "next_batch_required": not minima_met,
        "zero_unresolved_candidates": True,
        "zero_duplicate_physical_event_ids": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, required=True)
    parser.add_argument("--sampling-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-registration-minima", action="store_true")
    args = parser.parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_manifest_sha256 = sha256_file(args.manifest)
    handbook_sha256 = sha256_file(args.handbook)
    sampling_protocol_sha256 = sha256_file(args.sampling_protocol)
    summary = validate_review(
        review,
        manifest,
        source_manifest_sha256=source_manifest_sha256,
        handbook_sha256=handbook_sha256,
        sampling_protocol_sha256=sampling_protocol_sha256,
    )

    summary["review_sha256"] = sha256_file(args.review)
    summary["source_manifest_sha256"] = source_manifest_sha256
    summary["handbook_sha256"] = handbook_sha256
    summary["sampling_protocol_sha256"] = sampling_protocol_sha256

    if args.require_registration_minima and not summary["registration_minima_met"]:
        raise SystemExit("registered semantic count minima are not met")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
