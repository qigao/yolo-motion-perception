"""Create an acquisition-only human review pack from frozen R1-E3A source videos.

This tool never runs a detector, tracker, reservoir, or registered measurement.
It validates the frozen source bytes, groups synchronized selected views, and
renders review-only proxies/contact sheets for semantic annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "r1-e3a-source-video-manifest-v1"
REVIEW_SCHEMA = "r1-e3a-human-review-pack-v1"
REGISTERED_LABELS = ("approach", "touch", "pick_up", "pass_by")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"expected source manifest schema {MANIFEST_SCHEMA}")
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("source manifest videos must be a non-empty list")
    return payload


def validate_sources(video_root: Path, manifest: dict[str, Any]) -> None:
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for record in manifest["videos"]:
        video_id = str(record["source_video_id"])
        if video_id in seen_ids:
            raise ValueError(f"duplicate source_video_id: {video_id}")
        seen_ids.add(video_id)
        expected_hash = str(record["sha256"])
        if expected_hash in seen_hashes:
            raise ValueError(f"duplicate source video sha256: {expected_hash}")
        seen_hashes.add(expected_hash)
        path = video_root / f"{video_id}.avi"
        if not path.is_file():
            raise ValueError(f"missing frozen source video: {video_id}")
        if path.stat().st_size != int(record["byte_size"]):
            raise ValueError(f"source byte-size mismatch: {video_id}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"source sha256 mismatch: {video_id}")


def review_groups(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest["videos"]:
        grouped[str(record["anchor_capture_group_id"])].append(record)

    groups: list[dict[str, Any]] = []
    for anchor, records in grouped.items():
        records = sorted(records, key=lambda item: str(item["camera"]))
        splits = {str(item["split"]) for item in records}
        components = {str(item["source_window_component_id"]) for item in records}
        if len(splits) != 1 or len(components) != 1:
            raise ValueError(f"review group crosses split/component: {anchor}")
        groups.append(
            {
                "anchor_capture_group_id": anchor,
                "split": next(iter(splits)),
                "source_window_component_id": next(iter(components)),
                "videos": records,
            }
        )
    return sorted(groups, key=lambda item: item["anchor_capture_group_id"])


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render_contact_sheet(source: Path, output: Path) -> None:
    # 30 samples across a five-minute source clip: 0, 10, ... 290 s.
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                "fps=1/10,scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2,tile=5x6"
            ),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output),
        ]
    )


def render_overview(sources: list[Path], output: Path) -> None:
    if len(sources) == 1:
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(sources[0]),
                "-vf",
                (
                    "scale=960:540:force_original_aspect_ratio=decrease,"
                    "pad=960:540:(ow-iw)/2:(oh-ih)/2"
                ),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "29",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        return
    if len(sources) != 2:
        raise ValueError("review pack supports one or two selected views per anchor group")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(sources[0]),
            "-i",
            str(sources[1]),
            "-filter_complex",
            (
                "[0:v]scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2[left];"
                "[1:v]scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2[right];"
                "[left][right]hstack=inputs=2:shortest=1[out]"
            ),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "29",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def render_pack(
    video_root: Path,
    manifest_path: Path,
    handbook_path: Path,
    sampling_protocol_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    validate_sources(video_root, manifest)
    groups = review_groups(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    overviews = out_dir / "overviews"
    contacts = out_dir / "contact-sheets"
    overviews.mkdir(exist_ok=True)
    contacts.mkdir(exist_ok=True)

    index_groups: list[dict[str, Any]] = []
    for number, group in enumerate(groups, start=1):
        group_name = f"{number:02d}-{safe_name(group['anchor_capture_group_id'])}"
        source_paths = [
            video_root / f"{record['source_video_id']}.avi" for record in group["videos"]
        ]
        overview_path = overviews / f"{group_name}.mp4"
        render_overview(source_paths, overview_path)

        view_entries = []
        for side_index, (record, source_path) in enumerate(
            zip(group["videos"], source_paths)
        ):
            contact_path = contacts / f"{group_name}-{record['camera']}.jpg"
            render_contact_sheet(source_path, contact_path)
            view_entries.append(
                {
                    "side": "left" if side_index == 0 else "right",
                    "camera": record["camera"],
                    "source_video_id": record["source_video_id"],
                    "source_sha256": record["sha256"],
                    "fps": record["fps"],
                    "frame_count": record["frame_count"],
                    "duration_seconds": record["duration_seconds"],
                    "contact_sheet": str(contact_path.relative_to(out_dir)),
                }
            )

        index_groups.append(
            {
                "review_group": group_name,
                "split": group["split"],
                "source_window_component_id": group["source_window_component_id"],
                "anchor_capture_group_id": group["anchor_capture_group_id"],
                "overview_proxy": str(overview_path.relative_to(out_dir)),
                "views": view_entries,
            }
        )

    index = {
        "schema": REVIEW_SCHEMA,
        "status": "review_only_not_semantic_annotation",
        "source_manifest_sha256": sha256_file(manifest_path),
        "sampling_protocol_sha256": sha256_file(sampling_protocol_path),
        "source_artifact_head_sha": manifest["acquisition_head_sha"],
        "science_head_sha": manifest["science_head_sha"],
        "rules": {
            "proxy_is_not_source_evidence": True,
            "no_detector_or_tracker_run": True,
            "no_r1_e3_score_computed": True,
            "registered_labels": list(REGISTERED_LABELS),
            "contact_sheet_sampling_seconds": 10,
            "human_review_required": True,
            "frame_bounds_must_be_confirmed_against_frozen_source": True,
        },
        "groups": index_groups,
    }
    index_path = out_dir / "review-index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(out_dir, index)
    write_annotation_template(out_dir)
    write_review_completion_template(out_dir, manifest)
    write_annotation_review_template(
        out_dir,
        manifest,
        manifest_path,
        handbook_path,
        sampling_protocol_path,
    )

    digest_lines = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if path.name == "review-pack.sha256":
            continue
        digest_lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir)}")
    (out_dir / "review-pack.sha256").write_text(
        "\n".join(digest_lines) + "\n", encoding="utf-8"
    )
    return index



def write_annotation_review_template(
    out_dir: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    handbook_path: Path,
    sampling_protocol_path: Path,
) -> None:
    videos = sorted(
        manifest["videos"],
        key=lambda record: (
            0 if str(record["split"]) == "train" else 1,
            str(record["source_video_id"]),
        ),
    )
    template = {
        "schema": "r1-e3a-semantic-annotation-review-v1",
        "status": "draft_full_video_review_pending",
        "science_head_sha": manifest["science_head_sha"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "handbook_sha256": sha256_file(handbook_path),
        "sampling_protocol_sha256": sha256_file(sampling_protocol_path),
        "video_reviews": [
            {
                "video_id": record["source_video_id"],
                "full_range_reviewed": False,
                "candidate_count": 0,
                "reviewer": "UNASSIGNED",
            }
            for record in videos
        ],
        "records": [],
    }
    (out_dir / "annotation-review-template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_annotation_template(out_dir: Path) -> None:
    template_path = out_dir / "semantic-annotation-template.csv"
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )


def write_review_completion_template(
    out_dir: Path,
    manifest: dict[str, Any],
) -> None:
    template_path = out_dir / "semantic-review-completion-template.csv"
    videos = sorted(
        manifest["videos"],
        key=lambda record: str(record["source_video_id"]),
    )
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_video_id",
                "full_range_reviewed",
                "reviewed_candidate_count",
                "reviewer",
            )
        )
        for record in videos:
            writer.writerow(
                (
                    record["source_video_id"],
                    "false",
                    "0",
                    "UNASSIGNED",
                )
            )


def write_readme(out_dir: Path, index: dict[str, Any]) -> None:
    lines = [
        "# R1-E3A human semantic review pack",
        "",
        "This pack is review-only. The frozen source AVI bytes remain authoritative.",
        "No detector/tracker/model output was used to render or select these proxies.",
        "",
        "## Frozen review rules",
        "",
        (
            "- One physical actor-target event receives exactly one maximal-outcome "
            "label: approach, touch, pick_up, or pass_by."
        ),
        (
            "- Multiple synchronized camera views of the same event share one "
            "physical_event_id; only one view may be the registered episode view."
        ),
        (
            "- Record authoritative bounds as [start_frame_inclusive, "
            "end_frame_exclusive) against the frozen source video."
        ),
        (
            "- Proxies/contact sheets are navigation aids only; ambiguous contact/"
            "pickup decisions must be checked against the source bytes."
        ),
        "- Do not inspect detector/tracker/decoder outputs while annotating.",
        (
            "- Complete a full chronological source-video scan for every activated "
            "video; do not stop when a class quota is reached."
        ),
        (
            "- Fill semantic-review-completion-template.csv only after the full "
            "source-frame range has been reviewed."
        ),
        "",
        "## Review groups",
        "",
    ]
    for group in index["groups"]:
        lines.extend(
            [
                f"### {group['review_group']}",
                f"- split: {group['split']}",
                f"- source-window component: {group['source_window_component_id']}",
                f"- overview: {group['overview_proxy']}",
            ]
        )
        for view in group["views"]:
            lines.append(
                f"- {view['side']}: {view['camera']} / "
                f"{view['source_video_id']} / contact sheet "
                f"{view['contact_sheet']}"
            )
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, required=True)
    parser.add_argument("--sampling-protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    index = render_pack(
        args.video_root,
        args.manifest,
        args.handbook,
        args.sampling_protocol,
        args.out_dir,
    )
    print(json.dumps({"review_groups": len(index["groups"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
