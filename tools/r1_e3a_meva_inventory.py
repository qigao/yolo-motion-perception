#!/usr/bin/env python3
"""Enumerate a small MEVA source-video candidate inventory for R1-E3A.

This is acquisition-only tooling. It reads public MEVA KPF annotations to find
source videos containing person_picks_up_object and related interaction events,
then confirms the corresponding public S3 objects with HEAD requests.

It does NOT:
- download video bytes;
- run YOLO, BoT-SORT, OpenCV inference, or any R1-E3 model;
- assign R1-E3 approach/touch/pick_up/pass_by labels;
- create a frozen source manifest;
- compute any registered measurement.

The output is only a reviewable candidate inventory used before source-video
selection/split freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

DEFAULT_BUCKET = "mevadata-public-01"
DEFAULT_PREFIX = "drops-123-r13"

DISCOVERY_ACTIVITIES = (
    "person_picks_up_object",
    "person_sets_down_object",
    "hand_interaction",
    "object_transfer",
)

KPF_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\."
    r"(?P<hh>\d{2})-(?P<mm>\d{2})-(?P<ss>\d{2})\."
    r"[\d-]+\.(?P<site>\w+)\.(?P<camera>G\d+)"
    r"[.-]activities\.yml$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_activity_counts(path: Path) -> Counter[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    counts: Counter[str] = Counter()
    if not isinstance(doc, list):
        return counts
    for item in doc:
        if not isinstance(item, dict):
            continue
        act = item.get("act")
        if not isinstance(act, dict):
            continue
        act2 = act.get("act2") or {}
        if not isinstance(act2, dict) or not act2:
            continue
        name = str(next(iter(act2)))
        if name in DISCOVERY_ACTIVITIES:
            counts[name] += 1
    return counts


def derive_video_identity(path: Path, prefix: str) -> dict[str, str] | None:
    match = KPF_RE.search(path.name)
    if not match:
        return None
    stem = re.sub(r"[.-]activities\.yml$", "", path.name)
    date = match.group("date")
    hour = match.group("hh")
    return {
        "source_video_id": stem,
        "date": date,
        "hour": hour,
        "site": match.group("site"),
        "camera": match.group("camera"),
        "s3_key": f"{prefix}/{date}/{hour}/{stem}.r13.avi",
    }


def head_video(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        response = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        return {
            "available": False,
            "error_code": str(error.get("Code", "unknown")),
            "error_message": str(error.get("Message", "unknown")),
        }
    return {
        "available": True,
        "byte_size": int(response["ContentLength"]),
        "s3_etag": str(response.get("ETag", "")).strip('"'),
        "last_modified_utc": response["LastModified"].astimezone(timezone.utc).isoformat(),
    }


def build_inventory(kpf_root: Path, bucket: str, prefix: str) -> dict[str, Any]:
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )

    annotation_files = sorted(kpf_root.rglob("*activities.yml"))
    candidates: list[dict[str, Any]] = []
    malformed_names = 0
    pickup_files = 0

    for path in annotation_files:
        counts = load_activity_counts(path)
        if counts["person_picks_up_object"] <= 0:
            continue
        pickup_files += 1
        identity = derive_video_identity(path, prefix)
        if identity is None:
            malformed_names += 1
            continue
        head = head_video(s3, bucket, identity["s3_key"])
        candidates.append(
            {
                **identity,
                "annotation_relpath": str(path.relative_to(kpf_root)),
                "annotation_sha256": sha256_file(path),
                "discovery_activity_counts": {
                    name: int(counts[name]) for name in DISCOVERY_ACTIVITIES
                },
                "s3": head,
                "video_sha256": None,
                "split": None,
                "r1_e3_labels_frozen": False,
            }
        )

    candidates.sort(key=lambda row: row["source_video_id"])
    available = [row for row in candidates if row["s3"]["available"]]
    totals = Counter()
    for row in available:
        totals.update(row["discovery_activity_counts"])

    return {
        "schema": "r1-e3a-meva-candidate-inventory-v1",
        "status": "candidate_inventory_only_not_frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": {
            "dataset": "MEVA",
            "license": "CC BY 4.0",
            "bucket": f"s3://{bucket}/",
            "note": "MEVA annotations are used only for source-video discovery.",
        },
        "constraints": {
            "no_video_bytes_downloaded": True,
            "no_detector_or_tracker_run": True,
            "no_r1_e3_score_computed": True,
            "no_split_assigned": True,
            "video_sha256_pending": True,
            "registered_labels_pending_human_freeze": [
                "approach",
                "touch",
                "pick_up",
                "pass_by",
            ],
        },
        "summary": {
            "annotation_files_scanned": len(annotation_files),
            "pickup_annotation_files": pickup_files,
            "candidate_records": len(candidates),
            "s3_confirmed_candidate_videos": len(available),
            "s3_unavailable_candidates": len(candidates) - len(available),
            "malformed_annotation_filenames": malformed_names,
            "distinct_dates": len({row["date"] for row in available}),
            "distinct_cameras": len({row["camera"] for row in available}),
            "confirmed_discovery_activity_instances": {
                name: int(totals[name]) for name in DISCOVERY_ACTIVITIES
            },
        },
        "candidates": candidates,
    }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "meva_candidate_inventory.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = out_dir / "meva_candidate_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video_id",
                "date",
                "hour",
                "site",
                "camera",
                "s3_key",
                "available",
                "byte_size",
                *DISCOVERY_ACTIVITIES,
                "annotation_sha256",
            ],
        )
        writer.writeheader()
        for row in payload["candidates"]:
            counts = row["discovery_activity_counts"]
            writer.writerow(
                {
                    "source_video_id": row["source_video_id"],
                    "date": row["date"],
                    "hour": row["hour"],
                    "site": row["site"],
                    "camera": row["camera"],
                    "s3_key": row["s3_key"],
                    "available": row["s3"]["available"],
                    "byte_size": row["s3"].get("byte_size", ""),
                    **{name: counts[name] for name in DISCOVERY_ACTIVITIES},
                    "annotation_sha256": row["annotation_sha256"],
                }
            )

    digest_path = out_dir / "inventory_files.sha256"
    digest_path.write_text(
        f"{sha256_file(json_path)}  {json_path.name}\n"
        f"{sha256_file(csv_path)}  {csv_path.name}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kpf-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    payload = build_inventory(args.kpf_root, args.bucket, args.prefix)
    write_outputs(payload, args.out_dir)

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"candidate inventory -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
