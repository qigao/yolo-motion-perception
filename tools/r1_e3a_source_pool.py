#!/usr/bin/env python3
"""Build a deterministic R1-E3A MEVA source-pool plan from a frozen inventory.

Acquisition only: no video inference, tracking, registered labels, or decoder scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "r1-e3a-meva-candidate-inventory-v1"
PLAN_SCHEMA = "r1-e3a-meva-source-pool-plan-v1"
SOURCE_WINDOW_GUARD_SECONDS = 1
SITE = "school"
SPLIT_SALT = "r1-e3a-meva-source-pool-v1"
MAX_VIEWS_PER_ANCHOR_GROUP = 2
MAX_TRAIN_COMPONENTS = 12
MAX_EVAL_COMPONENTS = 6

VIDEO_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\."
    r"(?P<start>\d{2}-\d{2}-\d{2})\."
    r"(?P<end>\d{2}-\d{2}-\d{2})\."
    r"(?P<site>[A-Za-z0-9_]+)\."
    r"(?P<camera>G\d+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_clock(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split("-"))
    return hour * 3600 + minute * 60 + second


def format_clock(value: int) -> str:
    hour, rem = divmod(value, 3600)
    minute, second = divmod(rem, 60)
    return f"{hour:02d}-{minute:02d}-{second:02d}"


def parse_video_id(source_video_id: str) -> dict[str, Any]:
    match = VIDEO_RE.fullmatch(source_video_id)
    if match is None:
        raise ValueError(f"unsupported MEVA source_video_id: {source_video_id}")
    parsed = match.groupdict()
    start_s = parse_clock(parsed["start"])
    end_s = parse_clock(parsed["end"])
    if end_s <= start_s:
        raise ValueError(f"non-positive source window: {source_video_id}")
    parsed["start_s"] = start_s
    parsed["end_s"] = end_s
    parsed["capture_group_id"] = (
        f'{parsed["date"]}.{parsed["start"]}.{parsed["end"]}.{parsed["site"]}'
    )
    return parsed


def available_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"expected {INVENTORY_SCHEMA}")
    rows: list[dict[str, Any]] = []
    for row in payload.get("candidates", []):
        if not row.get("s3", {}).get("available"):
            continue
        parsed = parse_video_id(str(row["source_video_id"]))
        rows.append(
            {
                **row,
                **parsed,
                "pickup_count": int(
                    row.get("discovery_activity_counts", {}).get(
                        "person_picks_up_object", 0
                    )
                ),
            }
        )
    return rows


def build_guard_components(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge same-date/site windows that overlap, touch, or are <=1 s apart."""

    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[row["capture_group_id"]].append(row)

    exact_groups: list[dict[str, Any]] = []
    for capture_group_id, members in group_rows.items():
        first = members[0]
        exact_groups.append(
            {
                "capture_group_id": capture_group_id,
                "date": first["date"],
                "site": first["site"],
                "start_s": first["start_s"],
                "end_s": first["end_s"],
                "members": sorted(members, key=lambda row: row["source_video_id"]),
            }
        )

    by_date_site: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in exact_groups:
        by_date_site[(group["date"], group["site"])].append(group)

    components: list[dict[str, Any]] = []
    for (date, site), groups in sorted(by_date_site.items()):
        groups.sort(
            key=lambda row: (row["start_s"], row["end_s"], row["capture_group_id"])
        )
        current: list[dict[str, Any]] = []
        current_start = 0
        current_end = 0

        def flush() -> None:
            nonlocal current, current_start, current_end
            if not current:
                return
            component_id = (
                f"{date}.{format_clock(current_start)}."
                f"{format_clock(current_end)}.{site}.guard1s"
            )
            members = [member for group in current for member in group["members"]]
            components.append(
                {
                    "source_window_component_id": component_id,
                    "date": date,
                    "site": site,
                    "start_s": current_start,
                    "end_s": current_end,
                    "capture_groups": [group["capture_group_id"] for group in current],
                    "members": sorted(
                        members, key=lambda row: row["source_video_id"]
                    ),
                }
            )
            current = []

        for group in groups:
            if not current:
                current = [group]
                current_start = group["start_s"]
                current_end = group["end_s"]
                continue
            if group["start_s"] <= current_end + SOURCE_WINDOW_GUARD_SECONDS:
                current.append(group)
                current_end = max(current_end, group["end_s"])
            else:
                flush()
                current = [group]
                current_start = group["start_s"]
                current_end = group["end_s"]
        flush()

    return components


def split_for_component(component_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{component_id}".encode()).digest()
    return "eval" if int.from_bytes(digest, "big") % 3 == 0 else "train"


def anchor_group(component: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component["members"]:
        by_group[row["capture_group_id"]].append(row)
    ranked = sorted(
        by_group.items(),
        key=lambda item: (
            -sum(row["pickup_count"] for row in item[1]),
            -len(item[1]),
            item[0],
        ),
    )
    return ranked[0]


def batch_for_rank(split: str, rank: int) -> int:
    if split == "train":
        if rank <= 4:
            return 1
        return 2 + (rank - 5) // 2
    if rank <= 2:
        return 1
    return rank - 1


def build_plan(
    payload: dict[str, Any],
    *,
    inventory_sha256: str,
    inventory_workflow_run: int,
) -> dict[str, Any]:
    rows = available_rows(payload)
    components = build_guard_components(rows)
    exact_group_count = len({row["capture_group_id"] for row in rows})

    eligible: list[dict[str, Any]] = []
    for component in components:
        if component["site"] != SITE:
            continue
        group_id, members = anchor_group(component)
        selected = sorted(
            members,
            key=lambda row: (-row["pickup_count"], row["source_video_id"]),
        )[:MAX_VIEWS_PER_ANCHOR_GROUP]
        eligible.append(
            {
                "source_window_component_id": component[
                    "source_window_component_id"
                ],
                "date": component["date"],
                "site": component["site"],
                "split": split_for_component(
                    component["source_window_component_id"]
                ),
                "anchor_capture_group_id": group_id,
                "anchor_pickup_discovery_count": sum(
                    row["pickup_count"] for row in members
                ),
                "selected_views": [
                    {
                        "source_video_id": row["source_video_id"],
                        "camera": row["camera"],
                        "s3_key": row["s3_key"],
                        "expected_byte_size": int(row["s3"]["byte_size"]),
                        "annotation_sha256": row["annotation_sha256"],
                        "pickup_discovery_count": row["pickup_count"],
                    }
                    for row in selected
                ],
            }
        )

    ranked_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, limit in (
        ("train", MAX_TRAIN_COMPONENTS),
        ("eval", MAX_EVAL_COMPONENTS),
    ):
        ranked = sorted(
            (row for row in eligible if row["split"] == split),
            key=lambda row: (
                -row["anchor_pickup_discovery_count"],
                row["source_window_component_id"],
            ),
        )[:limit]
        for rank, row in enumerate(ranked, start=1):
            row["rank_within_split"] = rank
            row["activation_batch"] = batch_for_rank(split, rank)
        ranked_by_split[split] = ranked

    selected_components = ranked_by_split["train"] + ranked_by_split["eval"]
    selected_components.sort(
        key=lambda row: (
            row["activation_batch"],
            row["split"],
            row["rank_within_split"],
        )
    )

    return {
        "schema": PLAN_SCHEMA,
        "status": "pre-extraction_source_pool_ladder",
        "provenance": {
            "inventory_schema": INVENTORY_SCHEMA,
            "inventory_sha256": inventory_sha256,
            "inventory_workflow_run": inventory_workflow_run,
        },
        "policy": {
            "candidate_site": SITE,
            "source_window_guard_seconds": SOURCE_WINDOW_GUARD_SECONDS,
            "source_window_component_rule": (
                "same date/site exact source windows are transitively merged "
                "when the next window starts <= current_end + 1 second"
            ),
            "split_rule": (
                "sha256('r1-e3a-meva-source-pool-v1|' + component_id) mod 3; "
                "0=eval, otherwise=train"
            ),
            "priority_rule": (
                "within each split, descending KPF person_picks_up_object "
                "discovery count in the highest-count exact capture group; "
                "lexical tie-break"
            ),
            "view_rule": (
                "select at most two source videos from the same anchor capture "
                "group, highest discovery count then lexical source_video_id"
            ),
            "activation_rule": (
                "batch 1 activates train ranks 1-4 and eval ranks 1-2; each "
                "later batch adds two train ranks and one eval rank. Expand "
                "only if frozen human semantic counts miss the registered "
                "class minima. Never rerank."
            ),
            "maximum_components": {
                "train": MAX_TRAIN_COMPONENTS,
                "eval": MAX_EVAL_COMPONENTS,
            },
            "discovery_only_statement": (
                "MEVA KPF pickup annotations influence discovery/priority only. "
                "No YOLO, BoT-SORT, R1-E3 labels, track quality, or decoder "
                "score is used."
            ),
        },
        "inventory_structure": {
            "available_source_videos": len(rows),
            "exact_capture_groups": exact_group_count,
            "guarded_source_window_components": len(components),
            "eligible_school_components": len(eligible),
        },
        "components": selected_components,
    }


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"expected {PLAN_SCHEMA}")
    seen_videos: set[str] = set()
    split_components: dict[str, set[str]] = {"train": set(), "eval": set()}
    for component in plan.get("components", []):
        split = component.get("split")
        if split not in split_components:
            raise ValueError(f"invalid split: {split}")
        cid = str(component["source_window_component_id"])
        split_components[split].add(cid)
        for view in component.get("selected_views", []):
            video_id = str(view["source_video_id"])
            if video_id in seen_videos:
                raise ValueError(f"duplicate selected source video: {video_id}")
            seen_videos.add(video_id)
            if int(view["expected_byte_size"]) <= 0:
                raise ValueError(f"invalid expected byte size: {video_id}")
    overlap = split_components["train"] & split_components["eval"]
    if overlap:
        raise ValueError(
            f"source-window component crosses train/eval: {sorted(overlap)}"
        )
    if (
        len(split_components["train"]) < 2
        or len(split_components["eval"]) < 2
    ):
        raise ValueError(
            "need at least two source-window components in each split"
        )


def write_plan(plan: dict[str, Any], output: Path) -> None:
    validate_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--inventory-workflow-run", type=int, required=True)
    args = parser.parse_args()

    actual = sha256_file(args.inventory)
    if actual != args.inventory_sha256:
        raise SystemExit(
            "inventory SHA-256 mismatch: "
            f"expected {args.inventory_sha256}, got {actual}"
        )
    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    plan = build_plan(
        payload,
        inventory_sha256=actual,
        inventory_workflow_run=args.inventory_workflow_run,
    )
    write_plan(plan, args.output)
    print(json.dumps(plan["inventory_structure"], indent=2, sort_keys=True))
    print(f"source pool plan -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
