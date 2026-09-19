#!/usr/bin/env python3
"""Fail-closed source-overlap grouping for the R1-E3A MEVA corpus.

The split unit is stronger than an exact filename/capture-group match: any source
windows from the same date/site that overlap in time belong to the same connected
component. This closes leakage from one-second camera clock offsets and other
overlapping source windows.

This tool is acquisition-only. It does not run detection, tracking, or scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

VIDEO_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\."
    r"(?P<start>\d{2}-\d{2}-\d{2})\."
    r"(?P<end>\d{2}-\d{2}-\d{2})\."
    r"(?P<site>[A-Za-z0-9_]+)\."
    r"(?P<camera>G\d+)$"
)


@dataclass(frozen=True, order=True)
class SourceWindow:
    source_video_id: str
    date: str
    site: str
    camera: str
    start_seconds: int
    end_seconds: int


def _clock_seconds(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split("-"))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"invalid clock time: {value}")
    return hour * 3600 + minute * 60 + second


def _clock_text(seconds: int) -> str:
    seconds %= 24 * 3600
    hour, remainder = divmod(seconds, 3600)
    minute, second = divmod(remainder, 60)
    return f"{hour:02d}-{minute:02d}-{second:02d}"


def parse_source_window(source_video_id: str) -> SourceWindow:
    match = VIDEO_RE.fullmatch(source_video_id)
    if match is None:
        raise ValueError(f"invalid MEVA source_video_id: {source_video_id}")

    start = _clock_seconds(match.group("start"))
    end = _clock_seconds(match.group("end"))
    if end <= start:
        end += 24 * 3600

    return SourceWindow(
        source_video_id=source_video_id,
        date=match.group("date"),
        site=match.group("site"),
        camera=match.group("camera"),
        start_seconds=start,
        end_seconds=end,
    )


def _component_id(date: str, site: str, start: int, end: int) -> str:
    return (
        f"{date}.{_clock_text(start)}.{_clock_text(end)}.{site}"
        ".source-overlap"
    )


def assign_source_overlap_components(
    source_video_ids: Iterable[str],
) -> dict[str, str]:
    windows = [parse_source_window(video_id) for video_id in source_video_ids]
    grouped: dict[tuple[str, str], list[SourceWindow]] = {}
    for window in windows:
        grouped.setdefault((window.date, window.site), []).append(window)

    assignments: dict[str, str] = {}
    for (date, site), group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda item: (
                item.start_seconds,
                item.end_seconds,
                item.source_video_id,
            ),
        )
        if not ordered:
            continue

        members: list[SourceWindow] = []
        component_start = 0
        component_end = 0

        def flush() -> None:
            if not members:
                return
            component = _component_id(
                date,
                site,
                component_start,
                component_end,
            )
            for member in members:
                assignments[member.source_video_id] = component

        for window in ordered:
            if not members:
                members = [window]
                component_start = window.start_seconds
                component_end = window.end_seconds
                continue

            # Positive temporal overlap joins the same connected component.
            # Exactly touching windows remain separate.
            if window.start_seconds < component_end:
                members.append(window)
                component_end = max(component_end, window.end_seconds)
                continue

            flush()
            members = [window]
            component_start = window.start_seconds
            component_end = window.end_seconds

        flush()

    return assignments


def validate_source_overlap_split(assignments: Mapping[str, str]) -> None:
    invalid = sorted(
        {
            split
            for split in assignments.values()
            if split not in {"train", "eval"}
        }
    )
    if invalid:
        raise ValueError(f"invalid split values: {invalid}")

    components = assign_source_overlap_components(assignments)
    component_splits: dict[str, set[str]] = {}
    for source_video_id, component in components.items():
        component_splits.setdefault(component, set()).add(
            assignments[source_video_id]
        )

    leaked = sorted(
        component
        for component, splits in component_splits.items()
        if len(splits) > 1
    )
    if leaked:
        raise ValueError(
            "source-overlap component crosses split: " + ", ".join(leaked)
        )


def build_component_report(payload: dict) -> dict:
    candidates = [
        row
        for row in payload.get("candidates", [])
        if row.get("s3", {}).get("available") is True
    ]
    video_ids = [row["source_video_id"] for row in candidates]
    assignments = assign_source_overlap_components(video_ids)

    components: dict[str, list[str]] = {}
    for video_id, component in assignments.items():
        components.setdefault(component, []).append(video_id)

    rows = [
        {
            "source_overlap_component_id": component,
            "source_video_ids": sorted(members),
            "video_count": len(members),
        }
        for component, members in sorted(components.items())
    ]
    return {
        "schema": "r1-e3a-source-overlap-components-v1",
        "status": "acquisition_guard_only_not_split_assignment",
        "candidate_inventory_schema": payload.get("schema"),
        "available_source_videos": len(video_ids),
        "source_overlap_components": len(rows),
        "multi_video_components": sum(row["video_count"] > 1 for row in rows),
        "max_videos_in_component": max(
            (row["video_count"] for row in rows),
            default=0,
        ),
        "components": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    report = build_component_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "available_source_videos",
        "source_overlap_components",
        "multi_video_components",
        "max_videos_in_component",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
