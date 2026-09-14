from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TrackSummaryError(ValueError):
    pass


@dataclass(frozen=True)
class TrackRecord:
    track_id: int
    timestamp: float
    class_id: int
    lateral: str
    radial: str


@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    class_id: int
    start: float
    end: float
    samples: int
    lateral_counts: dict[str, int]
    radial_counts: dict[str, int]


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrackSummaryError(f"{context} must be a mapping")
    return value


def load_track_records(path: str | Path) -> list[TrackRecord]:
    records: list[TrackRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = _require_mapping(
                    json.loads(line), context=f"prediction line {line_number}"
                )
            except json.JSONDecodeError as exc:
                raise TrackSummaryError(f"invalid JSON at line {line_number}") from exc
            try:
                record = TrackRecord(
                    track_id=int(row["track_id"]),
                    timestamp=float(row["timestamp"]),
                    class_id=int(row["class_id"]),
                    lateral=str(row["lateral"]),
                    radial=str(row["radial"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TrackSummaryError(
                    f"prediction line {line_number} requires "
                    "track_id/timestamp/class_id/lateral/radial"
                ) from exc
            records.append(record)
    return records


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def summarize_tracks(records: list[TrackRecord]) -> list[TrackSummary]:
    grouped: dict[int, list[TrackRecord]] = {}
    for record in records:
        grouped.setdefault(record.track_id, []).append(record)

    summaries: list[TrackSummary] = []
    for track_id in sorted(grouped):
        rows = sorted(grouped[track_id], key=lambda item: item.timestamp)
        summaries.append(
            TrackSummary(
                track_id=track_id,
                class_id=_mode([row.class_id for row in rows]),
                start=rows[0].timestamp,
                end=rows[-1].timestamp,
                samples=len(rows),
                lateral_counts=dict(sorted(Counter(row.lateral for row in rows).items())),
                radial_counts=dict(sorted(Counter(row.radial for row in rows).items())),
            )
        )
    return summaries


def _format_distribution(counts: dict[str, int], samples: int) -> str:
    if samples == 0:
        return "-"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name}:{count / samples:.0%}" for name, count in ordered)


def render_summary_table(summaries: list[TrackSummary]) -> str:
    header = "track_id  class_id  start   end     samples  lateral                  radial"
    lines = [header]
    for summary in summaries:
        lines.append(
            f"{summary.track_id:<8}  {summary.class_id:<8}  "
            f"{summary.start:<6.2f}  {summary.end:<6.2f}  {summary.samples:<7}  "
            f"{_format_distribution(summary.lateral_counts, summary.samples):<23}  "
            f"{_format_distribution(summary.radial_counts, summary.samples)}"
        )
    return "\n".join(lines)


def render_annotation_draft(video: str, summaries: list[TrackSummary]) -> str:
    lines = [f"video: {video}", "intervals:"]
    for summary in summaries:
        lines.extend(
            [
                f"  - track_id: {summary.track_id}",
                f"    start: {summary.start:.2f}",
                f"    end: {summary.end:.2f}",
                "    # lateral:",
                "    # radial:",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="summarize-tracks",
        description="Summarize track IDs and create a human-labeled annotation draft.",
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--annotation-draft", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = summarize_tracks(load_track_records(args.predictions))
    print(render_summary_table(summaries))

    if args.annotation_draft is not None:
        video = f"benchmarks/videos/{args.annotation_draft.stem}.mp4"
        args.annotation_draft.parent.mkdir(parents=True, exist_ok=True)
        args.annotation_draft.write_text(
            render_annotation_draft(video, summaries), encoding="utf-8"
        )
    return 0
