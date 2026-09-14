from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_suite import load_suite
from .evaluation import EvaluationError, load_annotations


def inspect_dataset(suite_path: str | Path) -> dict[str, Any]:
    suite = load_suite(suite_path)
    rows: list[dict[str, Any]] = []
    enabled_count = 0
    ready_count = 0

    for scenario in suite.scenarios:
        if not scenario.enabled:
            rows.append(
                {
                    "name": scenario.name,
                    "status": "disabled",
                    "video": str(scenario.video),
                    "annotations": str(scenario.annotations),
                    "video_exists": scenario.video.is_file(),
                    "annotations_exists": scenario.annotations.is_file(),
                    "interval_count": None,
                    "issues": [],
                }
            )
            continue

        enabled_count += 1
        issues: list[str] = []
        video_exists = scenario.video.is_file() and scenario.video.stat().st_size > 0
        annotations_exists = scenario.annotations.is_file()
        interval_count: int | None = None

        if not video_exists:
            issues.append("missing_video")
        if not annotations_exists:
            issues.append("missing_annotations")
        else:
            try:
                annotation_set = load_annotations(scenario.annotations)
            except (EvaluationError, OSError, ValueError):
                issues.append("invalid_annotations")
            else:
                interval_count = len(annotation_set.intervals)
                if interval_count == 0:
                    issues.append("empty_annotations")

        status = "ready" if not issues else "not_ready"
        if status == "ready":
            ready_count += 1
        rows.append(
            {
                "name": scenario.name,
                "status": status,
                "video": str(scenario.video),
                "annotations": str(scenario.annotations),
                "video_exists": video_exists,
                "annotations_exists": annotations_exists,
                "interval_count": interval_count,
                "issues": issues,
            }
        )

    return {
        "ready": enabled_count > 0 and ready_count == enabled_count,
        "enabled_scenarios": enabled_count,
        "ready_scenarios": ready_count,
        "scenarios": rows,
    }


def render_dataset_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Dataset Readiness",
        "",
        f"Overall ready: **{'yes' if report['ready'] else 'no'}**",
        "",
        f"Enabled scenarios: {report['enabled_scenarios']}",
        f"Ready scenarios: {report['ready_scenarios']}",
        "",
        "| Scenario | Status | Video | Annotations | Intervals | Issues |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in report["scenarios"]:
        interval_count = "-" if row["interval_count"] is None else str(row["interval_count"])
        issues = ", ".join(row["issues"]) if row["issues"] else "-"
        lines.append(
            f"| {row['name']} | {row['status']} | "
            f"{'yes' if row['video_exists'] else 'no'} | "
            f"{'yes' if row['annotations_exists'] else 'no'} | "
            f"{interval_count} | {issues} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check-benchmark-dataset",
        description="Check whether enabled real-video benchmark scenarios are ready.",
    )
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite.yaml"))
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_dataset(args.suite)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_dataset_markdown(report), end="")
    return 0 if report["ready"] else 2
