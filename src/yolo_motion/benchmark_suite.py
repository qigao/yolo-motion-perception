from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .evaluation import evaluate_predictions, load_annotations, load_predictions


class BenchmarkSuiteError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    video: Path
    annotations: Path
    tags: tuple[str, ...] = ()
    expected_lateral: str | None = None
    expected_radial: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class BenchmarkSuite:
    scenarios: tuple[BenchmarkScenario, ...]


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkSuiteError(f"{context} must be a mapping")
    return value


def _resolve_path(base: Path, value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BenchmarkSuiteError(f"{context} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_suite(path: str | Path) -> BenchmarkSuite:
    suite_path = Path(path)
    with suite_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    root = _mapping(raw, context="benchmark suite")
    raw_scenarios = root.get("scenarios", [])
    if not isinstance(raw_scenarios, list):
        raise BenchmarkSuiteError("scenarios must be a list")

    scenarios: list[BenchmarkScenario] = []
    base = suite_path.parent
    for index, item in enumerate(raw_scenarios):
        row = _mapping(item, context=f"scenarios[{index}]")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise BenchmarkSuiteError(f"scenarios[{index}].name must be a non-empty string")
        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise BenchmarkSuiteError(f"scenarios[{index}].tags must be a list of strings")
        expected = row.get("expected", {})
        if expected is None:
            expected = {}
        expected = _mapping(expected, context=f"scenarios[{index}].expected")
        enabled = row.get("enabled", True)
        if not isinstance(enabled, bool):
            raise BenchmarkSuiteError(f"scenarios[{index}].enabled must be a boolean")
        scenarios.append(
            BenchmarkScenario(
                name=name,
                video=_resolve_path(base, row.get("video"), context=f"scenarios[{index}].video"),
                annotations=_resolve_path(
                    base,
                    row.get("annotations"),
                    context=f"scenarios[{index}].annotations",
                ),
                tags=tuple(tags),
                expected_lateral=expected.get("lateral"),
                expected_radial=expected.get("radial"),
                enabled=enabled,
            )
        )
    return BenchmarkSuite(scenarios=tuple(scenarios))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_scenario(scenario: BenchmarkScenario, report: dict[str, Any]) -> dict[str, Any]:
    lateral_latencies: list[float] = []
    radial_latencies: list[float] = []
    for interval in report.get("intervals", []):
        latencies = interval.get("first_correct_latency", {})
        lateral = latencies.get("lateral")
        radial = latencies.get("radial")
        if lateral is not None:
            lateral_latencies.append(float(lateral))
        if radial is not None:
            radial_latencies.append(float(radial))

    lateral = report["lateral"]
    radial = report["radial"]
    return {
        "name": scenario.name,
        "tags": list(scenario.tags),
        "expected_lateral": scenario.expected_lateral,
        "expected_radial": scenario.expected_radial,
        "lateral_samples": int(lateral["samples"]),
        "lateral_correct": int(lateral["correct"]),
        "lateral_accuracy": lateral["accuracy"],
        "lateral_latency": _mean(lateral_latencies),
        "radial_samples": int(radial["samples"]),
        "radial_correct": int(radial["correct"]),
        "radial_accuracy": radial["accuracy"],
        "radial_latency": _mean(radial_latencies),
    }


def _aggregate_dimension(results: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    samples = sum(int(result[f"{dimension}_samples"]) for result in results)
    correct = sum(int(result[f"{dimension}_correct"]) for result in results)
    latencies = [
        float(result[f"{dimension}_latency"])
        for result in results
        if result.get(f"{dimension}_latency") is not None
    ]
    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples if samples else None,
        "mean_latency": _mean(latencies),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": results,
        "overall": {
            "lateral": _aggregate_dimension(results, "lateral"),
            "radial": _aggregate_dimension(results, "radial"),
        },
    }


def _format_accuracy(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_latency(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# YOLO Motion Benchmark Summary",
        "",
        "| Scenario | Lateral Acc | Radial Acc | Lateral Latency | Radial Latency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["scenarios"]:
        lines.append(
            "| {name} | {lateral} | {radial} | {lateral_latency} | {radial_latency} |".format(
                name=result["name"],
                lateral=_format_accuracy(result.get("lateral_accuracy")),
                radial=_format_accuracy(result.get("radial_accuracy")),
                lateral_latency=_format_latency(result.get("lateral_latency")),
                radial_latency=_format_latency(result.get("radial_latency")),
            )
        )
    overall = summary["overall"]
    lines.append(
        "| **overall** | {lateral} | {radial} | {lateral_latency} | {radial_latency} |".format(
            lateral=_format_accuracy(overall["lateral"]["accuracy"]),
            radial=_format_accuracy(overall["radial"]["accuracy"]),
            lateral_latency=_format_latency(overall["lateral"]["mean_latency"]),
            radial_latency=_format_latency(overall["radial"]["mean_latency"]),
        )
    )
    return "\n".join(lines) + "\n"


def build_inference_command(
    scenario: BenchmarkScenario,
    output_jsonl: Path,
    *,
    model: str,
    tracker: str,
    config: str,
    confidence: float,
    python_executable: str,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "yolo_motion.cli",
        "--source",
        str(scenario.video),
        "--model",
        model,
        "--tracker",
        tracker,
        "--config",
        config,
        "--conf",
        str(confidence),
        "--jsonl",
        str(output_jsonl),
    ]


def run_suite(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = "yolo11n.pt",
    tracker: str = "botsort.yaml",
    config: str = "configs/baseline.yaml",
    confidence: float = 0.25,
    evaluate_only: bool = False,
    python_executable: str = sys.executable,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for scenario in suite.scenarios:
        if not scenario.enabled:
            continue
        scenario_dir = root / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = scenario_dir / "motion.jsonl"
        if not evaluate_only:
            command = build_inference_command(
                scenario,
                predictions_path,
                model=model,
                tracker=tracker,
                config=config,
                confidence=confidence,
                python_executable=python_executable,
            )
            command_runner(command, check=True)

        report = evaluate_predictions(
            load_predictions(predictions_path),
            load_annotations(scenario.annotations),
        )
        (scenario_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(summarize_scenario(scenario, report))

    summary = aggregate_results(results)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-benchmark-suite",
        description="Run and summarize a real-video YOLO motion benchmark suite.",
    )
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reuse existing per-scenario motion.jsonl files without running YOLO.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_suite(
        args.suite,
        args.output_dir,
        model=args.model,
        tracker=args.tracker,
        config=args.config,
        confidence=args.conf,
        evaluate_only=args.evaluate_only,
    )
    print(render_markdown(summary), end="")
    return 0
