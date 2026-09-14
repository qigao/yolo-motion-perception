import json
from pathlib import Path

import pytest

from yolo_motion.benchmark_suite import (
    BenchmarkScenario,
    aggregate_results,
    build_inference_command,
    load_suite,
    main,
    render_markdown,
    run_suite,
    summarize_scenario,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_suite_resolves_paths_and_metadata(tmp_path: Path):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
scenarios:
  - name: approaching
    video: videos/approaching.mp4
    annotations: labels/approaching.yaml
    tags: [person, frontal]
    expected:
      lateral: stationary
      radial: approaching
  - name: disabled_case
    video: videos/disabled.mp4
    annotations: labels/disabled.yaml
    enabled: false
""".strip(),
        encoding="utf-8",
    )

    suite = load_suite(suite_path)

    assert [scenario.name for scenario in suite.scenarios] == ["approaching", "disabled_case"]
    first = suite.scenarios[0]
    assert first.video == tmp_path / "videos/approaching.mp4"
    assert first.annotations == tmp_path / "labels/approaching.yaml"
    assert first.tags == ("person", "frontal")
    assert first.expected_lateral == "stationary"
    assert first.expected_radial == "approaching"
    assert first.enabled is True
    assert suite.scenarios[1].enabled is False


def test_summarize_scenario_averages_first_correct_latency():
    scenario = BenchmarkScenario(
        name="approaching",
        video=Path("approaching.mp4"),
        annotations=Path("approaching.yaml"),
    )
    report = {
        "lateral": {"samples": 4, "correct": 3, "accuracy": 0.75, "confusion": {}},
        "radial": {"samples": 4, "correct": 4, "accuracy": 1.0, "confusion": {}},
        "intervals": [
            {"first_correct_latency": {"lateral": 0.2, "radial": 0.3}},
            {"first_correct_latency": {"lateral": None, "radial": 0.5}},
        ],
    }

    result = summarize_scenario(scenario, report)

    assert result["lateral_latency"] == pytest.approx(0.2)
    assert result["radial_latency"] == pytest.approx(0.4)
    assert result["lateral_accuracy"] == 0.75
    assert result["radial_accuracy"] == 1.0


def test_aggregate_results_weights_accuracy_by_sample_count():
    results = [
        {
            "name": "a",
            "lateral_samples": 10,
            "lateral_correct": 8,
            "lateral_accuracy": 0.8,
            "lateral_latency": 0.2,
            "radial_samples": 5,
            "radial_correct": 4,
            "radial_accuracy": 0.8,
            "radial_latency": 0.3,
        },
        {
            "name": "b",
            "lateral_samples": 2,
            "lateral_correct": 2,
            "lateral_accuracy": 1.0,
            "lateral_latency": 0.4,
            "radial_samples": 5,
            "radial_correct": 5,
            "radial_accuracy": 1.0,
            "radial_latency": None,
        },
    ]

    summary = aggregate_results(results)

    assert summary["overall"]["lateral"]["samples"] == 12
    assert summary["overall"]["lateral"]["correct"] == 10
    assert summary["overall"]["lateral"]["accuracy"] == pytest.approx(10 / 12)
    assert summary["overall"]["radial"]["accuracy"] == pytest.approx(9 / 10)
    assert summary["overall"]["lateral"]["mean_latency"] == pytest.approx(0.3)
    assert summary["overall"]["radial"]["mean_latency"] == pytest.approx(0.3)


def test_render_markdown_contains_scenario_and_overall_rows():
    summary = aggregate_results(
        [
            {
                "name": "approaching",
                "lateral_samples": 5,
                "lateral_correct": 5,
                "lateral_accuracy": 1.0,
                "lateral_latency": 0.2,
                "radial_samples": 5,
                "radial_correct": 4,
                "radial_accuracy": 0.8,
                "radial_latency": 0.3,
            }
        ]
    )

    markdown = render_markdown(summary)

    assert "| approaching |" in markdown
    assert "| **overall** |" in markdown
    assert "80.0%" in markdown
    assert "0.300s" in markdown


def test_build_inference_command_uses_existing_video_cli(tmp_path: Path):
    scenario = BenchmarkScenario(
        name="approaching",
        video=tmp_path / "approaching.mp4",
        annotations=tmp_path / "approaching.yaml",
    )
    output = tmp_path / "runs/approaching/motion.jsonl"

    command = build_inference_command(
        scenario,
        output,
        model="yolo11n.pt",
        tracker="botsort.yaml",
        config="configs/baseline.yaml",
        confidence=0.35,
        python_executable="python-test",
    )

    assert command[:3] == ["python-test", "-m", "yolo_motion.cli"]
    assert command[command.index("--source") + 1] == str(scenario.video)
    assert command[command.index("--jsonl") + 1] == str(output)
    assert command[command.index("--model") + 1] == "yolo11n.pt"
    assert command[command.index("--tracker") + 1] == "botsort.yaml"
    assert command[command.index("--config") + 1] == "configs/baseline.yaml"
    assert command[command.index("--conf") + 1] == "0.35"


def test_run_suite_executes_enabled_scenarios_and_writes_summary(tmp_path: Path):
    suite_path = tmp_path / "suite.yaml"
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        """
intervals:
  - track_id: 1
    start: 0.0
    end: 1.0
    lateral: stationary
    radial: approaching
""".strip(),
        encoding="utf-8",
    )
    suite_path.write_text(
        """
scenarios:
  - name: approaching
    video: approaching.mp4
    annotations: labels.yaml
  - name: disabled
    video: disabled.mp4
    annotations: labels.yaml
    enabled: false
""".strip(),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        output = Path(command[command.index("--jsonl") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            output,
            [
                {
                    "track_id": 1,
                    "timestamp": 0.4,
                    "lateral": "stationary",
                    "radial": "approaching",
                }
            ],
        )

    output_dir = tmp_path / "runs"
    summary = run_suite(
        suite_path,
        output_dir,
        command_runner=fake_runner,
        python_executable="python-test",
    )

    assert len(calls) == 1
    assert summary["overall"]["radial"]["accuracy"] == 1.0
    assert (output_dir / "approaching/report.json").exists()
    assert (output_dir / "summary.json").exists()
    assert "| approaching |" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_run_suite_evaluate_only_reuses_existing_predictions(tmp_path: Path):
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        """
intervals:
  - track_id: 1
    start: 0.0
    end: 1.0
    radial: receding
""".strip(),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
scenarios:
  - name: receding
    video: receding.mp4
    annotations: labels.yaml
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "runs/receding/motion.jsonl"
    output.parent.mkdir(parents=True)
    _write_jsonl(
        output,
        [{"track_id": 1, "timestamp": 0.5, "lateral": "stationary", "radial": "receding"}],
    )

    def forbidden_runner(command: list[str], *, check: bool) -> None:
        raise AssertionError(f"runner should not be called: {command}, {check}")

    summary = run_suite(
        suite_path,
        tmp_path / "runs",
        evaluate_only=True,
        command_runner=forbidden_runner,
    )

    assert summary["overall"]["radial"]["accuracy"] == 1.0


def test_main_evaluate_only_writes_suite_reports(tmp_path: Path):
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        """
intervals:
  - track_id: 2
    start: 0.0
    end: 1.0
    radial: approaching
""".strip(),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
scenarios:
  - name: approaching
    video: approaching.mp4
    annotations: labels.yaml
""".strip(),
        encoding="utf-8",
    )
    output = tmp_path / "runs/approaching/motion.jsonl"
    output.parent.mkdir(parents=True)
    _write_jsonl(
        output,
        [{"track_id": 2, "timestamp": 0.5, "lateral": "stationary", "radial": "approaching"}],
    )

    assert main(
        [
            "--suite",
            str(suite),
            "--output-dir",
            str(tmp_path / "runs"),
            "--evaluate-only",
        ]
    ) == 0
    assert (tmp_path / "runs/summary.md").exists()
