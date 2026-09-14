from pathlib import Path

from yolo_motion.dataset import inspect_dataset, render_dataset_markdown


def _write_suite(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_inspect_dataset_reports_ready_enabled_scenario(tmp_path: Path):
    video = tmp_path / "videos/approaching.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not-empty")
    labels = tmp_path / "annotations/approaching.yaml"
    labels.parent.mkdir()
    labels.write_text(
        """
video: videos/approaching.mp4
intervals:
  - track_id: 1
    start: 0.5
    end: 2.0
    radial: approaching
""".strip(),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: approaching
    video: videos/approaching.mp4
    annotations: annotations/approaching.yaml
    enabled: true
""",
    )

    report = inspect_dataset(suite)

    assert report["ready"] is True
    assert report["enabled_scenarios"] == 1
    assert report["ready_scenarios"] == 1
    assert report["scenarios"][0]["interval_count"] == 1
    assert report["scenarios"][0]["issues"] == []


def test_inspect_dataset_flags_missing_video_empty_annotations_and_disabled(tmp_path: Path):
    labels = tmp_path / "annotations/approaching.yaml"
    labels.parent.mkdir()
    labels.write_text("intervals: []\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: approaching
    video: videos/approaching.mp4
    annotations: annotations/approaching.yaml
    enabled: true
  - name: ignored
    video: videos/ignored.mp4
    annotations: annotations/ignored.yaml
    enabled: false
""",
    )

    report = inspect_dataset(suite)

    assert report["ready"] is False
    assert report["enabled_scenarios"] == 1
    assert report["ready_scenarios"] == 0
    first = report["scenarios"][0]
    assert first["status"] == "not_ready"
    assert "missing_video" in first["issues"]
    assert "empty_annotations" in first["issues"]
    second = report["scenarios"][1]
    assert second["status"] == "disabled"
    assert second["issues"] == []


def test_inspect_dataset_is_not_ready_when_no_scenarios_are_enabled(tmp_path: Path):
    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: stationary
    video: videos/stationary.mp4
    annotations: annotations/stationary.yaml
    enabled: false
""",
    )

    report = inspect_dataset(suite)

    assert report["ready"] is False
    assert report["enabled_scenarios"] == 0
    assert report["ready_scenarios"] == 0


def test_inspect_dataset_reports_invalid_annotations(tmp_path: Path):
    video = tmp_path / "videos/stationary.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    labels = tmp_path / "annotations/stationary.yaml"
    labels.parent.mkdir()
    labels.write_text("intervals: nope\n", encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: stationary
    video: videos/stationary.mp4
    annotations: annotations/stationary.yaml
""",
    )

    report = inspect_dataset(suite)

    scenario = report["scenarios"][0]
    assert scenario["status"] == "not_ready"
    assert "invalid_annotations" in scenario["issues"]
    assert scenario["interval_count"] is None


def test_render_dataset_markdown_shows_readiness_and_issues(tmp_path: Path):
    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: approaching
    video: videos/approaching.mp4
    annotations: annotations/approaching.yaml
""",
    )

    markdown = render_dataset_markdown(inspect_dataset(suite))

    assert "# Benchmark Dataset Readiness" in markdown
    assert "| approaching | not_ready |" in markdown
    assert "missing_video" in markdown
    assert "missing_annotations" in markdown


def test_dataset_main_returns_two_when_not_ready_and_zero_when_ready(tmp_path: Path, capsys):
    from yolo_motion.dataset import main

    suite = tmp_path / "suite.yaml"
    _write_suite(
        suite,
        """
scenarios:
  - name: approaching
    video: videos/approaching.mp4
    annotations: annotations/approaching.yaml
""",
    )

    assert main(["--suite", str(suite), "--format", "json"]) == 2
    first = capsys.readouterr().out
    assert '"ready": false' in first

    video = tmp_path / "videos/approaching.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    labels = tmp_path / "annotations/approaching.yaml"
    labels.parent.mkdir()
    labels.write_text(
        """
intervals:
  - track_id: 3
    start: 0.0
    end: 1.0
    radial: approaching
""".strip(),
        encoding="utf-8",
    )

    assert main(["--suite", str(suite), "--format", "markdown"]) == 0
    second = capsys.readouterr().out
    assert "Overall ready: **yes**" in second
