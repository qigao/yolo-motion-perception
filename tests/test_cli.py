from pathlib import Path

from yolo_motion.cli import build_parser, load_motion_config


def test_load_motion_config_reads_yaml_thresholds(tmp_path: Path):
    path = tmp_path / "motion.yaml"
    path.write_text(
        "history_seconds: 1.5\n"
        "min_samples: 6\n"
        "min_duration: 0.4\n"
        "stationary_speed_threshold: 0.03\n"
        "radial_rate_threshold: 0.2\n"
        "min_radial_confidence: 0.7\n"
        "min_trend_consistency: 0.8\n",
        encoding="utf-8",
    )

    config = load_motion_config(path)

    assert config.history_seconds == 1.5
    assert config.min_samples == 6
    assert config.radial_rate_threshold == 0.2


def test_cli_parser_defaults_to_botsort_and_yolo11n():
    args = build_parser().parse_args(["--source", "0"])

    assert args.model == "yolo11n.pt"
    assert args.tracker == "botsort.yaml"
    assert args.source == "0"
