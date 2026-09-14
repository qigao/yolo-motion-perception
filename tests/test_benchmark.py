from yolo_motion.benchmark import run_benchmark


def test_synthetic_benchmark_recovers_expected_states():
    summary = run_benchmark()

    assert summary["stationary"]["lateral"] == "stationary"
    assert summary["stationary"]["radial"] == "stable"
    assert summary["lateral"]["lateral"] == "moving"
    assert summary["lateral"]["radial"] == "stable"
    assert summary["approaching"]["radial"] == "approaching"
    assert summary["receding"]["radial"] == "receding"
    assert summary["scale_spike"]["radial"] == "stable"
