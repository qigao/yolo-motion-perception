import math

from neural_state_machine.benchmark import run_benchmark


def test_benchmark_reports_replay_context_and_plasticity_evidence() -> None:
    result = run_benchmark(seed=7)

    assert result["seed"] == 7
    assert result["replay_equal"] is True
    assert math.isfinite(result["context_separation"])
    assert result["context_separation"] > 0.0
    assert result["positive_margin_delta"] > 0.0
    assert result["negative_margin_delta"] < 0.0
    assert 1 <= result["episode"]["steps"] <= 12
    assert result["episode"]["steps"] == len(result["episode"]["actions"])
    assert set(result["episode"]["terminal"]) == {
        "player_position",
        "enemy_position",
        "player_health",
        "enemy_health",
        "healing_available",
        "done",
    }


def test_benchmark_is_exactly_repeatable_for_same_seed() -> None:
    assert run_benchmark(seed=23) == run_benchmark(seed=23)
