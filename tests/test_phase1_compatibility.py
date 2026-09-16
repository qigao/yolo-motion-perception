import numpy as np
import pytest

from neural_state_machine import GameObservation, RecurrentController
from neural_state_machine.benchmark import run_benchmark


def test_phase_one_seeded_controller_output_is_frozen() -> None:
    observation = GameObservation(
        enemy_distance=0.25,
        enemy_direction=1,
        health=0.8,
        incoming_threat=0.1,
        healing_distance=0.75,
        healing_direction=-1,
        left_blocked=False,
        right_blocked=False,
    )
    decision = RecurrentController(hidden_size=12, seed=41, learning_rate=0.2).step(
        observation
    )

    assert decision.action.name == "ATTACK"
    np.testing.assert_allclose(
        decision.logits,
        [
            0.5458970211878782,
            0.5281858971630009,
            0.8572334978938736,
            -0.7405802043742861,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decision.hidden_state,
        [
            -0.8185519255871977,
            -0.2200558324105483,
            -0.45274824294880944,
            0.8680670805389271,
            -0.8680026124715702,
            0.3918013883639866,
            -0.6692504695763907,
            0.5308668694766017,
            -0.06530864276167365,
            -0.8351635483337463,
            -0.22228422572190384,
            -0.45996089701532794,
        ],
        rtol=0.0,
        atol=1e-12,
    )


def test_phase_one_benchmark_outputs_are_frozen() -> None:
    expected = {
        "seed": 7,
        "replay_equal": True,
        "context_separation": 1.2417694746099712,
        "positive_margin_delta": 0.6795494788487084,
        "negative_margin_delta": -0.6795494788487085,
        "episode": {
            "steps": 5,
            "total_reward": 0.7999999999999999,
            "actions": ["ATTACK", "ATTACK", "ATTACK", "ATTACK", "ATTACK"],
            "terminal": {
                "player_position": 1,
                "enemy_position": 2,
                "player_health": 1,
                "enemy_health": 0,
                "healing_available": True,
                "done": True,
            },
        },
    }
    result = run_benchmark(seed=7)

    assert result["seed"] == expected["seed"]
    assert result["replay_equal"] is expected["replay_equal"]
    assert result["episode"]["steps"] == expected["episode"]["steps"]
    assert result["episode"]["actions"] == expected["episode"]["actions"]
    assert result["episode"]["terminal"] == expected["episode"]["terminal"]
    assert result["context_separation"] == pytest.approx(
        expected["context_separation"], abs=1e-12, rel=0.0
    )
    assert result["positive_margin_delta"] == pytest.approx(
        expected["positive_margin_delta"], abs=1e-12, rel=0.0
    )
    assert result["negative_margin_delta"] == pytest.approx(
        expected["negative_margin_delta"], abs=1e-12, rel=0.0
    )
    assert result["episode"]["total_reward"] == pytest.approx(
        expected["episode"]["total_reward"], abs=1e-12, rel=0.0
    )
