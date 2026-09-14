from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .controller import RecurrentController
from .environment import ToyGame
from .experiment import context_separation, replay_equal, run_episode
from .types import Action, GameObservation


def _observation(
    *,
    enemy_distance: float,
    health: float,
    threat: float,
    healing_distance: float,
) -> GameObservation:
    return GameObservation(
        enemy_distance=enemy_distance,
        enemy_direction=1,
        health=health,
        incoming_threat=threat,
        healing_distance=healing_distance,
        healing_direction=-1,
        left_blocked=False,
        right_blocked=False,
    )


def _trajectory(seed: int, observations: list[GameObservation]) -> np.ndarray:
    controller = RecurrentController(hidden_size=16, seed=seed)
    return np.vstack([controller.step(observation).hidden_state for observation in observations])


def _margin(logits: np.ndarray, action: Action) -> float:
    selected = int(action)
    return float(logits[selected] - np.delete(logits, selected).max())


def _plasticity_delta(seed: int, reward: float) -> float:
    controller = RecurrentController(hidden_size=16, seed=seed, learning_rate=0.2)
    stimulus = _observation(
        enemy_distance=0.1,
        health=0.4,
        threat=1.0,
        healing_distance=0.6,
    )
    before = controller.step(stimulus)
    before_margin = _margin(before.logits, before.action)
    controller.learn(reward)
    controller.reset_state()
    after = controller.step(stimulus)
    return _margin(after.logits, before.action) - before_margin


def run_benchmark(seed: int = 7) -> dict[str, object]:
    approaching = [
        _observation(
            enemy_distance=distance,
            health=0.8,
            threat=threat,
            healing_distance=0.9,
        )
        for distance, threat in ((0.9, 0.0), (0.6, 0.0), (0.3, 0.8))
    ]
    seeking_healing = [
        _observation(
            enemy_distance=0.9,
            health=health,
            threat=0.0,
            healing_distance=distance,
        )
        for health, distance in ((0.4, 0.7), (0.25, 0.4), (0.1, 0.1))
    ]
    separation = context_separation(
        _trajectory(seed, approaching),
        _trajectory(seed, seeking_healing),
    )

    left = run_episode(RecurrentController(seed=seed), ToyGame(), max_steps=12)
    right = run_episode(RecurrentController(seed=seed), ToyGame(), max_steps=12)
    terminal = left.snapshots[-1]

    return {
        "seed": seed,
        "replay_equal": replay_equal(left, right),
        "context_separation": separation,
        "positive_margin_delta": _plasticity_delta(seed, 1.0),
        "negative_margin_delta": _plasticity_delta(seed, -1.0),
        "episode": {
            "steps": len(left.decisions),
            "total_reward": left.total_reward,
            "actions": [decision.action.name for decision in left.decisions],
            "terminal": asdict(terminal),
        },
    }
