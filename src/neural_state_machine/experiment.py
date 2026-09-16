from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import RecurrentController
from .environment import ToyGame
from .types import GameSnapshot, NeuralDecision


@dataclass(frozen=True)
class EpisodeTrace:
    decisions: tuple[NeuralDecision, ...]
    rewards: tuple[float, ...]
    snapshots: tuple[GameSnapshot, ...]
    total_reward: float
    done: bool


def run_episode(
    controller: RecurrentController,
    game: ToyGame,
    max_steps: int,
) -> EpisodeTrace:
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    observation = game.reset()
    controller.reset_state()
    decisions: list[NeuralDecision] = []
    rewards: list[float] = []
    snapshots: list[GameSnapshot] = []
    done = False

    for _ in range(max_steps):
        decision = controller.step(observation)
        result = game.step(decision.action)
        controller.learn(result.reward)
        decisions.append(decision)
        rewards.append(result.reward)
        snapshots.append(result.snapshot)
        observation = result.observation
        done = result.done
        if done:
            break

    return EpisodeTrace(
        decisions=tuple(decisions),
        rewards=tuple(rewards),
        snapshots=tuple(snapshots),
        total_reward=float(sum(rewards)),
        done=done,
    )


def replay_equal(left: EpisodeTrace, right: EpisodeTrace) -> bool:
    if (
        left.rewards != right.rewards
        or left.snapshots != right.snapshots
        or left.total_reward != right.total_reward
        or left.done != right.done
        or len(left.decisions) != len(right.decisions)
    ):
        return False

    return all(
        left_decision.action is right_decision.action
        and np.array_equal(left_decision.logits, right_decision.logits)
        and np.array_equal(left_decision.hidden_state, right_decision.hidden_state)
        for left_decision, right_decision in zip(left.decisions, right.decisions, strict=True)
    )


def context_separation(left_states: np.ndarray, right_states: np.ndarray) -> float:
    left = np.asarray(left_states, dtype=np.float64)
    right = np.asarray(right_states, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("state matrices must be rank two")
    if left.shape[0] == 0 or right.shape[0] == 0:
        raise ValueError("state matrices must contain at least one row")
    if left.shape[1] != right.shape[1]:
        raise ValueError("state matrices must have matching widths")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("state matrices must contain only finite values")

    left_centroid = left.mean(axis=0)
    right_centroid = right.mean(axis=0)
    centroid_distance = float(np.linalg.norm(left_centroid - right_centroid))
    squared_deviations = np.concatenate(
        (
            np.sum((left - left_centroid) ** 2, axis=1),
            np.sum((right - right_centroid) ** 2, axis=1),
        )
    )
    pooled_spread = float(np.sqrt(squared_deviations.mean()))
    if pooled_spread == 0.0:
        return centroid_distance
    return centroid_distance / pooled_spread
