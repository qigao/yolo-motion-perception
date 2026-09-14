from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    ATTACK = 2
    WAIT = 3


@dataclass(frozen=True)
class GameObservation:
    enemy_distance: float
    enemy_direction: int
    health: float
    incoming_threat: float
    healing_distance: float
    healing_direction: int
    left_blocked: bool
    right_blocked: bool

    def __post_init__(self) -> None:
        normalized = {
            "enemy_distance": self.enemy_distance,
            "health": self.health,
            "incoming_threat": self.incoming_threat,
            "healing_distance": self.healing_distance,
        }
        for name, value in normalized.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        directions = {
            "enemy_direction": self.enemy_direction,
            "healing_direction": self.healing_direction,
        }
        for name, value in directions.items():
            if type(value) is not int or value not in (-1, 0, 1):
                raise ValueError(f"{name} must be one of -1, 0, or 1")

        boundaries = {
            "left_blocked": self.left_blocked,
            "right_blocked": self.right_blocked,
        }
        for name, value in boundaries.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class NeuralDecision:
    action: Action
    logits: np.ndarray
    hidden_state: np.ndarray


@dataclass(frozen=True)
class GameConfig:
    arena_size: int = 7
    player_start: int = 1
    enemy_start: int = 5
    healing_start: int = 3
    player_max_health: int = 3
    player_initial_health: int = 2
    enemy_initial_health: int = 2
    attack_damage: int = 1
    enemy_damage: int = 1
    heal_amount: int = 1
    step_penalty: float = -0.01
    invalid_attack_penalty: float = -0.05
    hit_reward: float = 0.2
    kill_reward: float = 1.0
    damage_penalty: float = -0.2
    death_penalty: float = -1.0
    heal_reward: float = 0.1


@dataclass(frozen=True)
class GameSnapshot:
    player_position: int
    enemy_position: int
    player_health: int
    enemy_health: int
    healing_available: bool
    done: bool


@dataclass(frozen=True)
class StepResult:
    observation: GameObservation
    reward: float
    done: bool
    snapshot: GameSnapshot
