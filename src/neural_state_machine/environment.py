from __future__ import annotations

import math

from .types import Action, GameConfig, GameObservation, GameSnapshot, StepResult


def _direction(delta: int) -> int:
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


class ToyGame:
    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self._validate_config()
        self.reset()

    def _validate_config(self) -> None:
        config = self.config
        if type(config.arena_size) is not int or config.arena_size < 3:
            raise ValueError("arena_size must be an integer of at least 3")
        for name in ("player_start", "enemy_start", "healing_start"):
            value = getattr(config, name)
            if type(value) is not int or not 0 <= value < config.arena_size:
                raise ValueError(f"{name} must be a cell inside the arena")
        if config.player_start == config.enemy_start:
            raise ValueError("player and enemy must start in different cells")
        for name in (
            "player_max_health",
            "player_initial_health",
            "enemy_initial_health",
            "attack_damage",
            "enemy_damage",
            "heal_amount",
        ):
            value = getattr(config, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if config.player_initial_health > config.player_max_health:
            raise ValueError("player_initial_health cannot exceed player_max_health")
        for name in (
            "step_penalty",
            "invalid_attack_penalty",
            "hit_reward",
            "kill_reward",
            "damage_penalty",
            "death_penalty",
            "heal_reward",
        ):
            if not math.isfinite(getattr(config, name)):
                raise ValueError(f"{name} must be finite")

    def reset(self) -> GameObservation:
        self._player_position = self.config.player_start
        self._enemy_position = self.config.enemy_start
        self._player_health = self.config.player_initial_health
        self._enemy_health = self.config.enemy_initial_health
        self._healing_available = True
        self._done = False
        return self.observe()

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            player_position=self._player_position,
            enemy_position=self._enemy_position,
            player_health=self._player_health,
            enemy_health=self._enemy_health,
            healing_available=self._healing_available,
            done=self._done,
        )

    def observe(self) -> GameObservation:
        scale = self.config.arena_size - 1
        if self._enemy_health > 0:
            enemy_delta = self._enemy_position - self._player_position
            enemy_distance = abs(enemy_delta) / scale
            enemy_direction = _direction(enemy_delta)
            incoming_threat = 1.0 if abs(enemy_delta) == 1 else 0.0
        else:
            enemy_distance = 1.0
            enemy_direction = 0
            incoming_threat = 0.0

        if self._healing_available:
            healing_delta = self.config.healing_start - self._player_position
            healing_distance = abs(healing_delta) / scale
            healing_direction = _direction(healing_delta)
        else:
            healing_distance = 1.0
            healing_direction = 0

        return GameObservation(
            enemy_distance=enemy_distance,
            enemy_direction=enemy_direction,
            health=self._player_health / self.config.player_max_health,
            incoming_threat=incoming_threat,
            healing_distance=healing_distance,
            healing_direction=healing_direction,
            left_blocked=self._player_position == 0,
            right_blocked=self._player_position == self.config.arena_size - 1,
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("cannot step a terminal game")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")

        reward = self.config.step_penalty
        if action in (Action.MOVE_LEFT, Action.MOVE_RIGHT):
            offset = -1 if action is Action.MOVE_LEFT else 1
            target = max(0, min(self.config.arena_size - 1, self._player_position + offset))
            if self._enemy_health <= 0 or target != self._enemy_position:
                self._player_position = target
            if (
                self._healing_available
                and self._player_position == self.config.healing_start
                and self._player_health < self.config.player_max_health
            ):
                self._player_health = min(
                    self.config.player_max_health,
                    self._player_health + self.config.heal_amount,
                )
                self._healing_available = False
                reward += self.config.heal_reward
        elif action is Action.ATTACK:
            if self._enemy_health > 0 and abs(self._enemy_position - self._player_position) == 1:
                self._enemy_health = max(0, self._enemy_health - self.config.attack_damage)
                if self._enemy_health == 0:
                    reward += self.config.kill_reward
                    self._done = True
                else:
                    reward += self.config.hit_reward
            else:
                reward += self.config.invalid_attack_penalty

        if not self._done and self._enemy_health > 0:
            enemy_delta = self._player_position - self._enemy_position
            if abs(enemy_delta) == 1:
                self._player_health = max(0, self._player_health - self.config.enemy_damage)
                reward += self.config.damage_penalty
                if self._player_health == 0:
                    reward += self.config.death_penalty
                    self._done = True
            elif enemy_delta != 0:
                self._enemy_position += _direction(enemy_delta)

        snapshot = self.snapshot()
        return StepResult(
            observation=self.observe(),
            reward=reward,
            done=self._done,
            snapshot=snapshot,
        )
