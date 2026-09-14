from __future__ import annotations

from .encoding import encode_observation
from .policy import RecurrentPolicy
from .types import Action, GameObservation, NeuralDecision


class RecurrentController:
    def __init__(
        self,
        hidden_size: int = 16,
        seed: int = 0,
        learning_rate: float = 0.1,
    ) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = float(learning_rate)
        self._policy = RecurrentPolicy(
            input_size=9,
            action_count=len(Action),
            hidden_size=hidden_size,
            seed=seed,
            learning_rate=learning_rate,
            recurrent_radius=0.8,
        )

    def step(self, observation: GameObservation) -> NeuralDecision:
        result = self._policy.decide(
            encode_observation(observation), tuple(range(len(Action)))
        )
        return NeuralDecision(
            action=Action(result.action_index),
            logits=result.logits,
            hidden_state=result.hidden_state,
        )

    def learn(self, reward: float) -> None:
        self._policy.learn(reward)

    def reset_state(self) -> None:
        self._policy.reset_state()
