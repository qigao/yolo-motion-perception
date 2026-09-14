from __future__ import annotations

import numpy as np

from .types import GameObservation


def encode_observation(observation: GameObservation) -> np.ndarray:
    encoded = np.array(
        [
            observation.enemy_distance,
            observation.enemy_direction,
            observation.health,
            observation.incoming_threat,
            observation.healing_distance,
            observation.healing_direction,
            observation.left_blocked,
            observation.right_blocked,
            1.0,
        ],
        dtype=np.float64,
    )
    encoded.flags.writeable = False
    return encoded
