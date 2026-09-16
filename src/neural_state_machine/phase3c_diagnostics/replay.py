from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class D0Failure(RuntimeError):
    path: str
    expected: object
    observed: object
    step: int | None = None
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    protocol: object
    scalar_calls: tuple[float, ...]
    action_values: tuple[tuple[float, ...], ...]
    hidden_bytes: tuple[bytes, ...]
    final_parameter_digest: str
    queue_pending_final: int
    metadata: tuple[dict[str, Any], ...]


def run_diagnostic_replay(*args: object, **kwargs: object) -> ReplayResult:
    raise NotImplementedError


def validate_call_stream(expected: tuple[float, ...], observed: tuple[float, ...], *, attempt_id: str) -> None:
    return None


def assert_array_isolation(learner_array: np.ndarray, observer_array: np.ndarray, *, attempt_id: str) -> None:
    return None
